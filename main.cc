#include <deal.II/base/mpi_consensus_algorithms.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>

using namespace dealii;


template <int dim, int spacedim>
LinearAlgebra::distributed::Vector<double>
partition_distributed_triangulation(const Triangulation<dim, spacedim> &tria_in)
{
  const auto tria =
    dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(&tria_in);

  Assert(tria, ExcNotImplemented());

  LinearAlgebra::distributed::Vector<double> partition(
    tria->global_active_cell_index_partitioner().lock());

  const unsigned int n_partitions =
    Utilities::MPI::n_mpi_processes(tria_in.get_communicator());

  for (const auto &cell : tria_in.active_cell_iterators())
    if (cell->is_locally_owned())
      partition[cell->global_active_cell_index()] =
        std::floor(cell->center()[0] * n_partitions);

  partition.update_ghost_values();

  return partition;
}


namespace dealii
{
  namespace TriangulationDescription
  {
    namespace Utilities
    {
      template <int dim, int spacedim>
      struct DescriptionTemp
      {
        template <class Archive>
        void
        serialize(Archive &ar, const unsigned int /*version*/)
        {
          ar &coarse_cells;
          ar &coarse_cell_vertices;
          ar &coarse_cell_index_to_coarse_cell_id;
          ar &cell_infos;
        }

        std::vector<dealii::CellData<dim>> coarse_cells;

        std::vector<std::pair<unsigned int, Point<spacedim>>>
          coarse_cell_vertices;

        std::vector<types::coarse_cell_id> coarse_cell_index_to_coarse_cell_id;

        std::vector<std::vector<CellData<dim>>> cell_infos;
      };

      template <int dim, int spacedim>
      void
      fill_cell_infos(const TriaIterator<CellAccessor<dim, spacedim>> &cell,
                      std::vector<std::vector<CellData<dim>>> &cell_infos)
      {
        cell->set_user_flag();

        CellData<dim> cell_info;

        // save coarse-cell id
        cell_info.id = cell->id().template to_binary<dim>();

        // save boundary_ids of each face of this cell
        for (const auto f : cell->face_indices())
          {
            types::boundary_id boundary_ind = cell->face(f)->boundary_id();
            if (boundary_ind != numbers::internal_face_boundary_id)
              cell_info.boundary_ids.emplace_back(f, boundary_ind);
          }

        // save manifold id
        {
          // ... of cell
          cell_info.manifold_id = cell->manifold_id();

          // ... of lines
          if (dim >= 2)
            for (const auto line : cell->line_indices())
              cell_info.manifold_line_ids[line] =
                cell->line(line)->manifold_id();

          // ... of quads
          if (dim == 3)
            for (const auto f : cell->face_indices())
              cell_info.manifold_quad_ids[f] = cell->quad(f)->manifold_id();
        }

        // subdomain and level subdomain id
        cell_info.subdomain_id       = cell->subdomain_id();
        cell_info.level_subdomain_id = cell->level_subdomain_id();

        cell_infos[cell->level()].emplace_back(cell_info);

        if (cell->level() != 0)
          fill_cell_infos(cell->parent(), cell_infos);
      }

      template <int dim, int spacedim>
      Description<dim, spacedim>
      create_description_from_triangulation(
        const parallel::distributed::Triangulation<dim, spacedim> &tria,
        const LinearAlgebra::distributed::Vector<double> &         partition)
      {
        const auto relevant_processes = [&]() {
          std::set<unsigned int> relevant_processes;

          for (const auto &i : partition)
            relevant_processes.insert(i);

          return std::vector<unsigned int>(relevant_processes.begin(),
                                           relevant_processes.end());
        }();

        std::map<unsigned int, std::vector<unsigned int>>
                                             coinciding_vertex_groups;
        std::map<unsigned int, unsigned int> vertex_to_coinciding_vertex_group;

        GridTools::collect_coinciding_vertices(
          tria, coinciding_vertex_groups, vertex_to_coinciding_vertex_group);

        const auto
          add_vertices_of_cell_to_vertices_owned_by_locally_owned_cells =
            [&coinciding_vertex_groups, &vertex_to_coinciding_vertex_group](
              const auto &       cell,
              std::vector<bool> &vertices_owned_by_locally_owned_cells) {
              for (const auto v : cell->vertex_indices())
                {
                  vertices_owned_by_locally_owned_cells[cell->vertex_index(v)] =
                    true;
                  const auto coinciding_vertex_group =
                    vertex_to_coinciding_vertex_group.find(
                      cell->vertex_index(v));
                  if (coinciding_vertex_group !=
                      vertex_to_coinciding_vertex_group.end())
                    for (const auto &co_vertex : coinciding_vertex_groups.at(
                           coinciding_vertex_group->second))
                      vertices_owned_by_locally_owned_cells[co_vertex] = true;
                }
            };

        std::vector<DescriptionTemp<dim, spacedim>> description_temp(
          relevant_processes.size());

        for (unsigned int i = 0; i < description_temp.size(); ++i)
          {
            // 1) collect locally relevant cells (set user_flag)
            std::vector<bool> old_user_flags;
            tria.save_user_flags(old_user_flags);

            // 1a) clear user_flags
            const_cast<
              dealii::parallel::distributed::Triangulation<dim, spacedim> &>(
              tria)
              .clear_user_flags();

            const unsigned int proc               = relevant_processes[i];
            auto &             description_temp_i = description_temp[i];

            // mark all vertices attached to locally owned cells
            std::vector<bool> vertices_owned_by_locally_owned_cells_on_level(
              tria.n_vertices());
            for (const auto &cell : tria.active_cell_iterators())
              if (cell->is_locally_owned() &&
                  partition[cell->global_active_cell_index()] == proc)
                add_vertices_of_cell_to_vertices_owned_by_locally_owned_cells(
                  cell, vertices_owned_by_locally_owned_cells_on_level);

            const auto is_locally_relevant_on_level = [&](const auto &cell) {
              for (const auto v : cell->vertex_indices())
                if (vertices_owned_by_locally_owned_cells_on_level
                      [cell->vertex_index(v)])
                  return true;
              return false;
            };

            description_temp_i.cell_infos.resize(
              tria.get_triangulation().n_global_levels());

            // collect locally relevant cells (including their parents)
            for (const auto &cell : tria.active_cell_iterators())
              if (is_locally_relevant_on_level(cell))
                fill_cell_infos(cell, description_temp_i.cell_infos);

            // collect coarse-grid cells
            std::vector<bool> vertices_locally_relevant(tria.n_vertices(),
                                                        false);

            for (const auto &cell : tria.cell_iterators_on_level(0))
              {
                if (!cell->user_flag_set())
                  continue;

                // extract cell definition (with old numbering of vertices)
                dealii::CellData<dim> cell_data(cell->n_vertices());
                cell_data.material_id = cell->material_id();
                cell_data.manifold_id = cell->manifold_id();
                for (const auto v : cell->vertex_indices())
                  cell_data.vertices[v] = cell->vertex_index(v);
                description_temp_i.coarse_cells.push_back(cell_data);

                // save indices of each vertex of this cell
                for (const auto v : cell->vertex_indices())
                  vertices_locally_relevant[cell->vertex_index(v)] = true;

                // save translation for corase grid: lid -> gid
                description_temp_i.coarse_cell_index_to_coarse_cell_id
                  .push_back(cell->id().get_coarse_cell_id());
              }

            // collect coarse-grid vertices
            for (unsigned int i = 0; i < vertices_locally_relevant.size(); ++i)
              if (vertices_locally_relevant[i])
                description_temp_i.coarse_cell_vertices.emplace_back(
                  i, tria.get_vertices()[i]);

            const_cast<
              dealii::parallel::distributed::Triangulation<dim, spacedim> &>(
              tria)
              .load_user_flags(old_user_flags);
          }

        DescriptionTemp<dim, spacedim> description_merged;
        description_merged.cell_infos.resize(tria.n_global_levels());

        dealii::Utilities::MPI::ConsensusAlgorithms::AnonymousProcess<char,
                                                                      char>
          process(
            [&]() { return relevant_processes; },
            [&](const unsigned int other_rank, std::vector<char> &send_buffer) {
              const auto ptr = std::find(relevant_processes.begin(),
                                         relevant_processes.end(),
                                         other_rank);

              Assert(ptr != relevant_processes.end(), ExcInternalError());

              const auto other_rank_index =
                std::distance(relevant_processes.begin(), ptr);

              send_buffer =
                dealii::Utilities::pack(description_temp[other_rank_index],
                                        false);
            },
            [&](const unsigned int &     other_rank,
                const std::vector<char> &recv_buffer,
                std::vector<char> &      request_buffer) {
              (void)other_rank;
              (void)request_buffer;

              const auto result =
                dealii::Utilities::unpack<DescriptionTemp<dim, spacedim>>(
                  recv_buffer, false);

              description_merged.coarse_cells.insert(
                description_merged.coarse_cells.end(),
                result.coarse_cells.begin(),
                result.coarse_cells.end());
              description_merged.coarse_cell_vertices.insert(
                description_merged.coarse_cell_vertices.end(),
                result.coarse_cell_vertices.begin(),
                result.coarse_cell_vertices.end());
              description_merged.coarse_cell_index_to_coarse_cell_id.insert(
                description_merged.coarse_cell_index_to_coarse_cell_id.end(),
                result.coarse_cell_index_to_coarse_cell_id.begin(),
                result.coarse_cell_index_to_coarse_cell_id.end());

              for (unsigned int i = 0; i < tria.n_global_levels(); ++i)
                description_merged.cell_infos[i].insert(
                  description_merged.cell_infos[i].end(),
                  result.cell_infos[i].begin(),
                  result.cell_infos[i].end());
            });

        dealii::Utilities::MPI::ConsensusAlgorithms::Selector<char, char>(
          process, tria.get_communicator())
          .run();

        {
          {
            std::vector<std::pair<types::coarse_cell_id, dealii::CellData<dim>>>
              temp;

            for (unsigned int i = 0; i < description_merged.coarse_cells.size();
                 ++i)
              temp.emplace_back(
                description_merged.coarse_cell_index_to_coarse_cell_id[i],
                description_merged.coarse_cells[i]);

            std::sort(temp.begin(),
                      temp.end(),
                      [](const auto &a, const auto &b) {
                        return a.first < b.first;
                      });
            temp.erase(std::unique(temp.begin(),
                                   temp.end(),
                                   [](const auto &a, const auto &b) {
                                     return a.first == b.first;
                                   }),
                       temp.end());

            for (unsigned int i = 0; i < description_merged.coarse_cells.size();
                 ++i)
              {
                description_merged.coarse_cell_index_to_coarse_cell_id[i] =
                  temp[i].first;
                description_merged.coarse_cells[i] = temp[i].second;
              }
          }

          {
            std::sort(description_merged.coarse_cell_vertices.begin(),
                      description_merged.coarse_cell_vertices.end(),
                      [](const auto &a, const auto &b) {
                        return a.first < b.first;
                      });
            description_merged.coarse_cell_vertices.erase(
              std::unique(description_merged.coarse_cell_vertices.begin(),
                          description_merged.coarse_cell_vertices.end(),
                          [](const auto &a, const auto &b) {
                            return a.first == b.first;
                          }),
              description_merged.coarse_cell_vertices.end());
          }

          for (unsigned int i = 0; i < tria.n_global_levels(); ++i)
            {
              std::sort(description_merged.cell_infos[i].begin(),
                        description_merged.cell_infos[i].end(),
                        [](const auto &a, const auto &b) {
                          return a.id < b.id;
                        });
              description_merged.cell_infos[i].erase(
                std::unique(description_merged.cell_infos[i].begin(),
                            description_merged.cell_infos[i].end(),
                            [](const auto &a, const auto &b) {
                              return a.id == b.id;
                            }),
                description_merged.cell_infos[i].end());
            }
        }

        Description<dim, spacedim> description;

        {
          description.comm = tria.get_communicator();

          std::map<unsigned int, unsigned int> map;

          for (unsigned int i = 0;
               i < description_merged.coarse_cell_vertices.size();
               ++i)
            {
              description.coarse_cell_vertices.push_back(
                description_merged.coarse_cell_vertices[i].second);
              map[description_merged.coarse_cell_vertices[i].first] = i;
            }

          description.coarse_cells = description_merged.coarse_cells;

          for (auto &cell : description.coarse_cells)
            for (unsigned int v = 0; v < cell.vertices.size(); ++v)
              cell.vertices[v] = map[cell.vertices[v]];

          description.coarse_cell_index_to_coarse_cell_id =
            description_merged.coarse_cell_index_to_coarse_cell_id;
          description.cell_infos = description_merged.cell_infos;
        }

        return description;
      }
    } // namespace Utilities
  }   // namespace TriangulationDescription
} // namespace dealii


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int dim  = 2;
  const MPI_Comm     comm = MPI_COMM_WORLD;

  parallel::distributed::Triangulation<dim> tria(comm);
  GridGenerator::subdivided_hyper_cube(tria, 4);

  const auto partition_new = partition_distributed_triangulation(tria);

  Vector<double> partition_new_for_data_out(tria.n_active_cells());
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      partition_new_for_data_out[cell->active_cell_index()] =
        partition_new[cell->global_active_cell_index()];

  Vector<double> partition_old(tria.n_active_cells());
  partition_old = Utilities::MPI::this_mpi_process(comm);

  DataOut<dim> data_out;
  data_out.attach_triangulation(tria);
  data_out.add_data_vector(partition_new_for_data_out, "partition_new");
  data_out.add_data_vector(partition_old, "partition_old");
  data_out.build_patches();
  data_out.write_vtu_in_parallel("partition.vtu", comm);

  const auto construction_data =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      tria, partition_new);

  parallel::fullydistributed::Triangulation<dim> tria_pft(comm);
  tria_pft.create_triangulation(construction_data);
}
