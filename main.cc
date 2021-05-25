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
      Description<dim, spacedim>
      create_description_from_triangulation(
        const Triangulation<dim, spacedim> &              tria,
        const LinearAlgebra::distributed::Vector<double> &partition)
      {
        (void)tria;
        (void)partition;

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

        // helper function, which collects all vertices belonging to a cell
        // (also taking into account periodicity)
        const auto
          add_vertices_of_cell_to_vertices_owned_by_locally_owned_cells =
            [&coinciding_vertex_groups, &vertex_to_coinciding_vertex_group](
              const auto &       cell,
              std::vector<bool> &vertices_owned_by_locally_owned_cells) {
              // add local vertices
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

        for (const auto &proc : relevant_processes)
          {
            std::cout << proc << std::endl;

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

            for (const auto &cell : tria.active_cell_iterators())
              if (is_locally_relevant_on_level(cell))
                std::cout << cell->id() << std::endl;
          }



        Description<dim, spacedim> description;

        return description;
      }
    } // namespace Utilities
  }   // namespace TriangulationDescription
} // namespace dealii


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int dim = 2;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_cube(tria, 4);

  const auto partition_new = partition_distributed_triangulation(tria);

  Vector<double> partition_new_for_data_out(tria.n_active_cells());
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      partition_new_for_data_out[cell->active_cell_index()] =
        partition_new[cell->global_active_cell_index()];

  Vector<double> partition_old(tria.n_active_cells());
  partition_old = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  DataOut<dim> data_out;
  data_out.attach_triangulation(tria);
  data_out.add_data_vector(partition_new_for_data_out, "partition_new");
  data_out.add_data_vector(partition_old, "partition_old");
  data_out.build_patches();
  data_out.write_vtu_in_parallel("partition.vtu", MPI_COMM_WORLD);

  const auto description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      tria, partition_new);
}
