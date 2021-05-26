#include <deal.II/base/mpi_consensus_algorithms.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.templates.h>

#include <deal.II/numerics/data_out.h>

using namespace dealii;

template <typename MeshType>
unsigned int
n_coarse_cells(const MeshType &mesh)
{
  types::coarse_cell_id n_coarse_cells = 0;

  for (const auto &cell : mesh.get_triangulation().active_cell_iterators())
    if (!cell->is_artificial())
      n_coarse_cells =
        std::max(n_coarse_cells, cell->id().get_coarse_cell_id());

  return Utilities::MPI::max(n_coarse_cells, mesh.get_communicator()) + 1;
}

template <typename MeshType>
unsigned int
n_global_levels(const MeshType &mesh)
{
  return mesh.get_triangulation().n_global_levels();
}


template <int dim, int spacedim>
void
add_indices(const TriaIterator<CellAccessor<dim, spacedim>> &cell,
            const internal::CellIDTranslator<dim> &          cell_id_translator,
            IndexSet &                                       is_fine)
{
  is_fine.add_index(cell_id_translator.translate(cell));

  if (cell->level() > 0 &&
      (cell->index() % Utilities::pow<unsigned int>(2, dim)) == 0)
    add_indices(cell->parent(), cell_id_translator, is_fine);
}



template <int dim, int spacedim>
LinearAlgebra::distributed::Vector<double>
partition_distributed_triangulation(
  const Triangulation<dim, spacedim> &tria_fine,
  const Triangulation<dim, spacedim> &tria_coarse_in)
{
  const auto communicator = tria_coarse_in.get_communicator();

  const internal::CellIDTranslator<dim> cell_id_translator(
    n_coarse_cells(tria_fine), n_global_levels(tria_fine));

  IndexSet is_fine(cell_id_translator.size());
  IndexSet is_coarse(cell_id_translator.size());

  for (const auto &cell : tria_fine.active_cell_iterators())
    if (cell->is_locally_owned())
      add_indices(cell, cell_id_translator, is_fine);

  for (const auto &cell : tria_coarse_in.active_cell_iterators())
    if (cell->is_locally_owned())
      is_coarse.add_index(cell_id_translator.translate(cell));

  std::vector<unsigned int> owning_ranks_of_ghosts(is_coarse.n_elements());
  {
    Utilities::MPI::internal::ComputeIndexOwner::ConsensusAlgorithmsPayload
      process(is_fine, is_coarse, communicator, owning_ranks_of_ghosts, false);

    Utilities::MPI::ConsensusAlgorithms::Selector<
      std::pair<types::global_cell_index, types::global_cell_index>,
      unsigned int>
      consensus_algorithm(process, communicator);
    consensus_algorithm.run();
  }

  const auto tria =
    dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
      &tria_coarse_in);

  Assert(tria, ExcNotImplemented());

  LinearAlgebra::distributed::Vector<double> partition(
    tria->global_active_cell_index_partitioner().lock());

  for (const auto &cell : tria_coarse_in.active_cell_iterators())
    if (cell->is_locally_owned())
      partition[cell->global_active_cell_index()] =
        owning_ranks_of_ghosts[is_coarse.index_within_set(
          cell_id_translator.translate(cell))];

  partition.update_ghost_values();

  return partition;
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int dim  = 2;
  const MPI_Comm     comm = MPI_COMM_WORLD;

  parallel::distributed::Triangulation<dim> tria_fine(comm);
  parallel::distributed::Triangulation<dim> tria_coarse(comm);

  const auto create_fine_mesh = [](auto &tria) {
    const unsigned int n_ref_global = 3;
    const unsigned int n_ref_local  = 1;
    GridGenerator::hyper_cube(tria, -1.0, +1.0);
    tria.refine_global(n_ref_global);

    for (unsigned int i = 0; i < n_ref_local; ++i)
      {
        for (auto cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              bool flag = true;
              for (unsigned int d = 0; d < dim; d++)
                if (cell->center()[d] > 0.0)
                  flag = false;
              if (flag)
                cell->set_refine_flag();
            }
        tria.execute_coarsening_and_refinement();
      }
  };

  create_fine_mesh(tria_fine);
  create_fine_mesh(tria_coarse);
  tria_coarse.coarsen_global();

  const auto partition_new =
    partition_distributed_triangulation(tria_fine, tria_coarse);

  {
    GridOut grid_out;
    grid_out.write_mesh_per_processor_as_vtu(tria_fine, "grid_fine");
    grid_out.write_mesh_per_processor_as_vtu(tria_coarse, "grid_coarse");
  }

  {
    Vector<double> partition_new_for_data_out(tria_coarse.n_active_cells());
    for (const auto &cell : tria_coarse.active_cell_iterators())
      if (cell->is_locally_owned())
        partition_new_for_data_out[cell->active_cell_index()] =
          partition_new[cell->global_active_cell_index()];

    Vector<double> partition_old(tria_coarse.n_active_cells());
    partition_old = Utilities::MPI::this_mpi_process(comm);

    DataOut<dim> data_out;
    data_out.attach_triangulation(tria_coarse);
    data_out.add_data_vector(partition_new_for_data_out, "partition_new");
    data_out.add_data_vector(partition_old, "partition_old");
    data_out.build_patches();
    data_out.write_vtu_in_parallel("partition.0.vtu", comm);
  }

  const auto construction_data =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      tria_coarse, partition_new);

  parallel::fullydistributed::Triangulation<dim> tria_pft(comm);
  tria_pft.create_triangulation(construction_data);

  {
    Vector<double> partition_old(tria_pft.n_active_cells());
    partition_old = Utilities::MPI::this_mpi_process(comm);

    DataOut<dim> data_out;
    data_out.attach_triangulation(tria_pft);
    data_out.add_data_vector(partition_old, "partition_old");
    data_out.build_patches();
    data_out.write_vtu_in_parallel("partition.1.vtu", comm);

    GridOut grid_out;
    grid_out.write_mesh_per_processor_as_vtu(tria_pft, "grid_new");
  }
}
