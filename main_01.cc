#include <deal.II/base/mpi_consensus_algorithms.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
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


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int dim  = 2;
  const MPI_Comm     comm = MPI_COMM_WORLD;

  parallel::distributed::Triangulation<dim> tria(comm);
  GridGenerator::subdivided_hyper_cube(tria, 4);
  tria.refine_global(3);

  const auto partition_new = partition_distributed_triangulation(tria);

  {
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
    data_out.write_vtu_in_parallel("partition.0.vtu", comm);

    GridOut grid_out;
    grid_out.write_mesh_per_processor_as_vtu(tria, "grid_old");
  }

  const auto construction_data =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      tria, partition_new);

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
