#include <deal.II/base/mpi_consensus_algorithms.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <deal.II/numerics/data_out.h>

#include "util.h"

using namespace dealii;

template <int dim, int spacedim>
void
create_triangulation(Triangulation<dim, spacedim> &tria)
{
  const unsigned int n_ref_global = 3;
  const unsigned int n_ref_local  = 3;
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
}

template <int dim, int spacedim>
void
output_grid(
  const std::vector<std::shared_ptr<const Triangulation<dim, spacedim>>> &trias,
  const std::string &                                                     label)
{
  const auto comm    = trias.front()->get_communicator();
  const auto my_rank = Utilities::MPI::this_mpi_process(comm);

  for (unsigned int i = 0; i < trias.size(); ++i)
    {
      const auto &tria = *trias[i];

      Vector<double> ranks(tria.n_active_cells());
      ranks = my_rank;

      DataOut<dim> data_out;
      data_out.attach_triangulation(tria);
      data_out.add_data_vector(ranks, "ranks");
      data_out.build_patches();

      data_out.write_vtu_with_pvtu_record("./", label, i, comm, 2, 0);
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int dim  = 2;
  const MPI_Comm     comm = MPI_COMM_WORLD;

  parallel::distributed::Triangulation<dim> tria(comm);
  create_triangulation(tria);

  const auto test = [&](const auto &policy, const std::string &label) {
    const auto trias =
      MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
        tria, policy);
    output_grid(trias, label);
  };

  // default policy (simply copy p:d:T)
  test(RepartitioningPolicyTools::DefaultPolicy<dim>(), "grid_policy_default");

  // first-child policy
  test(RepartitioningPolicyTools::FirstChildPolicy<dim>(tria),
       "grid_policy_first");

  // first-child policy
  test(RepartitioningPolicyTools::MinimalGranularityPolicy<dim>(4),
       "grid_policy_minimal");
}
