using namespace dealii;



namespace dealii
{
  namespace RepartitioningPolicyTools
  {
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
                const internal::CellIDTranslator<dim> &cell_id_translator,
                IndexSet &                             is_fine)
    {
      is_fine.add_index(cell_id_translator.translate(cell));

      if (cell->level() > 0 &&
          (cell->index() % Utilities::pow<unsigned int>(2, dim)) == 0)
        add_indices(cell->parent(), cell_id_translator, is_fine);
    }

    template <int dim, int spacedim = dim>
    class Base
    {
    public:
      virtual LinearAlgebra::distributed::Vector<double>
      partition(const Triangulation<dim, spacedim> &tria_coarse_in) const = 0;
    };

    template <int dim, int spacedim = dim>
    class DefaultPolicy : public Base<dim, spacedim>
    {
    public:
      virtual LinearAlgebra::distributed::Vector<double>
      partition(
        const Triangulation<dim, spacedim> &tria_coarse_in) const override
      {
        (void)tria_coarse_in; // nothing to do

        return {};
      }
    };

    template <int dim, int spacedim = dim>
    class FirstChildPolicy : public Base<dim, spacedim>
    {
    public:
      FirstChildPolicy(const Triangulation<dim, spacedim> &tria_fine)
        : cell_id_translator(n_coarse_cells(tria_fine),
                             n_global_levels(tria_fine))
        , is_fine(cell_id_translator.size())
      {
        for (const auto &cell : tria_fine.active_cell_iterators())
          if (cell->is_locally_owned())
            add_indices(cell, cell_id_translator, is_fine);
      }

      virtual LinearAlgebra::distributed::Vector<double>
      partition(
        const Triangulation<dim, spacedim> &tria_coarse_in) const override
      {
        const auto communicator = tria_coarse_in.get_communicator();

        IndexSet is_coarse(cell_id_translator.size());

        for (const auto &cell : tria_coarse_in.active_cell_iterators())
          if (cell->is_locally_owned())
            is_coarse.add_index(cell_id_translator.translate(cell));

        std::vector<unsigned int> owning_ranks_of_ghosts(
          is_coarse.n_elements());
        {
          Utilities::MPI::internal::ComputeIndexOwner::
            ConsensusAlgorithmsPayload process(
              is_fine, is_coarse, communicator, owning_ranks_of_ghosts, false);

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

    private:
      const internal::CellIDTranslator<dim> cell_id_translator;
      IndexSet                              is_fine;
    };

    template <int dim, int spacedim = dim>
    class MinimalGranularityPolicy : public Base<dim, spacedim>
    {
    public:
      MinimalGranularityPolicy(const unsigned int n_min_cells)
        : n_min_cells(n_min_cells)
      {}

      virtual LinearAlgebra::distributed::Vector<double>
      partition(const Triangulation<dim, spacedim> &tria_in) const override
      {
        const auto tria =
          dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
            &tria_in);

        Assert(tria, ExcNotImplemented());

        LinearAlgebra::distributed::Vector<double> partition(
          tria->global_active_cell_index_partitioner().lock());

        unsigned int n_locally_owned_active_cells = 0;
        for (const auto &cell : tria_in.active_cell_iterators())
          if (cell->is_locally_owned())
            ++n_locally_owned_active_cells;

        if (Utilities::MPI::min(n_locally_owned_active_cells,
                                tria_in.get_communicator()) > n_min_cells)
          return {};

        const unsigned int n_global_active_cells =
          tria_in.n_global_active_cells();

        const unsigned int n_partitions =
          std::min(n_global_active_cells / n_min_cells + 1,
                   Utilities::MPI::n_mpi_processes(tria_in.get_communicator()));

        for (const auto &cell : tria_in.active_cell_iterators())
          if (cell->is_locally_owned())
            partition[cell->global_active_cell_index()] =
              cell->global_active_cell_index() * n_partitions /
              n_global_active_cells;

        partition.update_ghost_values();

        return partition;
      }

    private:
      const unsigned int n_min_cells;
    };

  } // namespace RepartitioningPolicyTools


  namespace MGTransferGlobalCoarseningTools
  {
    template <int dim, int spacedim>
    std::vector<std::shared_ptr<const Triangulation<dim, spacedim>>>
    create_geometric_coarsening_sequence(
      const Triangulation<dim, spacedim> &fine_triangulation_in,
      const RepartitioningPolicyTools::Base<dim, spacedim> &policy)
    {
      std::vector<std::shared_ptr<const Triangulation<dim, spacedim>>>
        coarse_grid_triangulations(fine_triangulation_in.n_global_levels());

      coarse_grid_triangulations.back().reset(&fine_triangulation_in,
                                              [](auto *) {
                                                // empty deleter, since
                                                // fine_triangulation_in is an
                                                // external field and its
                                                // destructor is called
                                                // somewhere else
                                              });

      // for a single level nothing has to be done
      if (fine_triangulation_in.n_global_levels() == 1)
        return coarse_grid_triangulations;

#ifndef DEAL_II_WITH_P4EST
      Assert(false, ExcNotImplemented());
#else
      const auto fine_triangulation = dynamic_cast<
        const parallel::distributed::Triangulation<dim, spacedim> *>(
        &fine_triangulation_in);

      Assert(fine_triangulation, ExcNotImplemented());

      parallel::distributed::Triangulation<dim, spacedim> temp_tria(
        fine_triangulation->get_communicator(),
        fine_triangulation->get_mesh_smoothing());

      temp_tria.copy_triangulation(*fine_triangulation);

      const unsigned int max_level = fine_triangulation->n_global_levels() - 1;

      // create coarse meshes
      for (unsigned int l = max_level; l > 0; --l)
        {
          // coarsen mesh
          temp_tria.coarsen_global();

          const auto comm      = temp_tria.get_communicator();
          const auto partition = policy.partition(temp_tria);

          partition.update_ghost_values();

          const auto construction_data =
            partition.size() == 0 ?
              TriangulationDescription::Utilities::
                create_description_from_triangulation(temp_tria, comm) :
              TriangulationDescription::Utilities::
                create_description_from_triangulation(temp_tria, partition);

          const auto new_tria =
            std::make_shared<parallel::fullydistributed::Triangulation<dim>>(
              comm);
          new_tria->create_triangulation(construction_data);

          // save mesh
          coarse_grid_triangulations[l - 1] = new_tria;

          MPI_Barrier(MPI_COMM_WORLD);
        }
#endif

      return coarse_grid_triangulations;
    }
  } // namespace MGTransferGlobalCoarseningTools
} // namespace dealii
