module ice_model_utilities
  !< Generally useful functions used by the ice model.

  use mpi
  use mpi_basic, only: par
  use precisions, only: dp
  use control_resources_and_error_messaging, only: init_routine, finalise_routine, crash, colour_string
  use model_configuration, only: C
  use parameters, only: ice_density, seawater_density
  use mesh_types, only: type_mesh
  use ice_model_types, only: type_ice_model
  use reference_geometry_types, only: type_reference_geometry
  use CSR_sparse_matrix_type, only: type_sparse_matrix_CSR_dp
  use remapping_main, only: Atlas
  use SMB_model_types, only: type_SMB_model
  use BMB_model_types, only: type_BMB_model
  use LMB_model_types, only: type_LMB_model
  use AMB_model_types, only: type_AMB_model
  use netcdf_io_main
  use mesh_utilities, only: interpolate_to_point_dp_2D, extrapolate_Gaussian
  use mesh_ROI_polygons, only: calc_polygon_Patagonia
  use plane_geometry, only: is_in_polygon, triangle_area
  use mpi_distributed_memory, only: gather_to_all
  use ice_geometry_basics, only: is_floating
  use projections, only: oblique_sg_projection
  use mesh_disc_apply_operators, only: ddx_a_a_2D, ddy_a_a_2D, map_a_b_2D, ddx_a_b_2D, ddy_a_b_2D, &
    ddx_b_a_2D, ddy_b_a_2D
  use create_maps_grid_mesh, only: create_map_from_xy_grid_to_mesh, create_map_from_xy_grid_to_mesh_triangles
  use mpi_distributed_memory_grid, only: gather_gridded_data_to_master
  use bedrock_cumulative_density_functions
  use subgrid_grounded_fractions_main

  implicit none

contains

! == Masks
! ========

  subroutine determine_masks( mesh, ice)
    !< Determine the different masks

    ! mask_icefree_land       ! T: ice-free land , F: otherwise
    ! mask_icefree_ocean      ! T: ice-free ocean, F: otherwise
    ! mask_grounded_ice       ! T: grounded ice  , F: otherwise
    ! mask_floating_ice       ! T: floating ice  , F: otherwise
    ! mask_icefree_land_prev  ! T: ice-free land , F: otherwise (during previous time step)
    ! mask_icefree_ocean_prev ! T: ice-free ocean, F: otherwise (during previous time step)
    ! mask_grounded_ice_prev  ! T: grounded ice  , F: otherwise (during previous time step)
    ! mask_floating_ice_prev  ! T: floating ice  , F: otherwise (during previous time step)
    ! mask_margin             ! T: ice next to ice-free, F: otherwise
    ! mask_gl_gr              ! T: grounded ice next to floating ice, F: otherwise
    ! mask_gl_fl              ! T: floating ice next to grounded ice, F: otherwise
    ! mask_cf_gr              ! T: grounded ice next to ice-free water (sea or lake), F: otherwise
    ! mask_cf_fl              ! T: floating ice next to ice-free water (sea or lake), F: otherwise
    ! mask_coastline          ! T: ice-free land next to ice-free ocean, F: otherwise

    ! In- and output variables
    type(type_mesh),      intent(in   ) :: mesh
    type(type_ice_model), intent(inout) :: ice

    ! Local variables:
    character(len=1024), parameter :: routine_name = 'determine_masks'
    integer                        :: vi, ci, vj
    logical, dimension(mesh%nV)    :: mask_icefree_land_tot
    logical, dimension(mesh%nV)    :: mask_icefree_ocean_tot
    logical, dimension(mesh%nV)    :: mask_grounded_ice_tot
    logical, dimension(mesh%nV)    :: mask_floating_ice_tot

    ! Add routine to path
    call init_routine( routine_name)

    ! === Basic masks ===
    ! ===================

    ! Store previous basic masks
    ice%mask_icefree_land_prev  = ice%mask_icefree_land
    ice%mask_icefree_ocean_prev = ice%mask_icefree_ocean
    ice%mask_grounded_ice_prev  = ice%mask_grounded_ice
    ice%mask_floating_ice_prev  = ice%mask_floating_ice

    ! Initialise basic masks
    ice%mask_icefree_land  = .false.
    ice%mask_icefree_ocean = .false.
    ice%mask_grounded_ice  = .false.
    ice%mask_floating_ice  = .false.
    ice%mask               = 0

    ! Calculate
    do vi = mesh%vi1, mesh%vi2

      if (is_floating( ice%Hi( vi), ice%Hb( vi), ice%SL( vi))) then
        ! Ice thickness is below the floatation thickness; either floating ice, or ice-free ocean

        if (ice%Hi( vi) > 0._dp) then
          ! Floating ice

          ice%mask_floating_ice( vi) = .true.
          ice%mask( vi) = C%type_floating_ice

        else
          ! Ice-free ocean

          ice%mask_icefree_ocean( vi) = .true.
          ice%mask( vi) = C%type_icefree_ocean

        end if

      else
        ! Ice thickness is above the floatation thickness; either grounded ice, or ice-free land

        if (ice%Hi( vi) > 0._dp) then
          ! Grounded ice

          ice%mask_grounded_ice( vi) = .true.
          ice%mask( vi) = C%type_grounded_ice

        else
          ! Ice-free land

          ice%mask_icefree_land( vi) = .true.
          ice%mask( vi) = C%type_icefree_land

        end if

      end if

    end do

    ! === Transitional masks ===
    ! ==========================

    ! Gather basic masks to all processes
    call gather_to_all( ice%mask_icefree_land , mask_icefree_land_tot )
    call gather_to_all( ice%mask_icefree_ocean, mask_icefree_ocean_tot)
    call gather_to_all( ice%mask_grounded_ice , mask_grounded_ice_tot )
    call gather_to_all( ice%mask_floating_ice , mask_floating_ice_tot )

    ! Initialise transitional masks
    ice%mask_margin    = .false.
    ice%mask_gl_gr     = .false.
    ice%mask_gl_fl     = .false.
    ice%mask_cf_gr     = .false.
    ice%mask_cf_fl     = .false.
    ice%mask_coastline = .false.

    do vi = mesh%vi1, mesh%vi2

      ! Ice margin
      if (mask_grounded_ice_tot( vi) .OR. mask_floating_ice_tot( vi)) then
        do ci = 1, mesh%nC( vi)
          vj = mesh%C( vi,ci)
          if (.not. (mask_grounded_ice_tot( vj) .OR. mask_floating_ice_tot( vj))) then
            ice%mask_margin( vi) = .true.
            ice%mask( vi) = C%type_margin
          end if
        end do
      end if

      ! Grounding line (grounded side)
      if (mask_grounded_ice_tot( vi)) then
        do ci = 1, mesh%nC( vi)
          vj = mesh%C( vi,ci)
          if (mask_floating_ice_tot( vj)) then
            ice%mask_gl_gr( vi) = .true.
            ice%mask( vi) = C%type_groundingline_gr
          end if
        end do
      end if

      ! Grounding line (floating side)
      if (mask_floating_ice_tot( vi)) then
        do ci = 1, mesh%nC( vi)
          vj = mesh%C( vi,ci)
          if (mask_grounded_ice_tot( vj)) then
            ice%mask_gl_fl( vi) = .true.
            ice%mask( vi) = C%type_groundingline_fl
          end if
        end do
      end if

      ! Calving front (grounded)
      if (mask_grounded_ice_tot( vi)) then
        do ci = 1, mesh%nC(vi)
          vj = mesh%C( vi,ci)
          if (mask_icefree_ocean_tot( vj)) then
            ice%mask_cf_gr( vi) = .true.
            ice%mask( vi) = C%type_calvingfront_gr
          end if
        end do
      end if

      ! Calving front (floating)
      if (mask_floating_ice_tot( vi)) then
        do ci = 1, mesh%nC(vi)
          vj = mesh%C( vi,ci)
          if (mask_icefree_ocean_tot( vj)) then
            ice%mask_cf_fl( vi) = .true.
            ice%mask( vi) = C%type_calvingfront_fl
          end if
        end do
      end if

      ! Coastline
      if (mask_icefree_land_tot( vi)) then
        do ci = 1, mesh%nC(vi)
          vj = mesh%C( vi,ci)
          if (mask_icefree_ocean_tot( vj)) then
            ice%mask_coastline( vi) = .true.
            ice%mask( vi) = C%type_coastline
          end if
        end do
      end if

    end do ! do vi = mesh%vi1, mesh%vi2

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine determine_masks

! == Effective ice thickness
! ==========================

  subroutine calc_effective_thickness( mesh, ice, Hi, Hi_eff, fraction_margin)
    !< Determine the ice-filled fraction and effective ice thickness of floating margin pixels

    ! In- and output variables
    type(type_mesh),      intent(in   )                 :: mesh
    type(type_ice_model), intent(in   )                 :: ice
    real(dp), dimension(mesh%vi1:mesh%vi2), intent(in)  :: Hi
    real(dp), dimension(mesh%vi1:mesh%vi2), intent(out) :: Hi_eff
    real(dp), dimension(mesh%vi1:mesh%vi2), intent(out) :: fraction_margin

    ! Local variables:
    character(len=1024), parameter         :: routine_name = 'calc_effective_thickness'
    integer                                :: vi, ci, vc
    real(dp)                               :: Hi_neighbour_max
    real(dp), dimension(mesh%nV)           :: Hi_tot, Hb_tot, SL_tot
    logical,  dimension(mesh%vi1:mesh%vi2) :: mask_margin, mask_floating
    logical,  dimension(mesh%nV)           :: mask_margin_tot, mask_floating_tot

    ! == Initialisation
    ! =================

    ! Add routine to path
    call init_routine( routine_name)

    ! Collect Hi from all processes
    call gather_to_all( Hi,     Hi_tot)
    call gather_to_all( ice%Hb, Hb_tot)
    call gather_to_all( ice%SL, SL_tot)

    ! == Margin mask
    ! ==============

    ! Initialise
    mask_margin = .false.

    do vi = mesh%vi1, mesh%vi2
      do ci = 1, mesh%nC( vi)
        vc = mesh%C( vi,ci)
        if (Hi_tot( vi) > 0._dp .and. Hi_tot( vc) == 0._dp) then
          mask_margin( vi) = .true.
        end if
      end do
    end do

    ! Gather mask values from all processes
    call gather_to_all( mask_margin, mask_margin_tot)

    ! == Floating mask
    ! ================

    ! Initialise
    mask_floating = .false.

    do vi = mesh%vi1, mesh%vi2
      if (is_floating( Hi_tot( vi), ice%Hb( vi), ice%SL( vi))) then
        mask_floating( vi) = .true.
      end if
    end do

    ! Gather mask values from all processes
    call gather_to_all( mask_floating, mask_floating_tot)

    ! == default values
    ! =================

    ! Initialise values
    do vi = mesh%vi1, mesh%vi2
      ! Grounded regions: ice-free land and grounded ice
      ! NOTE: important to let ice-free land have a non-zero
      ! margin fraction to let it get SMB in the ice thickness equation
      if (.not. mask_floating( vi)) then
        fraction_margin( vi) = 1._dp
        Hi_eff( vi) = Hi_tot( vi)
      ! Old and new floating regions
      elseif (Hi_tot( vi) > 0._dp) then
        fraction_margin( vi) = 1._dp
        Hi_eff( vi) = Hi_tot( vi)
      ! New ice-free ocean regions
      else
        fraction_margin( vi) = 0._dp
        Hi_eff( vi) = 0._dp
      end if
    end do

    ! === Compute ===
    ! ===============

    do vi = mesh%vi1, mesh%vi2

      ! Only check margin vertices
      if (.not. mask_margin_tot( vi)) then
        ! Simply use initialised values
        cycle
      end if

      ! === Max neighbour thickness ===
      ! ===============================

      ! Find the max ice thickness among non-margin neighbours
      Hi_neighbour_max = 0._dp

      do ci = 1, mesh%nC( vi)
        vc = mesh%C( vi,ci)

        ! Ignore margin neighbours
        if (mask_margin_tot( vc)) then
          cycle
        end if

        ! Floating margins check for neighbours
        if (mask_floating( vi)) then
          Hi_neighbour_max = max( Hi_neighbour_max, Hi_tot( vc))
        end if

      end do

      ! === Effective ice thickness ===
      ! ===============================

      ! Only apply if the thickest non-margin neighbour is thicker than
      ! this vertex. Otherwise, simply use initialised values.
      if (Hi_neighbour_max > Hi_tot( vi)) then
        ! Calculate sub-grid ice-filled fraction
        Hi_eff( vi) = Hi_neighbour_max
        fraction_margin( vi) = Hi_tot( vi) / Hi_neighbour_max
      end if

    end do

    ! === Finalisation ===
    ! ====================

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine calc_effective_thickness

! == Zeta gradients
! =================

  subroutine calc_zeta_gradients( mesh, ice)
    !< Calculate all the gradients of zeta,
    !< needed to perform the scaled vertical coordinate transformation

    ! In/output variables:
    type(type_mesh),      intent(in   ) :: mesh
    type(type_ice_model), intent(inout) :: ice

    ! Local variables:
    character(len=1024), parameter    :: routine_name = 'calc_zeta_gradients'
    ! real(dp), dimension(:), pointer :: Hi_a
    real(dp), dimension(:), pointer   :: dHi_dx_a
    real(dp), dimension(:), pointer   :: dHi_dy_a
    real(dp), dimension(:), pointer   :: d2Hi_dx2_a
    real(dp), dimension(:), pointer   :: d2Hi_dxdy_a
    real(dp), dimension(:), pointer   :: d2Hi_dy2_a
    real(dp), dimension(:), pointer   :: Hi_b
    real(dp), dimension(:), pointer   :: dHi_dx_b
    real(dp), dimension(:), pointer   :: dHi_dy_b
    real(dp), dimension(:), pointer   :: d2Hi_dx2_b
    real(dp), dimension(:), pointer   :: d2Hi_dxdy_b
    real(dp), dimension(:), pointer   :: d2Hi_dy2_b
    real(dp), dimension(:), pointer   :: Hs_b
    ! real(dp), dimension(:), pointer :: Hs_a
    real(dp), dimension(:), pointer   :: dHs_dx_a
    real(dp), dimension(:), pointer   :: dHs_dy_a
    real(dp), dimension(:), pointer   :: d2Hs_dx2_a
    real(dp), dimension(:), pointer   :: d2Hs_dxdy_a
    real(dp), dimension(:), pointer   :: d2Hs_dy2_a
    real(dp), dimension(:), pointer   :: dHs_dx_b
    real(dp), dimension(:), pointer   :: dHs_dy_b
    real(dp), dimension(:), pointer   :: d2Hs_dx2_b
    real(dp), dimension(:), pointer   :: d2Hs_dxdy_b
    real(dp), dimension(:), pointer   :: d2Hs_dy2_b
    integer                           :: vi,ti,k,ks
    real(dp)                          :: Hi, zeta

    ! Add routine to path
    call init_routine( routine_name)

    ! allocate shared memory
    ! allocate( Hi_a(        mesh%vi1:mesh%vi2))
    allocate( dHi_dx_a(    mesh%vi1:mesh%vi2))
    allocate( dHi_dy_a(    mesh%vi1:mesh%vi2))
    allocate( d2Hi_dx2_a(  mesh%vi1:mesh%vi2))
    allocate( d2Hi_dxdy_a( mesh%vi1:mesh%vi2))
    allocate( d2Hi_dy2_a(  mesh%vi1:mesh%vi2))

    allocate( Hi_b(        mesh%ti1:mesh%ti2))
    allocate( dHi_dx_b(    mesh%ti1:mesh%ti2))
    allocate( dHi_dy_b(    mesh%ti1:mesh%ti2))
    allocate( d2Hi_dx2_b(  mesh%ti1:mesh%ti2))
    allocate( d2Hi_dxdy_b( mesh%ti1:mesh%ti2))
    allocate( d2Hi_dy2_b(  mesh%ti1:mesh%ti2))

    ! allocate( Hs_a(        mesh%vi1:mesh%vi2))
    allocate( dHs_dx_a(    mesh%vi1:mesh%vi2))
    allocate( dHs_dy_a(    mesh%vi1:mesh%vi2))
    allocate( d2Hs_dx2_a(  mesh%vi1:mesh%vi2))
    allocate( d2Hs_dxdy_a( mesh%vi1:mesh%vi2))
    allocate( d2Hs_dy2_a(  mesh%vi1:mesh%vi2))

    allocate( Hs_b(        mesh%ti1:mesh%ti2))
    allocate( dHs_dx_b(    mesh%ti1:mesh%ti2))
    allocate( dHs_dy_b(    mesh%ti1:mesh%ti2))
    allocate( d2Hs_dx2_b(  mesh%ti1:mesh%ti2))
    allocate( d2Hs_dxdy_b( mesh%ti1:mesh%ti2))
    allocate( d2Hs_dy2_b(  mesh%ti1:mesh%ti2))

    ! Calculate gradients of Hi and Hs on both grids

    ! call map_a_a_2D( mesh, ice%Hi_a, Hi_a       )
    call ddx_a_a_2D( mesh, ice%Hi  , dHi_dx_a   )
    call ddy_a_a_2D( mesh, ice%Hi  , dHi_dy_a   )
    call map_a_b_2D( mesh, ice%Hi  , Hi_b       )
    call ddx_a_b_2D( mesh, ice%Hi  , dHi_dx_b   )
    call ddy_a_b_2D( mesh, ice%Hi  , dHi_dy_b   )
    call ddx_b_a_2D( mesh, dHi_dx_b, d2Hi_dx2_a )
    call ddy_b_a_2D( mesh, dHi_dx_b, d2Hi_dxdy_a)
    call ddy_b_a_2D( mesh, dHi_dy_b, d2Hi_dy2_a )
    call ddx_a_b_2D( mesh, dHi_dx_a, d2Hi_dx2_b )
    call ddy_a_b_2D( mesh, dHi_dx_a, d2Hi_dxdy_b)
    call ddy_a_b_2D( mesh, dHi_dy_a, d2Hi_dy2_b )

    ! call map_a_a_2D( mesh, ice%Hs_a, Hs_a       )
    call ddx_a_a_2D( mesh, ice%Hs  , dHs_dx_a   )
    call ddy_a_a_2D( mesh, ice%Hs  , dHs_dy_a   )
    call map_a_b_2D( mesh, ice%Hs  , Hs_b       )
    call ddx_a_b_2D( mesh, ice%Hs  , dHs_dx_b   )
    call ddy_a_b_2D( mesh, ice%Hs  , dHs_dy_b   )
    call ddx_b_a_2D( mesh, dHs_dx_b, d2Hs_dx2_a )
    call ddy_b_a_2D( mesh, dHs_dx_b, d2Hs_dxdy_a)
    call ddy_b_a_2D( mesh, dHs_dy_b, d2Hs_dy2_a )
    call ddx_a_b_2D( mesh, dHs_dx_a, d2Hs_dx2_b )
    call ddy_a_b_2D( mesh, dHs_dx_a, d2Hs_dxdy_b)
    call ddy_a_b_2D( mesh, dHs_dy_a, d2Hs_dy2_b )

    ! Calculate zeta gradients on all grids

    ! ak
    do vi = mesh%vi1, mesh%vi2

      Hi = max( 10._dp, ice%Hi( vi))

      do k = 1, mesh%nz

        zeta = mesh%zeta( k)

        ice%dzeta_dt_ak(    vi,k) = ( 1._dp / Hi) * (ice%dHs_dt( vi) - zeta * ice%dHi_dt( vi))

        ice%dzeta_dx_ak(    vi,k) = ( 1._dp / Hi) * (dHs_dx_a( vi) - zeta * dHi_dx_a( vi))
        ice%dzeta_dy_ak(    vi,k) = ( 1._dp / Hi) * (dHs_dy_a( vi) - zeta * dHi_dy_a( vi))
        ice%dzeta_dz_ak(    vi,k) = (-1._dp / Hi)

        ice%d2zeta_dx2_ak(  vi,k) = (dHi_dx_a( vi) * -1._dp / Hi) * ice%dzeta_dx_ak( vi,k) + (1._dp / Hi) * (d2Hs_dx2_a(  vi) - zeta * d2Hi_dx2_a(  vi))
        ice%d2zeta_dxdy_ak( vi,k) = (dHi_dy_a( vi) * -1._dp / Hi) * ice%dzeta_dx_ak( vi,k) + (1._dp / Hi) * (d2Hs_dxdy_a( vi) - zeta * d2Hi_dxdy_a( vi))
        ice%d2zeta_dy2_ak(  vi,k) = (dHi_dy_a( vi) * -1._dp / Hi) * ice%dzeta_dy_ak( vi,k) + (1._dp / Hi) * (d2Hs_dy2_a(  vi) - zeta * d2Hi_dy2_a(  vi))

      end do
    end do

    ! bk
    do ti = mesh%ti1, mesh%ti2

      Hi = max( 10._dp, Hi_b( ti))

      do k = 1, mesh%nz

        zeta = mesh%zeta( k)

        ice%dzeta_dx_bk(    ti,k) = ( 1._dp / Hi) * (dHs_dx_b( ti) - zeta * dHi_dx_b( ti))
        ice%dzeta_dy_bk(    ti,k) = ( 1._dp / Hi) * (dHs_dy_b( ti) - zeta * dHi_dy_b( ti))
        ice%dzeta_dz_bk(    ti,k) = (-1._dp / Hi)

        ice%d2zeta_dx2_bk(  ti,k) = (dHi_dx_b( ti) * -1._dp / Hi) * ice%dzeta_dx_bk( ti,k) + (1._dp / Hi) * (d2Hs_dx2_b(  ti) - zeta * d2Hi_dx2_b(  ti))
        ice%d2zeta_dxdy_bk( ti,k) = (dHi_dy_b( ti) * -1._dp / Hi) * ice%dzeta_dx_bk( ti,k) + (1._dp / Hi) * (d2Hs_dxdy_b( ti) - zeta * d2Hi_dxdy_b( ti))
        ice%d2zeta_dy2_bk(  ti,k) = (dHi_dy_b( ti) * -1._dp / Hi) * ice%dzeta_dy_bk( ti,k) + (1._dp / Hi) * (d2Hs_dy2_b(  ti) - zeta * d2Hi_dy2_b(  ti))

      end do
    end do

    ! bks
    do ti = mesh%ti1, mesh%ti2

      Hi = max( 10._dp, Hi_b( ti))

      do ks = 1, mesh%nz-1

        zeta = mesh%zeta_stag( ks)

        ice%dzeta_dx_bks(    ti,ks) = ( 1._dp / Hi) * (dHs_dx_b( ti) - zeta * dHi_dx_b( ti))
        ice%dzeta_dy_bks(    ti,ks) = ( 1._dp / Hi) * (dHs_dy_b( ti) - zeta * dHi_dy_b( ti))
        ice%dzeta_dz_bks(    ti,ks) = (-1._dp / Hi)

        ice%d2zeta_dx2_bks(  ti,ks) = (dHi_dx_b( ti) * -1._dp / Hi) * ice%dzeta_dx_bks( ti,ks) + (1._dp / Hi) * (d2Hs_dx2_b(  ti) - zeta * d2Hi_dx2_b(  ti))
        ice%d2zeta_dxdy_bks( ti,ks) = (dHi_dy_b( ti) * -1._dp / Hi) * ice%dzeta_dx_bks( ti,ks) + (1._dp / Hi) * (d2Hs_dxdy_b( ti) - zeta * d2Hi_dxdy_b( ti))
        ice%d2zeta_dy2_bks(  ti,ks) = (dHi_dy_b( ti) * -1._dp / Hi) * ice%dzeta_dy_bks( ti,ks) + (1._dp / Hi) * (d2Hs_dy2_b(  ti) - zeta * d2Hi_dy2_b(  ti))

      end do
    end do

    ! Clean after yourself
    ! deallocate( Hi_a        )
    deallocate( dHi_dx_a    )
    deallocate( dHi_dy_a    )
    deallocate( d2Hi_dx2_a  )
    deallocate( d2Hi_dxdy_a )
    deallocate( d2Hi_dy2_a  )

    deallocate( Hi_b        )
    deallocate( dHi_dx_b    )
    deallocate( dHi_dy_b    )
    deallocate( d2Hi_dx2_b  )
    deallocate( d2Hi_dxdy_b )
    deallocate( d2Hi_dy2_b  )

    ! deallocate( Hs_a        )
    deallocate( dHs_dx_a    )
    deallocate( dHs_dy_a    )
    deallocate( d2Hs_dx2_a  )
    deallocate( d2Hs_dxdy_a )
    deallocate( d2Hs_dy2_a  )

    deallocate( Hs_b        )
    deallocate( dHs_dx_b    )
    deallocate( dHs_dy_b    )
    deallocate( d2Hs_dx2_b  )
    deallocate( d2Hs_dxdy_b )
    deallocate( d2Hs_dy2_b  )

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine calc_zeta_gradients

! == No-ice mask
! ==============

  subroutine calc_mask_noice( mesh, ice)
    !< Calculate the no-ice mask

    ! In/output variables:
    type(type_mesh),      intent(in   ) :: mesh
    type(type_ice_model), intent(inout) :: ice

    ! Local variables:
    character(len=1024), parameter :: routine_name = 'calc_mask_noice'
    integer                        :: vi

    ! Add routine to path
    call init_routine( routine_name)

    ! Initialise
    ! ==========

    ice%mask_noice = .false.

    ! domain-specific cases (mutually exclusive)
    ! ==========================================

    select case (C%choice_mask_noice)
    case default
      call crash('unknown choice_mask_noice "' // trim( C%choice_mask_noice) // '"')
      case ('none')
        ! Ice is (in principle) allowed everywhere

        ice%mask_noice = .false.

      case ('MISMIP_mod')
        ! Kill all ice when r > 900 km

        do vi = mesh%vi1, mesh%vi2
          if (NORM2( mesh%V( vi,:)) > 900E3_dp) then
            ice%mask_noice( vi) = .true.
          else
            ice%mask_noice( vi) = .false.
          end if
        end do

      case ('MISMIP+')
        ! Kill all ice when x > 640 km

        do vi = mesh%vi1, mesh%vi2
          if (mesh%V( vi,1) > 640E3_dp) then
            ice%mask_noice( vi) = .true.
          else
            ice%mask_noice( vi) = .false.
          end if
        end do

      case ('remove_Ellesmere')
        ! Prevent ice growth in the Ellesmere Island part of the Greenland domain

        call calc_mask_noice_remove_Ellesmere( mesh, ice%mask_noice)

    end select

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine calc_mask_noice

  subroutine calc_mask_noice_remove_Ellesmere( mesh, mask_noice)
    !< Prevent ice growth in the Ellesmere Island part of the Greenland domain

    ! In- and output variables
    type(type_mesh),                        intent(in   ) :: mesh
    logical,  dimension(mesh%vi1:mesh%vi2), intent(inout) :: mask_noice

    ! Local variables:
    character(len=1024), parameter :: routine_name = 'calc_mask_noice_remove_Ellesmere'
    integer                        :: vi
    real(dp), dimension(2)         :: pa_latlon, pb_latlon, pa, pb
    real(dp)                       :: xa, ya, xb, yb, yl_ab

    ! Add routine to path
    call init_routine( routine_name)

    ! The two endpoints in lat,lon
    pa_latlon = [76.74_dp, -74.79_dp]
    pb_latlon = [82.19_dp, -60.00_dp]

    ! The two endpoints in x,y
    call oblique_sg_projection( pa_latlon(2), pa_latlon(1), mesh%lambda_M, mesh%phi_M, mesh%beta_stereo, xa, ya)
    call oblique_sg_projection( pb_latlon(2), pb_latlon(1), mesh%lambda_M, mesh%phi_M, mesh%beta_stereo, xb, yb)

    pa = [xa,ya]
    pb = [xb,yb]

    do vi = mesh%vi1, mesh%vi2
      yl_ab = pa(2) + (mesh%V( vi,1) - pa(1)) * (pb(2)-pa(2)) / (pb(1)-pa(1))
      if (mesh%V( vi,2) > pa(2) .and. mesh%V( vi,2) > yl_ab .and. mesh%V( vi,1) < pb(1)) then
        mask_noice( vi) = .true.
      else
        mask_noice( vi) = .false.
      end if
    end do

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine calc_mask_noice_remove_Ellesmere

! == Ice thickness modification
! =============================

  subroutine alter_ice_thickness( mesh, ice, Hi_old, Hi_new, refgeo, time)
    !< Modify the predicted ice thickness in some sneaky way

    ! In- and output variables:
    type(type_mesh),                        intent(in   ) :: mesh
    type(type_ice_model),                   intent(in   ) :: ice
    real(dp), dimension(mesh%vi1:mesh%vi2), intent(in   ) :: Hi_old
    real(dp), dimension(mesh%vi1:mesh%vi2), intent(inout) :: Hi_new
    type(type_reference_geometry),          intent(in   ) :: refgeo
    real(dp),                               intent(in   ) :: time

    ! Local variables:
    character(len=1024), parameter         :: routine_name = 'alter_ice_thickness'
    integer                                :: vi
    real(dp)                               :: decay_start, decay_end
    real(dp)                               :: fixiness, limitness, fix_H_applied, limit_H_applied
    real(dp), dimension(mesh%vi1:mesh%vi2) :: modiness_up, modiness_down
    real(dp), dimension(mesh%vi1:mesh%vi2) :: Hi_save, Hi_eff_new, fraction_margin_new
    real(dp)                               :: floating_area, calving_area, mass_lost

    ! Add routine to path
    call init_routine( routine_name)

    ! Save predicted ice thickness for future reference
    Hi_save = Hi_new

    ! Calculate would-be effective thickness
    call calc_effective_thickness( mesh, ice, Hi_new, Hi_eff_new, fraction_margin_new)

    ! == Mask conservation
    ! ====================

    ! if desired, don't let grounded ice cross the floatation threshold
    if (C%do_protect_grounded_mask .and. time <= C%protect_grounded_mask_t_end) then
      do vi = mesh%vi1, mesh%vi2
        if (ice%mask_grounded_ice( vi)) then
          Hi_new( vi) = max( Hi_new( vi), (ice%SL( vi) - ice%Hb( vi)) * seawater_density/ice_density + .1_dp)
        end if
      end do
    end if

    ! General cases
    ! =============

    ! if so specified, remove very thin ice
    do vi = mesh%vi1, mesh%vi2
      if (Hi_eff_new( vi) < C%Hi_min) then
        Hi_new( vi) = 0._dp
      end if
    end do

    ! if so specified, remove thin floating ice
    if (C%choice_calving_law == 'threshold_thickness') then
      do vi = mesh%vi1, mesh%vi2
        if (is_floating( Hi_eff_new( vi), ice%Hb( vi), ice%SL( vi)) .and. Hi_eff_new( vi) < C%calving_threshold_thickness_shelf) then
          Hi_new( vi) = 0._dp
        end if
      end do
    end if

    ! DENK DROM
    if (C%remove_ice_absent_at_PD) then
      do vi = mesh%vi1, mesh%vi2
        if (refgeo%Hi( vi) == 0._dp) then
          Hi_new( vi) = 0._dp
        end if
      end do
    end if

    ! if so specified, remove all floating ice
    if (C%do_remove_shelves) then
      do vi = mesh%vi1, mesh%vi2
        if (is_floating( Hi_eff_new( vi), ice%Hb( vi), ice%SL( vi))) then
          Hi_new( vi) = 0._dp
        end if
      end do
    end if

    ! if so specified, remove all floating ice beyond the present-day calving front
    if (C%remove_shelves_larger_than_PD) then
      do vi = mesh%vi1, mesh%vi2
        if (refgeo%Hi( vi) == 0._dp .and. refgeo%Hb( vi) < 0._dp) then
          Hi_new( vi) = 0._dp
        end if
      end do
    end if

    ! if so specified, remove all floating ice crossing the continental shelf edge
    if (C%continental_shelf_calving) then
      do vi = mesh%vi1, mesh%vi2
        if (refgeo%Hi( vi) == 0._dp .and. refgeo%Hb( vi) < C%continental_shelf_min_height) then
          Hi_new( vi) = 0._dp
        end if
      end do
    end if

    ! === Fixiness ===
    ! ================

    ! Intial value
    fixiness = 1._dp

    ! Make sure that the start and end times make sense
    decay_start = C%fixiness_t_start
    decay_end   = C%fixiness_t_end

    ! Compute decaying fixiness
    if (decay_start >= decay_end) then
      ! Fix interval makes no sense: ignore fixiness
      fixiness = 0._dp
    elseif (time <= decay_start) then
      ! Before fix interval: check chosen option
      if (C%do_fixiness_before_start) then
        fixiness = 1._dp
      else
        fixiness = 0._dp
      end if
    elseif (time >= decay_end) then
      ! After fix interval: remove any fix/delay
      fixiness = 0._dp
    else
      ! We're within the fix interval: fixiness decreases with time
      fixiness = 1._dp - (time - decay_start) / (decay_end - decay_start)
    end if

    ! Just in case
    fixiness = min( 1._dp, max( 0._dp, fixiness))

    ! === Limitness ===
    ! =================

    ! Intial value
    limitness = 1._dp

    ! Make sure that the start and end times make sense
    decay_start = C%limitness_t_start
    decay_end   = C%limitness_t_end

    ! Compute decaying limitness
    if (decay_start >= decay_end) then
      ! Limit interval makes no sense: ignore limitness
      limitness = 0._dp
    elseif (time <= decay_start) then
      ! Before limit interval: check chosen option
      if (C%do_limitness_before_start) then
        limitness = 1._dp
      else
        limitness = 0._dp
      end if
    elseif (time >= decay_end) then
      ! After limit interval: remove any limits
      limitness = 0._dp
    else
      ! Limitness decreases with time
      limitness = 1._dp - (time - decay_start) / (decay_end - decay_start)
    end if

    ! Just in case
    limitness = min( 1._dp, max( 0._dp, limitness))

    ! === Modifier ===
    ! ================

    ! Intial value
    modiness_up   = 0._dp
    modiness_down = 0._dp

    select case (C%modiness_H_style)
    case default
      call crash('unknown modiness_H_choice "' // trim( C%modiness_H_style) // '"')
    case ('none')
      modiness_up   = 0._dp
      modiness_down = 0._dp

    case ('Ti_hom')
      modiness_up   = 1._dp - exp(ice%Ti_hom/C%modiness_T_hom_ref)
      modiness_down = 1._dp - exp(ice%Ti_hom/C%modiness_T_hom_ref)

    case ('Ti_hom_up')
      modiness_up   = 1._dp - exp(ice%Ti_hom/C%modiness_T_hom_ref)
      modiness_down = 0._dp

    case ('Ti_hom_down')
      modiness_up   = 0._dp
      modiness_down = 1._dp - exp(ice%Ti_hom/C%modiness_T_hom_ref)

    case ('no_thick_inland')
      do vi = mesh%vi1, mesh%vi2
        if (ice%mask_grounded_ice( vi) .and. .not. ice%mask_gl_gr( vi)) then
          modiness_up( vi) = 1._dp
        end if
      end do
      modiness_down = 0._dp

    case ('no_thin_inland')
      do vi = mesh%vi1, mesh%vi2
        if (ice%mask_grounded_ice( vi) .and. .not. ice%mask_gl_gr( vi)) then
          modiness_down( vi) = 1._dp
        end if
      end do
      modiness_up = 0._dp

    end select

    ! Just in case
    modiness_up   = min( 1._dp, max( 0._dp, modiness_up  ))
    modiness_down = min( 1._dp, max( 0._dp, modiness_down))

    ! === Fix, delay, limit ===
    ! =========================

    do vi = mesh%vi1, mesh%vi2

      ! Initialise
      fix_H_applied   = 0._dp
      limit_H_applied = 0._dp

      if (    ice%mask_gl_gr( vi)) then
        fix_H_applied   = C%fixiness_H_gl_gr * fixiness
        limit_H_applied = C%limitness_H_gl_gr * limitness

      elseif (ice%mask_gl_fl( vi)) then
        fix_H_applied   = C%fixiness_H_gl_fl * fixiness
        limit_H_applied = C%limitness_H_gl_fl * limitness

      elseif (ice%mask_grounded_ice( vi)) then
        fix_H_applied   = C%fixiness_H_grounded * fixiness
        limit_H_applied = C%limitness_H_grounded * limitness

      elseif (ice%mask_floating_ice( vi)) then
        fix_H_applied   = C%fixiness_H_floating * fixiness
        limit_H_applied = C%limitness_H_floating * limitness

      elseif (ice%mask_icefree_land( vi)) then
        if (C%fixiness_H_freeland .and. fixiness > 0._dp) fix_H_applied = 1._dp
        limit_H_applied = C%limitness_H_grounded * limitness

      elseif (ice%mask_icefree_ocean( vi)) then
        if (C%fixiness_H_freeocean .and. fixiness > 0._dp) fix_H_applied = 1._dp
        limit_H_applied = C%limitness_H_floating * limitness
      else
        ! if we reached this point, vertex is neither grounded, floating, nor ice free. That's a problem
        call crash('vertex neither grounded, floating, nor ice-free?')
      end if

      ! Apply fixiness
      Hi_new( vi) = Hi_old( vi) * fix_H_applied + Hi_new( vi) * (1._dp - fix_H_applied)

      ! Apply limitness
      Hi_new( vi) = min( Hi_new( vi), refgeo%Hi( vi) + (1._dp - modiness_up(   vi)) * limit_H_applied &
                                                     + (1._dp - limitness         ) * (Hi_new( vi) - refgeo%Hi( vi)) )

      Hi_new( vi) = max( Hi_new( vi), refgeo%Hi( vi) - (1._dp - modiness_down( vi)) * limit_H_applied &
                                                     - (1._dp - limitness         ) * (refgeo%Hi( vi) - Hi_new( vi)) )

    end do

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine alter_ice_thickness

  subroutine MB_inversion( mesh, ice, refgeo, SMB, BMB, LMB, AMB, dHi_dt_predicted, Hi_predicted, dt, time, region_name)
    !< Calculate the basal mass balance
    !< Use an inversion based on the computed dHi_dt

    ! In/output variables:
    type(type_mesh),                        intent(in   ) :: mesh
    type(type_ice_model),                   intent(in   ) :: ice
    type(type_reference_geometry),          intent(in   ) :: refgeo
    type(type_SMB_model),                   intent(inout) :: SMB
    type(type_BMB_model),                   intent(inout) :: BMB
    type(type_LMB_model),                   intent(inout) :: LMB
    type(type_AMB_model),                   intent(inout) :: AMB
    real(dp), dimension(mesh%vi1:mesh%vi2), intent(inout) :: dHi_dt_predicted
    real(dp), dimension(mesh%vi1:mesh%vi2), intent(inout) :: Hi_predicted
    real(dp),                               intent(in   ) :: dt
    real(dp),                               intent(in   ) :: time
    character(len=3)                                      :: region_name

    ! Local variables:
    character(len=1024), parameter         :: routine_name = 'MB_inversion'
    integer                                :: vi
    logical                                :: do_BMB_inversion, do_LMB_inversion, do_SMB_inversion, do_SMB_absorb
    integer,  dimension(mesh%vi1:mesh%vi2) :: mask
    real(dp), dimension(mesh%vi1:mesh%vi2) :: previous_field
    real(dp)                               :: value_change
    real(dp), dimension(:,:), allocatable  :: poly_ROI
    real(dp), dimension(2)                 :: p

    ! == Initialisation
    ! =================

    if (C%choice_regions_of_interest == 'Patagonia') then
      ! Compute polygon for reconstruction
      call calc_polygon_Patagonia( poly_ROI)
    else
      ! Create a dummy polygon
      allocate( poly_ROI(1,2))
      poly_ROI(1,1) = 0._dp
      poly_ROI(1,2) = 0._dp
    end if

    ! Add routine to path
    call init_routine( routine_name)

    do_BMB_inversion = .false.
    do_LMB_inversion = .false.
    do_SMB_inversion = .false.
    do_SMB_absorb    = .false.

    ! Check whether we want a BMB inversion
    if (C%do_BMB_inversion .and. &
        time >= C%BMB_inversion_t_start .and. &
        time <= C%BMB_inversion_t_end) then
      do_BMB_inversion = .true.
    end if

    ! Check whether we want a LMB inversion
    if (C%do_LMB_inversion .and. &
        time >= C%LMB_inversion_t_start .and. &
        time <= C%LMB_inversion_t_end) then
      do_LMB_inversion = .true.
    end if

    if (C%do_SMB_removal_icefree_land) then
      do_SMB_inversion = .true.
    end if

    ! Check whether we want a SMB adjustment
    if (C%do_SMB_residual_absorb .and. &
        time >= C%SMB_residual_absorb_t_start .and. &
        time <= C%SMB_residual_absorb_t_end) then
      do_SMB_absorb = .true.
    end if

    ! == BMB: first pass
    ! ==================

    ! Store previous values
    previous_field = BMB%BMB_inv

    ! Initialise extrapolation mask
    mask = 0

    do vi = mesh%vi1, mesh%vi2

      ! Get x and y coordinates of this vertex
      p = mesh%V( vi,:)

      ! Skip vertices within reconstruction polygon
      if (is_in_polygon(poly_ROI, p)) cycle

      ! Skip if not desired
      if (.not. do_BMB_inversion) cycle

      ! Floating calving fronts
      if (ice%mask_cf_fl( vi)) then

        ! Just mark for eventual extrapolation
        mask( vi) = 1

      elseif (ice%mask_floating_ice( vi)) then

        ! Basal melt will account for all change here
        BMB%BMB_inv( vi) = BMB%BMB( vi) - dHi_dt_predicted( vi)

        ! Adjust rate of ice thickness change dHi/dt to compensate the change
        dHi_dt_predicted( vi) = 0._dp

        ! Adjust corrected ice thickness to compensate the change
        Hi_predicted( vi) = ice%Hi_prev( vi)

        ! Use this vertex as seed during extrapolation
        mask( vi) = 2

      elseif (ice%mask_gl_gr( vi)) then

        ! For grounding lines, BMB accounts for melt
        if (dHi_dt_predicted( vi) >= 0._dp) then

          ! Basal melt will account for all change here
          BMB%BMB_inv( vi) = BMB%BMB( vi) - dHi_dt_predicted( vi)
          ! Adjust rate of ice thickness change dHi/dt to compensate the change
          dHi_dt_predicted( vi) = 0._dp
          ! Adjust corrected ice thickness to compensate the change
          Hi_predicted( vi) = ice%Hi_prev( vi)

        else
          ! Remove basal melt, but do not add refreezing
          BMB%BMB_inv( vi) = 0._dp
        end if

        ! Ignore this vertex during extrapolation
        mask( vi) = 0

      else
        ! Not a place where basal melt operates
        BMB%BMB_inv( vi) = 0._dp
        ! Ignore this vertex during extrapolation
        mask( vi) = 0
      end if

    end do

    ! == Extrapolate into calving fronts
    ! ==================================

    ! Perform the extrapolation - mask: 2 -> use as seed; 1 -> extrapolate; 0 -> ignore
    call extrapolate_Gaussian( mesh, mask, BMB%BMB_inv, 16000._dp)

    do vi = mesh%vi1, mesh%vi2
      if (ice%mask_cf_fl( vi)) then

        ! Get x and y coordinates of this vertex
        p = mesh%V( vi,:)

        ! Skip vertices within reconstruction polygon
        if (is_in_polygon(poly_ROI, p)) cycle

        ! Skip if not desired
        if (.not. do_BMB_inversion) cycle

        ! Compute change after extrapolation
        value_change = BMB%BMB_inv( vi) - previous_field( vi)

        ! Adjust rate of ice thickness change dHi/dt to compensate the change
        dHi_dt_predicted( vi) = dHi_dt_predicted( vi) + value_change

        ! Adjust new ice thickness to compensate the change
        Hi_predicted( vi) = ice%Hi_prev( vi) + dHi_dt_predicted( vi) * dt

      end if
    end do

    ! == LMB: remaining positive dHi_dt
    ! =================================

    do vi = mesh%vi1, mesh%vi2

      ! Get x and y coordinates of this vertex
      p = mesh%V( vi,:)

      ! Skip vertices within reconstruction polygon
      if (is_in_polygon(poly_ROI, p)) cycle

      ! Skip if not desired
      if (.not. do_LMB_inversion) cycle

      ! For these areas, use dHi_dt to get an "inversion" of equilibrium LMB.
      if (ice%mask_cf_fl( vi) .OR. ice%mask_cf_gr( vi) .OR. ice%mask_icefree_ocean( vi)) then

        if (dHi_dt_predicted( vi) >= 0._dp .and. ice%fraction_margin( vi) < 1._dp) then

          ! Assume that calving accounts for all remaining mass loss here (after first BMB pass)
          LMB%LMB_inv( vi) = LMB%LMB( vi) - dHi_dt_predicted( vi)
          ! Adjust rate of ice thickness change dHi/dt to compensate the change
          dHi_dt_predicted( vi) = 0._dp
          ! Adjust corrected ice thickness to compensate the change
          Hi_predicted( vi) = ice%Hi_prev( vi)

        else
          ! Remove lateral melt, but do not add mass
          LMB%LMB_inv( vi) = 0._dp
        end if

      else
        ! Not a place where lateral melt operates
        LMB%LMB_inv( vi) = 0._dp
      end if

    end do ! vi = mesh%vi1, mesh%vi2

    ! ! == BMB: final pass
    ! ! ==================

    ! do vi = mesh%vi1, mesh%vi2
    !   if (ice%mask_cf_fl( vi)) then

    !     ! Get x and y coordinates of this vertex
    !     p = mesh%V( vi,:)

    !     ! Skip vertices within reconstruction polygon
    !     if (is_in_polygon(poly_ROI, p)) cycle

    !     ! Skip if not desired
    !     if (.not. do_BMB_inversion) cycle

    !     ! BMB will absorb all remaining change after calving did its thing
    !     BMB%BMB( vi) = BMB%BMB( vi) - dHi_dt_predicted( vi)

    !     ! Adjust rate of ice thickness change dHi/dt to compensate the change
    !     dHi_dt_predicted( vi) = 0._dp

    !     ! Adjust corrected ice thickness to compensate the change
    !     Hi_predicted( vi) = ice%Hi_prev( vi)

    !   end if
    ! end do

    ! == SMB: reference ice-free land areas
    ! =====================================

    ! Store pre-adjustment values
    previous_field = SMB%SMB

    do vi = mesh%vi1, mesh%vi2

      ! Get x and y coordinates of this vertex
      p = mesh%V( vi,:)

      ! Skip vertices within reconstruction polygon
      if (is_in_polygon(poly_ROI, p)) cycle

      ! Skip if not desired
      if (.not. do_SMB_inversion) cycle

      ! Skip other areas
      if (refgeo%Hb( vi) < refgeo%SL( vi) .OR. refgeo%Hi( vi) > 0._dp) cycle

      ! For reference ice-free land, use dHi_dt to get an "inversion" of equilibrium SMB.
      SMB%SMB( vi) = min( 0._dp, ice%divQ( vi))

      ! Adjust rate of ice thickness change dHi/dt to compensate the change
      dHi_dt_predicted( vi) = 0._dp

      ! Adjust corrected ice thickness to compensate the change
      Hi_predicted( vi) = ice%Hi_prev( vi)

    end do

    ! == SMB: absorb remaining change
    ! ===============================

    ! Store pre-adjustment values
    previous_field = SMB%SMB

    do vi = mesh%vi1, mesh%vi2

      ! Get x and y coordinates of this vertex
      p = mesh%V( vi,:)

      ! Skip vertices within reconstruction polygon
      if (is_in_polygon(poly_ROI, p)) cycle

      if (.not. do_SMB_absorb) cycle

      ! For grounded ice, use dHi_dt to get an "inversion" of equilibrium SMB.
      SMB%SMB( vi) = SMB%SMB( vi) - dHi_dt_predicted( vi)

      ! Adjust rate of ice thickness change dHi/dt to compensate the change
      dHi_dt_predicted( vi) = 0._dp

      ! Adjust corrected ice thickness to compensate the change
      Hi_predicted( vi) = ice%Hi_prev( vi)

    end do

    ! == Assign main fields
    ! =====================

    ! Clean up after yourself
    deallocate( poly_ROI)

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine MB_inversion

! == Trivia
! =========

  subroutine MISMIPplus_adapt_flow_factor( mesh, ice)
    !< Automatically adapt the uniform flow factor A to achieve
    ! a steady-state mid-stream grounding-line position at x = 450 km in the MISMIP+ experiment

    ! In- and output variables
    type(type_mesh),      intent(in   ) :: mesh
    type(type_ice_model), intent(in   ) :: ice

    ! Local variables:
    character(len=1024), parameter :: routine_name = 'MISMIPplus_adapt_flow_factor'
    real(dp), dimension(2)         :: pp,qq
    integer                        :: ti_in
    real(dp)                       :: TAFp,TAFq,lambda_GL, x_GL
    real(dp)                       :: A_flow_old, f, A_flow_new

    ! Add routine to path
    call init_routine( routine_name)

    ! Safety
    if (C%choice_ice_rheology_Glen /= 'uniform') then
      call crash('only works in MISMIP+ geometry with a uniform flow factor!')
    end if

    ! Determine mid-channel grounding-line position
    pp = [mesh%xmin, 0._dp]
    qq = pp
    TAFp = 1._dp
    TAFq = 1._dp
    ti_in = 1
    do while (TAFp * TAFq > 0._dp)
      pp   = qq
      TAFp = TAFq
      qq = pp + [C%maximum_resolution_grounding_line, 0._dp]
      call interpolate_to_point_dp_2D( mesh, ice%TAF, qq, ti_in, TAFq)
    end do

    lambda_GL = TAFp / (TAFp - TAFq)
    x_GL = lambda_GL * qq( 1) + (1._dp - lambda_GL) * pp( 1)

    ! Adjust the flow factor
    f = 2._dp ** ((x_GL - 450E3_dp) / 80000._dp)
    C%uniform_Glens_flow_factor = C%uniform_Glens_flow_factor * f

    if (par%master) write(0,*) '    MISMIPplus_adapt_flow_factor: x_GL = ', x_GL/1E3, ' km; changed flow factor to ', C%uniform_Glens_flow_factor

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine MISMIPplus_adapt_flow_factor

! == Target dHi_dt initialisation
! ===============================

  subroutine initialise_dHi_dt_target( mesh, ice, region_name)
    !< Prescribe a target dHi_dt from a file without a time dimension

    ! In- and output variables
    type(type_mesh),      intent(in   ) :: mesh
    type(type_ice_model), intent(inout) :: ice
    character(len=3),     intent(in   ) :: region_name

    ! Local variables:
    character(len=1024), parameter :: routine_name = 'initialise_dHi_dt_target'
    character(len=256)             :: filename_dHi_dt_target
    real(dp)                       :: timeframe_dHi_dt_target

    ! Add routine to path
    call init_routine( routine_name)

    ! Determine filename for this model region
    select case (region_name)
    case default
      call crash('unknown region_name "' // trim( region_name) // '"!')
    case ('NAM')
      filename_dHi_dt_target  = C%filename_dHi_dt_target_NAM
      timeframe_dHi_dt_target = C%timeframe_dHi_dt_target_NAM
    case ('EAS')
      filename_dHi_dt_target  = C%filename_dHi_dt_target_EAS
      timeframe_dHi_dt_target = C%timeframe_dHi_dt_target_EAS
    case ('GRL')
      filename_dHi_dt_target  = C%filename_dHi_dt_target_GRL
      timeframe_dHi_dt_target = C%timeframe_dHi_dt_target_GRL
    case ('ANT')
      filename_dHi_dt_target  = C%filename_dHi_dt_target_ANT
      timeframe_dHi_dt_target = C%timeframe_dHi_dt_target_ANT
    end select

    ! Print to terminal
    if (par%master)  write(*,"(A)") '     Initialising target ice rates of change from file "' // colour_string( trim( filename_dHi_dt_target),'light blue') // '"...'

    ! Read dHi_dt from file
    if (timeframe_dHi_dt_target == 1E9_dp) then
      ! Assume the file has no time dimension
      call read_field_from_file_2D( filename_dHi_dt_target, 'dHdt||dHi_dt', mesh, ice%dHi_dt_target)
    else
      ! Assume the file has a time dimension, and read the specified timeframe
      call read_field_from_file_2D( filename_dHi_dt_target, 'dHdt||dHi_dt', mesh, ice%dHi_dt_target, time_to_read = timeframe_dHi_dt_target)
    end if

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine initialise_dHi_dt_target

! == Target uabs_surf initialisation
! ==================================

  subroutine initialise_uabs_surf_target( mesh, ice, region_name)
    !< Initialise surface ice velocity data from an external NetCDF file

    ! Input variables:
    type(type_mesh),      intent(in   ) :: mesh
    type(type_ice_model), intent(inout) :: ice
    character(len=3),     intent(in   ) :: region_name

    ! Local variables:
    character(len=1024), parameter :: routine_name = 'initialise_uabs_surf_target'
    character(len=256)             :: filename_uabs_surf_target
    real(dp)                       :: timeframe_uabs_surf_target

    ! Add routine to path
    call init_routine( routine_name)

    ! Determine filename and timeframe for this model region
    select case (region_name)
    case default
      call crash('unknown region_name "' // trim( region_name) // '"!')
    case('NAM')
      filename_uabs_surf_target  = C%filename_uabs_surf_target_NAM
      timeframe_uabs_surf_target = C%timeframe_uabs_surf_target_NAM
    case ('EAS')
      filename_uabs_surf_target  = C%filename_uabs_surf_target_EAS
      timeframe_uabs_surf_target = C%timeframe_uabs_surf_target_EAS
    case ('GRL')
      filename_uabs_surf_target  = C%filename_uabs_surf_target_GRL
      timeframe_uabs_surf_target = C%timeframe_uabs_surf_target_GRL
    case ('ANT')
      filename_uabs_surf_target  = C%filename_uabs_surf_target_ANT
      timeframe_uabs_surf_target = C%timeframe_uabs_surf_target_ANT
    end select

    ! Print to terminal
    if (par%master)  write(*,"(A)") '     Initialising target surface ice speed from file "' // colour_string( trim( filename_uabs_surf_target),'light blue') // '"...'

    if (timeframe_uabs_surf_target == 1E9_dp) then
      call read_field_from_file_2D( filename_uabs_surf_target, 'uabs_surf', mesh, ice%uabs_surf_target)
    else
      call read_field_from_file_2D( filename_uabs_surf_target, 'uabs_surf', mesh, ice%uabs_surf_target, timeframe_uabs_surf_target)
    end if

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine initialise_uabs_surf_target

end module ice_model_utilities
