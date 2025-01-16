module ice_model_main

  !< The main ice-dynamical model

  use mpi
  use mpi_basic, only: par
  use precisions, only: dp
  use control_resources_and_error_messaging, only: init_routine, finalise_routine, crash, warning, colour_string
  use model_configuration, only: C
  use parameters, only: grav, ice_density, seawater_density
  use region_types, only: type_model_region
  use mesh_types, only: type_mesh
  use ice_model_types, only: type_ice_model, type_ice_pc
  use reference_geometry_types, only: type_reference_geometry
  use GIA_model_types, only: type_GIA_model
  use SMB_model_types, only: type_SMB_model
  use BMB_model_types, only: type_BMB_model
  use LMB_model_types, only: type_LMB_model
  use AMB_model_types, only: type_AMB_model
  use CSR_sparse_matrix_type, only: type_sparse_matrix_CSR_dp
  use remapping_main, only: Atlas
  use conservation_of_mass_main, only: calc_dHi_dt, apply_ice_thickness_BC_explicit, &
    apply_mask_noice_direct
  use mpi_distributed_memory, only: gather_to_all, distribute_from_master
  use conservation_of_momentum_main, only: solve_stress_balance, map_velocities_from_b_to_c_2D, &
    remap_velocity_solver, create_restart_file_ice_velocity, write_to_restart_file_ice_velocity, &
    initialise_velocity_solver
  use mesh_utilities, only: extrapolate_Gaussian
  use ice_model_utilities, only: MB_inversion, alter_ice_thickness, determine_masks, &
    calc_grounded_fractions, calc_effective_thickness, calc_zeta_gradients, calc_mask_noice, &
    initialise_dHi_dt_target, initialise_uabs_surf_target, initialise_bedrock_CDFs, calc_bedrock_CDFs
  use reallocate_mod, only: reallocate_bounds
  use netcdf_io_main
  use ice_geometry_basics, only: ice_surface_elevation, thickness_above_floatation, Hi_from_Hb_Hs_and_SL
  use mesh_disc_apply_operators, only: ddx_a_a_2D, ddy_a_a_2D, ddx_a_b_2D, ddy_a_b_2D
  use geothermal_heat_flux, only: initialise_geothermal_heat_flux
  use basal_hydrology, only: initialise_basal_hydrology_model
  use bed_roughness, only: initialise_bed_roughness
  use ice_model_memory, only: allocate_ice_model

  implicit none

contains

! ===== Main routines =====
! =========================

  subroutine run_ice_dynamics_model( region)
    !< Calculate ice geometry at the desired time, and update
    !< velocities, thinning rates, and predicted geometry if necessary

    ! In/output variables:
    type(type_model_region), intent(inout) :: region

    ! Local variables:
    character(len=1024), parameter                       :: routine_name = 'run_ice_dynamics_model'
    real(dp)                                             :: wt_prev, wt_next
    integer                                              :: vi
    real(dp)                                             :: dt_max
    real(dp), dimension(region%mesh%vi1:region%mesh%vi2) :: dHs_dx, dHs_dy

    ! Add routine to path
    call init_routine( routine_name)

    ! if the current model time is at or beyond the point
    ! when the target dH/dt should be removed from the
    ! continuity equation, set its field to 0
    if (region%time >= C%target_dHi_dt_t_end) then
      region%ice%dHi_dt_target = 0._dp
    end if

    ! if the desired time is beyond the time of the next modelled ice thickness,
    ! run the ice dynamics model to calculate ice velocities, thinning rates,
    ! and a new next modelled ice thickness.
    ! ======================================

    if (region%time == region%ice%t_Hi_next) then
      ! Need to calculate new ice velocities, thinning rates, and predicted ice thickness

      ! Start with the maximum allowed ice model time step
      dt_max = C%dt_ice_max

      ! Limit time step during the model start-up phase
      ! FIXME

      ! Run the ice dynamics model to calculate ice velocities, thinning rates,
      ! and a new next modelled ice thickness.
      select case (C%choice_timestepping)
      case default
        call crash('unknown choice_timestepping "' // trim( C%choice_timestepping) // '"!')
      case ('direct')
        call run_ice_dynamics_model_direct( region, dt_max)
      case ('pc')
        call run_ice_dynamics_model_pc( region, dt_max)
      end select

    elseif (region%time > region%ice%t_Hi_next) then
      ! This should not be possible
      call crash('overshot the ice dynamics time step')
    else
      ! We're within the current ice dynamics prediction window
    end if

    ! Interpolate between previous and next modelled ice thickness
    ! to find the geometry at the desired time
    ! ========================================

    ! Calculate time interpolation weights
    wt_prev = (region%ice%t_Hi_next - region%time) / (region%ice%t_Hi_next - region%ice%t_Hi_prev)
    wt_next = 1._dp - wt_prev

    ! Interpolate modelled ice thickness to desired time
    do vi = region%mesh%vi1, region%mesh%vi2
      region%ice%Hi( vi) = wt_prev * region%ice%Hi_prev( vi) + wt_next * region%ice%Hi_next( vi)
    end do

    ! Calculate all other ice geometry quantities
    ! ===========================================

    do vi = region%mesh%vi1, region%mesh%vi2

      ! Basic geometry
      region%ice%Hs ( vi) = ice_surface_elevation( region%ice%Hi( vi), region%ice%Hb( vi), region%ice%SL( vi))
      region%ice%Hib( vi) = region%ice%Hs( vi) - region%ice%Hi( vi)
      region%ice%TAF( vi) = thickness_above_floatation( region%ice%Hi( vi), region%ice%Hb( vi), region%ice%SL( vi))

      ! Differences w.r.t. present-day
      region%ice%dHi ( vi)  = region%ice%Hi ( vi) - region%refgeo_PD%Hi ( vi)
      region%ice%dHb ( vi)  = region%ice%Hb ( vi) - region%refgeo_PD%Hb ( vi)
      region%ice%dHs ( vi)  = region%ice%Hs ( vi) - region%refgeo_PD%Hs ( vi)
      region%ice%dHib( vi)  = region%ice%Hib( vi) - (region%refgeo_PD%Hs ( vi) - region%refgeo_PD%Hi( vi))

      ! Rates of change
      region%ice%dHi_dt( vi) = (region%ice%Hi_next( vi) - region%ice%Hi_prev( vi)) / (region%ice%t_Hi_next - region%ice%t_Hi_prev)
      if (region%ice%TAF( vi) > 0._dp) then
        ! Grounded ice
        region%ice%dHs_dt ( vi) = region%ice%dHb_dt( vi) + region%ice%dHi_dt( vi)
        region%ice%dHib_dt( vi) = region%ice%dHb_dt( vi)
      else
        ! Floating ice
        region%ice%dHs_dt ( vi) = region%ice%dHi_dt( vi) * (1._dp - ice_density / seawater_density)
        region%ice%dHib_dt( vi) = region%ice%dHi_dt( vi) *          ice_density / seawater_density
      end if

    end do

    ! Update masks
    call determine_masks( region%mesh, region%ice)

    ! Calculate new effective thickness
    call calc_effective_thickness( region%mesh, region%ice, region%ice%Hi, region%ice%Hi_eff, region%ice%fraction_margin)

    ! Calculate ice shelf draft gradients
    call ddx_a_b_2D( region%mesh, region%ice%Hib, region%ice%dHib_dx_b)
    call ddy_a_b_2D( region%mesh, region%ice%Hib, region%ice%dHib_dy_b)

    ! Calculate absolute surface gradient
    call ddx_a_a_2D( region%mesh, region%ice%Hs, dHs_dx)
    call ddy_a_a_2D( region%mesh, region%ice%Hs, dHs_dy)
    region%ice%Hs_slope = SQRT( dHs_dx**2 + dHs_dy**2)

    ! NOTE: as calculating the zeta gradients is quite expensive, only do so when necessary,
    !       i.e. when solving the heat equation or the Blatter-Pattyn stress balance
    ! Calculate zeta gradients
    call calc_zeta_gradients( region%mesh, region%ice)

    ! Calculate sub-grid grounded-area fractions
    call calc_grounded_fractions( region%mesh, region%ice)

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine run_ice_dynamics_model

  subroutine initialise_ice_dynamics_model( mesh, ice, refgeo_init, refgeo_PD, refgeo_GIAeq, GIA, region_name)
    !< Initialise all data fields of the ice module

    ! In- and output variables
    type(type_mesh),               intent(in   ) :: mesh
    type(type_ice_model),          intent(inout) :: ice
    type(type_reference_geometry), intent(in   ) :: refgeo_init
    type(type_reference_geometry), intent(in   ) :: refgeo_PD
    type(type_reference_geometry), intent(in   ) :: refgeo_GIAeq
    type(type_GIA_model),          intent(in   ) :: GIA
    character(len=3),              intent(in   ) :: region_name

    ! Local variables:
    character(len=1024), parameter         :: routine_name = 'initialise_ice_dynamics_model'
    integer                                :: vi
    real(dp), dimension(mesh%vi1:mesh%vi2) :: dHs_dx, dHs_dy

    ! Add routine to path
    call init_routine( routine_name)

    if (par%master) write(*,"(A)") '   Initialising ice dynamics model...'

    ! === Memory allocation ===
    ! =========================

    ! allocate memory
    call allocate_ice_model( mesh, ice)

    ! === Value initialisation ===
    ! ============================

    ! Sea level
    ! =========

    select case (C%choice_sealevel_model)

    case default
      call crash('unknown choice_sealevel_model "' // trim( C%choice_sealevel_model) // '"!')

    case ('fixed')
      ! Fixed sea level
      ice%SL = C%fixed_sealevel

    case ('prescribed')
      ! Sea-level prescribed from external record file
      call crash('Sea level initialisation: prescribed method not implemented yet!')
      ! ice%SL = forcing%sealevel_obs

    case ('eustatic')
      ! Eustatic sea level
      call crash('Sea level initialisation: eustatic method not implemented yet!')
      ! ice%SL = C%initial_guess_sealevel

    case ('SELEN')
      ! Sea level from SELEN
      call crash('Sea level initialisation: SELEN method not implemented yet!')
      ! ice%SL = C%initial_guess_sealevel

    end select

    ! Initialise ice geometry
    ! =======================

    ! ! DENK DROM
    ! if (par%master) call warning('GIA model isnt finished yet - need to include dHb in ice model initialisation')

    ! Basic geometry
    do vi = mesh%vi1, mesh%vi2
      ice%Hb( vi) = refgeo_GIAeq%Hb( vi)
      ice%Hs( vi) = refgeo_init%Hs ( vi)
      ice%Hi( vi) = Hi_from_Hb_Hs_and_SL( ice%Hb( vi), ice%Hs( vi), ice%SL( vi))
    end do

    ! Calculate the no-ice mask
    call calc_mask_noice( mesh, ice)

    ! Apply no-ice mask
    call apply_mask_noice_direct( mesh, ice%mask_noice, ice%Hi)

    ! Apply boundary conditions at the domain border
    call apply_ice_thickness_BC_explicit( mesh, ice%mask_noice, ice%Hb, ice%SL, ice%Hi)

    do vi = mesh%vi1, mesh%vi2

      ! Derived geometry
      ice%Hs ( vi) = ice_surface_elevation( ice%Hi( vi), ice%Hb( vi), ice%SL( vi))
      ice%Hib( vi) = ice%Hs( vi) - ice%Hi( vi)
      ice%TAF( vi) = thickness_above_floatation( ice%Hi( vi), ice%Hb( vi), ice%SL( vi))

      ! Differences w.r.t. present-day
      ice%dHi ( vi)  = ice%Hi ( vi) - refgeo_PD%Hi ( vi)
      ice%dHb ( vi)  = ice%Hb ( vi) - refgeo_PD%Hb ( vi)
      ice%dHs ( vi)  = ice%Hs ( vi) - refgeo_PD%Hs ( vi)
      ice%dHib( vi)  = ice%Hib( vi) - (refgeo_PD%Hs ( vi) - refgeo_PD%Hi( vi))

      ! Rates of change
      ice%dHi_dt ( vi) = 0._dp
      ice%dHb_dt ( vi) = 0._dp
      ice%dHs_dt ( vi) = 0._dp
      ice%dHib_dt( vi) = 0._dp

    end do ! do vi = mesh%vi1, mesh%vi2

    ! Calculate ice shelf draft gradients
    call ddx_a_b_2D( mesh, ice%Hib, ice%dHib_dx_b)
    call ddy_a_b_2D( mesh, ice%Hib, ice%dHib_dy_b)

    ! Calculate zeta gradients
    call calc_zeta_gradients( mesh, ice)

    ! Model states for ice dynamics model
    ice%t_Hi_prev = C%start_time_of_run
    ice%t_Hi_next = C%start_time_of_run
    ice%Hi_prev   = ice%Hi
    ice%Hi_next   = ice%Hi

    ! Initialise masks
    ! ================

    ! call it twice so also the "prev" versions are set
    call determine_masks( mesh, ice)
    call determine_masks( mesh, ice)

    ! Effective ice thickness
    ! =======================

    ! Compute effective thickness at calving fronts
    call calc_effective_thickness( mesh, ice, ice%Hi, ice%Hi_eff, ice%fraction_margin)

    ! Surface gradients
    ! =================

    ! Calculate absolute surface gradient
    call ddx_a_a_2D( mesh, ice%Hs, dHs_dx)
    call ddy_a_a_2D( mesh, ice%Hs, dHs_dy)
    ice%Hs_slope = sqrt( dHs_dx**2 + dHs_dy**2)

    ! Target thinning rates
    ! =====================

    ! Load target dHi_dt for inversions
    if (C%do_target_dHi_dt) then
      call initialise_dHi_dt_target(mesh, ice, region_name)
    else
      ice%dHi_dt_target = 0._dp
    end if

    ! Target surface ice speed
    ! ========================

    ! Load target dHi_dt for inversions
    if (C%do_target_uabs_surf) then
      call initialise_uabs_surf_target(mesh, ice, region_name)
    else
      ice%uabs_surf_target = 0._dp
    end if

    ! Sub-grid fractions
    ! ==================

    ! Initialise bedrock cumulative density functions
    call initialise_bedrock_CDFs( mesh, refgeo_PD, ice, region_name)
    ! Initialise sub-grid grounded-area fractions
    call calc_grounded_fractions( mesh, ice)

    ! Basal conditions
    ! ================

    ! allocate and initialise basal conditions
    call initialise_geothermal_heat_flux(  mesh, ice)
    call initialise_basal_hydrology_model( mesh, ice, region_name)
    call initialise_bed_roughness(         mesh, ice, region_name)

    ! Velocities
    ! ==========

    ! Initialise data for the chosen velocity solver(s)
    call initialise_velocity_solver( mesh, ice, region_name)

    ! Time stepping
    ! =============

    select case (C%choice_timestepping)
    case default
      call crash('unknown choice_timestepping "' // trim( C%choice_timestepping) // '"!')
    case ('direct')
      ! No need to initialise anything here
    case ('pc')
      call initialise_pc_scheme( mesh, ice%pc, region_name)
    end select

    ! Numerical stability info
    ice%dt_ice     = 0._dp
    ice%n_visc_its = 0
    ice%n_Axb_its  = 0

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine initialise_ice_dynamics_model

  subroutine write_to_restart_files_ice_model( mesh, ice, time)
    !< Write to all the restart files for the ice dynamics model

    ! In/output variables:
    type(type_mesh),      intent(in   ) :: mesh
    type(type_ice_model), intent(in   ) :: ice
    real(dp),             intent(in   ) :: time

    ! Local variables:
    character(len=1024), parameter :: routine_name = 'write_to_restart_files_ice_model'

    ! Add routine to path
    call init_routine( routine_name)

    ! First for the velocity solver
    call write_to_restart_file_ice_velocity( mesh, ice, time)

    ! then for the time-stepper
    select case (C%choice_timestepping)
    case default
      call crash('unknown choice_timestepping "' // trim( C%choice_timestepping) // '"!')
    case ('direct')
      ! Direct time stepping doesn't require a restart file
    case ('pc')
      call write_to_restart_file_pc_scheme( mesh, ice%pc, time)
    end select

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine write_to_restart_files_ice_model

  subroutine create_restart_files_ice_model( mesh, ice)
    !< Create all the restart files for the ice dynamics model

    ! In/output variables:
    type(type_mesh),      intent(in   ) :: mesh
    type(type_ice_model), intent(inout) :: ice

    ! Local variables:
    character(len=1024), parameter :: routine_name = 'create_restart_files_ice_model'

    ! Add routine to path
    call init_routine( routine_name)

    ! First for the velocity solver
    call create_restart_file_ice_velocity( mesh, ice)

    ! then for the time-stepper
    select case (C%choice_timestepping)
    case default
      call crash('unknown choice_timestepping "' // trim( C%choice_timestepping) // '"!')
    case ('direct')
      ! Direct time stepping doesn't require a restart file
    case ('pc')
      call create_restart_file_pc_scheme( mesh, ice%pc)
    end select

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine create_restart_files_ice_model

  subroutine remap_ice_dynamics_model( mesh_old, mesh_new, ice, refgeo_PD, SMB, BMB, LMB, AMB, GIA, time, region_name)
    !< Remap/reallocate all the data of the ice dynamics model

    ! In/output variables:
    type(type_mesh),               intent(in   ) :: mesh_old
    type(type_mesh),               intent(inout) :: mesh_new
    type(type_ice_model),          intent(inout) :: ice
    type(type_reference_geometry), intent(in   ) :: refgeo_PD
    type(type_SMB_model),          intent(in   ) :: SMB
    type(type_BMB_model),          intent(in   ) :: BMB
    type(type_LMB_model),          intent(in   ) :: LMB
    type(type_AMB_model),          intent(in   ) :: AMB
    type(type_GIA_model),          intent(in   ) :: GIA
    real(dp),                      intent(in   ) :: time
    character(len=3),              intent(in   ) :: region_name

    ! Local variables:
    character(len=1024), parameter                 :: routine_name = 'remap_ice_dynamics_model'
    integer                                        :: vi, ti, k
    real(dp), dimension(mesh_new%vi1:mesh_new%vi2) :: dHs_dx, dHs_dy
    real(dp)                                       :: Ti_min

    ! Add routine to path
    call init_routine( routine_name)

    ! Print to terminal
    if (par%master) write(0,'(A)') '    Remapping ice model data to the new mesh...'

    ! Remap conserved ice model quantities
    ! ====================================

    ! === Ice-sheet geometry ===
    ! ==========================

    ! Remap basic ice geometry Hi,Hb,Hs,SL
    call remap_basic_ice_geometry( mesh_old, mesh_new, refgeo_PD, GIA, ice)

    ! Remap dHi/dt to improve stability of the P/C scheme after mesh updates
    call map_from_mesh_to_mesh_with_reallocation_2D( mesh_old, mesh_new, ice%dHi_dt, '2nd_order_conservative')

    ! === Thermodynamics and rheology ===
    ! ===================================

    ! Save minimum temperature from the entire Ti field, so we can
    ! replace any 0s that pop up after the remapping with that.
    ! Otherwise, divisions by 0 might occur during the computation
    ! of the ice flow factor A.
    Ti_min = minval(ice%Ti)

    ! Use 2nd-order conservative remapping for the ice temperature.
    call map_from_mesh_to_mesh_with_reallocation_3D( mesh_old, mesh_new, ice%Ti, '2nd_order_conservative')

    ! Make sure that no values are smaller than the original minimum
    ice%Ti = max( ice%Ti, Ti_min)

    ! Predicted model state at next time step
    call reallocate_bounds( ice%Ti_prev, mesh_new%vi1, mesh_new%vi2, mesh_new%nz)  ! [K]  The previous state
    call reallocate_bounds( ice%Ti_next, mesh_new%vi1, mesh_new%vi2, mesh_new%nz)  ! [K]  The next state

    ! Re-initialise
    ice%Ti_prev = ice%Ti
    ice%Ti_next = ice%Ti

    ! reallocate memory for all other data fields
    ! (list copied from ice_model_memory/allocate_ice_model)
    ! ======================================================

    ! === Ice-sheet geometry ===
    ! ==========================

    ! Basic geometry
    ! call reallocate_bounds( ice%Hi    , mesh_new%vi1, mesh_new%vi2)  ! [m] Ice thickness
    ! call reallocate_bounds( ice%Hb    , mesh_new%vi1, mesh_new%vi2)  ! [m] Bedrock elevation (w.r.t. PD sea level)
    ! call reallocate_bounds( ice%Hs    , mesh_new%vi1, mesh_new%vi2)  ! [m] Surface elevation (w.r.t. PD sea level)
    ! call reallocate_bounds( ice%SL    , mesh_new%vi1, mesh_new%vi2)  ! [m] Sea level (geoid) elevation (w.r.t. PD sea level)
    call reallocate_bounds( ice%Hib     , mesh_new%vi1, mesh_new%vi2)  ! [m] Ice base elevation (w.r.t. PD sea level)
    call reallocate_bounds( ice%TAF     , mesh_new%vi1, mesh_new%vi2)  ! [m] Thickness above flotation
    call reallocate_bounds( ice%Hi_eff  , mesh_new%vi1, mesh_new%vi2)  ! [m] Effective ice thickness
    call reallocate_bounds( ice%Hs_slope, mesh_new%vi1, mesh_new%vi2)  ! [-] Absolute surface gradients

    ! Geometry changes
    call reallocate_bounds( ice%dHi  , mesh_new%vi1, mesh_new%vi2)  ! [m] Ice thickness difference (w.r.t. reference)
    call reallocate_bounds( ice%dHb  , mesh_new%vi1, mesh_new%vi2)  ! [m] Bedrock elevation difference (w.r.t. reference)
    call reallocate_bounds( ice%dHs  , mesh_new%vi1, mesh_new%vi2)  ! [m] Surface elevation difference (w.r.t. reference)
    call reallocate_bounds( ice%dHib , mesh_new%vi1, mesh_new%vi2)  ! [m] Base elevation difference (w.r.t. reference)

    ! Rates of change
    ! call reallocate_bounds( ice%dHi_dt       , mesh_new%vi1, mesh_new%vi2)  ! [m yr^-1] Ice thickness rate of change
    call reallocate_bounds( ice%dHb_dt         , mesh_new%vi1, mesh_new%vi2)  ! [m yr^-1] Bedrock elevation rate of change
    call reallocate_bounds( ice%dHs_dt         , mesh_new%vi1, mesh_new%vi2)  ! [m yr^-1] Ice surface elevation rate of change
    call reallocate_bounds( ice%dHib_dt        , mesh_new%vi1, mesh_new%vi2)  ! [m yr^-1] Ice base elevation rate of change
    call reallocate_bounds( ice%dHi_dt_raw     , mesh_new%vi1, mesh_new%vi2)  ! [m yr^-1] Ice thickness rate of change before any imposed modifications
    call reallocate_bounds( ice%dHi_dt_residual, mesh_new%vi1, mesh_new%vi2)  ! [m yr^-1] Residual ice thickness rate of change after imposed modifications

    ! Horizontal derivatives
    call reallocate_bounds( ice%dHib_dx_b, mesh_new%ti1, mesh_new%ti2)  ! [] Horizontal derivative of ice draft on b-grid
    call reallocate_bounds( ice%dHib_dy_b, mesh_new%ti1, mesh_new%ti2)  ! [] Horizontal derivative of ice draft on b-grid

    ! Target quantities
    call reallocate_bounds( ice%dHi_dt_target   , mesh_new%vi1, mesh_new%vi2)  ! [m yr^-1] Target ice thickness rate of change for inversions
    call reallocate_bounds( ice%uabs_surf_target, mesh_new%vi1, mesh_new%vi2)  ! [m yr^-1] Target ice surface speed for inversions

    ! Masks
    call reallocate_bounds( ice%mask_icefree_land      , mesh_new%vi1, mesh_new%vi2)  ! T: ice-free land , F: otherwise
    call reallocate_bounds( ice%mask_icefree_ocean     , mesh_new%vi1, mesh_new%vi2)  ! T: ice-free ocean, F: otherwise
    call reallocate_bounds( ice%mask_grounded_ice      , mesh_new%vi1, mesh_new%vi2)  ! T: grounded ice  , F: otherwise
    call reallocate_bounds( ice%mask_floating_ice      , mesh_new%vi1, mesh_new%vi2)  ! T: floating ice  , F: otherwise
    call reallocate_bounds( ice%mask_icefree_land_prev , mesh_new%vi1, mesh_new%vi2)  ! T: ice-free land , F: otherwise (during previous time step)
    call reallocate_bounds( ice%mask_icefree_ocean_prev, mesh_new%vi1, mesh_new%vi2)  ! T: ice-free ocean, F: otherwise (during previous time step)
    call reallocate_bounds( ice%mask_grounded_ice_prev , mesh_new%vi1, mesh_new%vi2)  ! T: grounded ice  , F: otherwise (during previous time step)
    call reallocate_bounds( ice%mask_floating_ice_prev , mesh_new%vi1, mesh_new%vi2)  ! T: floating ice  , F: otherwise (during previous time step)
    call reallocate_bounds( ice%mask_margin            , mesh_new%vi1, mesh_new%vi2)  ! T: ice next to ice-free, F: otherwise
    call reallocate_bounds( ice%mask_gl_gr             , mesh_new%vi1, mesh_new%vi2)  ! T: grounded ice next to floating ice, F: otherwise
    call reallocate_bounds( ice%mask_gl_fl             , mesh_new%vi1, mesh_new%vi2)  ! T: floating ice next to grounded ice, F: otherwise
    call reallocate_bounds( ice%mask_cf_gr             , mesh_new%vi1, mesh_new%vi2)  ! T: grounded ice next to ice-free water (sea or lake), F: otherwise
    call reallocate_bounds( ice%mask_cf_fl             , mesh_new%vi1, mesh_new%vi2)  ! T: floating ice next to ice-free water (sea or lake), F: otherwise
    call reallocate_bounds( ice%mask_coastline         , mesh_new%vi1, mesh_new%vi2)  ! T: ice-free land next to ice-free ocean, F: otherwise
    ! call reallocate_bounds( ice%mask_noice           , mesh_new%vi1, mesh_new%vi2)  ! T: no ice is allowed here, F: ice is allowed here
    call reallocate_bounds( ice%mask                   , mesh_new%vi1, mesh_new%vi2)  ! Diagnostic, only meant for quick visual inspection in output
    call reallocate_bounds( ice%basin_ID               , mesh_new%vi1, mesh_new%vi2)  ! The drainage basin to which each vertex belongs

    ! Area fractions
    call reallocate_bounds( ice%fraction_gr    , mesh_new%vi1, mesh_new%vi2)  ! [0-1] Grounded area fractions of vertices
    call reallocate_bounds( ice%fraction_gr_b  , mesh_new%ti1, mesh_new%ti2)  ! [0-1] Grounded area fractions of triangles
    call reallocate_bounds( ice%fraction_margin, mesh_new%vi1, mesh_new%vi2)  ! [0-1] Ice-covered area fractions of ice margins

    ! Sub-grid bedrock cumulative density functions (CDFs)
    call reallocate_bounds( ice%bedrock_cdf  , mesh_new%vi1, mesh_new%vi2, C%subgrid_bedrock_cdf_nbins)  ! [-] Sub-grid bedrock cumulative density functions on the a-grid (vertices)
    call reallocate_bounds( ice%bedrock_cdf_b, mesh_new%ti1, mesh_new%ti2, C%subgrid_bedrock_cdf_nbins)  ! [-] Sub-grid bedrock cumulative density functions on the b-grid (triangles)

    ! === Terrain-following coordinate zeta gradients ===
    ! ===================================================

    ! Gradients of the terrain-following (i.e. ice-geometry-dependent) vertical coordinate zeta

    ! On the ak-grid (vertices, vertically regular)
    call reallocate_bounds( ice%dzeta_dt_ak   , mesh_new%vi1, mesh_new%vi2, mesh_new%nz)
    call reallocate_bounds( ice%dzeta_dx_ak   , mesh_new%vi1, mesh_new%vi2, mesh_new%nz)
    call reallocate_bounds( ice%dzeta_dy_ak   , mesh_new%vi1, mesh_new%vi2, mesh_new%nz)
    call reallocate_bounds( ice%dzeta_dz_ak   , mesh_new%vi1, mesh_new%vi2, mesh_new%nz)
    call reallocate_bounds( ice%d2zeta_dx2_ak , mesh_new%vi1, mesh_new%vi2, mesh_new%nz)
    call reallocate_bounds( ice%d2zeta_dxdy_ak, mesh_new%vi1, mesh_new%vi2, mesh_new%nz)
    call reallocate_bounds( ice%d2zeta_dy2_ak , mesh_new%vi1, mesh_new%vi2, mesh_new%nz)

    ! On the bk-grid (triangles, vertically regular)
    call reallocate_bounds( ice%dzeta_dx_bk   , mesh_new%ti1, mesh_new%ti2, mesh_new%nz)
    call reallocate_bounds( ice%dzeta_dy_bk   , mesh_new%ti1, mesh_new%ti2, mesh_new%nz)
    call reallocate_bounds( ice%dzeta_dz_bk   , mesh_new%ti1, mesh_new%ti2, mesh_new%nz)
    call reallocate_bounds( ice%d2zeta_dx2_bk , mesh_new%ti1, mesh_new%ti2, mesh_new%nz)
    call reallocate_bounds( ice%d2zeta_dxdy_bk, mesh_new%ti1, mesh_new%ti2, mesh_new%nz)
    call reallocate_bounds( ice%d2zeta_dy2_bk , mesh_new%ti1, mesh_new%ti2, mesh_new%nz)

    ! On the bks-grid (triangles, vertically staggered)
    call reallocate_bounds( ice%dzeta_dx_bks   , mesh_new%ti1, mesh_new%ti2, mesh_new%nz-1)
    call reallocate_bounds( ice%dzeta_dy_bks   , mesh_new%ti1, mesh_new%ti2, mesh_new%nz-1)
    call reallocate_bounds( ice%dzeta_dz_bks   , mesh_new%ti1, mesh_new%ti2, mesh_new%nz-1)
    call reallocate_bounds( ice%d2zeta_dx2_bks , mesh_new%ti1, mesh_new%ti2, mesh_new%nz-1)
    call reallocate_bounds( ice%d2zeta_dxdy_bks, mesh_new%ti1, mesh_new%ti2, mesh_new%nz-1)
    call reallocate_bounds( ice%d2zeta_dy2_bks , mesh_new%ti1, mesh_new%ti2, mesh_new%nz-1)

    ! === Thermodynamics and rheology ===
    ! ===================================

    ! Ice temperatures
    ! call reallocate_bounds( ice%Ti  , mesh_new%vi1, mesh_new%vi2, mesh_new%nz)  ! [K] Englacial temperature
    call reallocate_bounds( ice%Ti_pmp, mesh_new%vi1, mesh_new%vi2, mesh_new%nz)  ! [K] Pressure melting point temperature
    call reallocate_bounds( ice%Ti_hom, mesh_new%vi1, mesh_new%vi2             )  ! [K] Basal temperature w.r.t. pressure melting point

    ! Physical quantities
    call reallocate_bounds( ice%Cpi, mesh_new%vi1, mesh_new%vi2, mesh_new%nz)  ! [J kg^-1 K^-1] Specific heat capacity
    call reallocate_bounds( ice%Ki , mesh_new%vi1, mesh_new%vi2, mesh_new%nz)  ! [J m^-1 K^-1 yr^-1] Thermal conductivity

    ! Heating
    call reallocate_bounds( ice%internal_heating  , mesh_new%vi1, mesh_new%vi2, mesh_new%nz)  ! [?] Internal heating
    call reallocate_bounds( ice%frictional_heating, mesh_new%vi1, mesh_new%vi2             )  ! [?] Frictional heating

    ! Glen's flow law factor
    call reallocate_bounds( ice%A_flow, mesh_new%vi1, mesh_new%vi2, mesh_new%nz)  ! [Pa^-3 y^-1] Glen's flow law factor

    ! === Ice velocities ===
    ! ======================

    ! 3-D
    call reallocate_bounds( ice%u_3D  , mesh_new%vi1, mesh_new%vi2, mesh_new%nz)  ! [m yr^-1] 3-D ice velocity
    call reallocate_bounds( ice%v_3D  , mesh_new%vi1, mesh_new%vi2, mesh_new%nz)
    call reallocate_bounds( ice%u_3D_b, mesh_new%ti1, mesh_new%ti2, mesh_new%nz)
    call reallocate_bounds( ice%v_3D_b, mesh_new%ti1, mesh_new%ti2, mesh_new%nz)
    call reallocate_bounds( ice%w_3D  , mesh_new%vi1, mesh_new%vi2, mesh_new%nz)

    ! Vertically integrated
    call reallocate_bounds( ice%u_vav     , mesh_new%vi1, mesh_new%vi2)  ! [m yr^-1] Vertically averaged ice velocity
    call reallocate_bounds( ice%v_vav     , mesh_new%vi1, mesh_new%vi2)
    call reallocate_bounds( ice%u_vav_b   , mesh_new%ti1, mesh_new%ti2)
    call reallocate_bounds( ice%v_vav_b   , mesh_new%ti1, mesh_new%ti2)
    call reallocate_bounds( ice%uabs_vav  , mesh_new%vi1, mesh_new%vi2)
    call reallocate_bounds( ice%uabs_vav_b, mesh_new%ti1, mesh_new%ti2)

    ! Surface
    call reallocate_bounds( ice%u_surf     , mesh_new%vi1, mesh_new%vi2)  ! [m yr^-1] Ice velocity at the surface
    call reallocate_bounds( ice%v_surf     , mesh_new%vi1, mesh_new%vi2)
    call reallocate_bounds( ice%u_surf_b   , mesh_new%ti1, mesh_new%ti2)
    call reallocate_bounds( ice%v_surf_b   , mesh_new%ti1, mesh_new%ti2)
    call reallocate_bounds( ice%w_surf     , mesh_new%vi1, mesh_new%vi2)
    call reallocate_bounds( ice%uabs_surf  , mesh_new%vi1, mesh_new%vi2)
    call reallocate_bounds( ice%uabs_surf_b, mesh_new%ti1, mesh_new%ti2)

    ! Basal
    call reallocate_bounds( ice%u_base     , mesh_new%vi1, mesh_new%vi2)  ! [m yr^-1] Ice velocity at the base
    call reallocate_bounds( ice%v_base     , mesh_new%vi1, mesh_new%vi2)
    call reallocate_bounds( ice%u_base_b   , mesh_new%ti1, mesh_new%ti2)
    call reallocate_bounds( ice%v_base_b   , mesh_new%ti1, mesh_new%ti2)
    call reallocate_bounds( ice%w_base     , mesh_new%vi1, mesh_new%vi2)
    call reallocate_bounds( ice%uabs_base  , mesh_new%vi1, mesh_new%vi2)
    call reallocate_bounds( ice%uabs_base_b, mesh_new%ti1, mesh_new%ti2)

    ! == Strain rates ==
    ! ==================

    call reallocate_bounds( ice%du_dx_3D, mesh_new%vi1, mesh_new%vi2, mesh_new%nz)  ! [yr^-1]
    call reallocate_bounds( ice%du_dy_3D, mesh_new%vi1, mesh_new%vi2, mesh_new%nz)
    call reallocate_bounds( ice%du_dz_3D, mesh_new%vi1, mesh_new%vi2, mesh_new%nz)
    call reallocate_bounds( ice%dv_dx_3D, mesh_new%vi1, mesh_new%vi2, mesh_new%nz)
    call reallocate_bounds( ice%dv_dy_3D, mesh_new%vi1, mesh_new%vi2, mesh_new%nz)
    call reallocate_bounds( ice%dv_dz_3D, mesh_new%vi1, mesh_new%vi2, mesh_new%nz)
    call reallocate_bounds( ice%dw_dx_3D, mesh_new%vi1, mesh_new%vi2, mesh_new%nz)
    call reallocate_bounds( ice%dw_dy_3D, mesh_new%vi1, mesh_new%vi2, mesh_new%nz)
    call reallocate_bounds( ice%dw_dz_3D, mesh_new%vi1, mesh_new%vi2, mesh_new%nz)

    ! == Ice flow regime ==
    ! =====================

    call reallocate_bounds( ice%divQ   , mesh_new%vi1, mesh_new%vi2)  ! [m yr^-1] Horizontal ice flux divergence
    call reallocate_bounds( ice%R_shear, mesh_new%vi1, mesh_new%vi2)  ! [0-1]     uabs_base / uabs_surf (0 = pure vertical shear, viscous flow; 1 = pure sliding, plug flow)

    ! == Basal hydrology ==
    ! =====================

    ! Basal hydrology
    call reallocate_bounds( ice%pore_water_pressure  , mesh_new%vi1, mesh_new%vi2)  ! [Pa]  Basal pore water pressure
    call reallocate_bounds( ice%overburden_pressure  , mesh_new%vi1, mesh_new%vi2)  ! [Pa]  Basal overburden pressure
    call reallocate_bounds( ice%effective_pressure   , mesh_new%vi1, mesh_new%vi2)  ! [Pa]  Basal effective pressure
    call reallocate_bounds( ice%pore_water_likelihood, mesh_new%vi1, mesh_new%vi2)  ! [0-1] Basal pore water likelihood
    call reallocate_bounds( ice%pore_water_fraction  , mesh_new%vi1, mesh_new%vi2)  ! [0-1] Fraction of overburden pressure reduced by pore water pressure

    ! == Basal sliding ==
    ! ===================

    ! Sliding law coefficients
    call reallocate_bounds( ice%till_friction_angle, mesh_new%vi1, mesh_new%vi2)  ! [degrees]          Till friction angle (degrees)
    call reallocate_bounds( ice%bed_roughness      , mesh_new%vi1, mesh_new%vi2)  ! [0-1]              Bed roughness fraction
    call reallocate_bounds( ice%till_yield_stress  , mesh_new%vi1, mesh_new%vi2)  ! [Pa]               Till yield stress (used when choice_sliding_law = "Coloumb", "Budd", or "Zoet-Iverson")
    call reallocate_bounds( ice%slid_alpha_sq      , mesh_new%vi1, mesh_new%vi2)  ! [-]                Coulomb-law friction coefficient (used when choice_sliding_law = "Tsai2015", or "Schoof2005")
    call reallocate_bounds( ice%slid_beta_sq       , mesh_new%vi1, mesh_new%vi2)  ! [Pa m^−1/m yr^1/m] Power-law friction coefficient (used when choice_sliding_law = "Weertman", "Tsai2015", or "Schoof2005")

    ! Basal friction and shear stress
    call reallocate_bounds( ice%basal_friction_coefficient, mesh_new%vi1, mesh_new%vi2)  ! [Pa yr m^-1]       Effective basal friction coefficient (basal_shear_stress = u_base * basal_friction_coefficient)
    call reallocate_bounds( ice%basal_shear_stress        , mesh_new%vi1, mesh_new%vi2)  ! [Pa]               Basal shear stress

    ! == Geothermal heat ==
    ! =====================

    call reallocate_bounds( ice%geothermal_heat_flux, mesh_new%vi1, mesh_new%vi2)  ! [J m^-2 yr^-1] Geothermal heat flux

    ! === Ice thickness time stepping ===
    ! ===================================

    ! Predicted model state at next time step
    call reallocate_bounds( ice%Hi_prev, mesh_new%vi1, mesh_new%vi2)  ! [m]  The previous state
    call reallocate_bounds( ice%Hi_next, mesh_new%vi1, mesh_new%vi2)  ! [m]  The next state

    ! Re-initialise the rest of the ice dynamics model
    ! ================================================

    ! Initialise ice geometry
    ! =======================

    do vi = mesh_new%vi1, mesh_new%vi2

      ! Basic geometry
      ! ice%Hi ( vi) = refgeo_init%Hi( vi)
      ! ice%Hb ( vi) = refgeo_init%Hb( vi)
      ice%Hs ( vi) = ice_surface_elevation( ice%Hi( vi), ice%Hb( vi), ice%SL( vi))
      ice%Hib( vi) = ice%Hs( vi) - ice%Hi( vi)
      ice%TAF( vi) = thickness_above_floatation( ice%Hi( vi), ice%Hb( vi), ice%SL( vi))

      ! Differences w.r.t. present-day
      ice%dHi ( vi)  = ice%Hi ( vi) - refgeo_PD%Hi ( vi)
      ice%dHb ( vi)  = ice%Hb ( vi) - refgeo_PD%Hb ( vi)
      ice%dHs ( vi)  = ice%Hs ( vi) - refgeo_PD%Hs ( vi)
      ice%dHib( vi)  = ice%Hib( vi) - (refgeo_PD%Hs ( vi) - refgeo_PD%Hi( vi))

      ! Rates of change
      ice%dHi_dt ( vi) = 0._dp
      ice%dHb_dt ( vi) = 0._dp
      ice%dHs_dt ( vi) = 0._dp
      ice%dHib_dt( vi) = 0._dp

    end do ! do vi = mesh_new%vi1, mesh_new%vi2

    do ti = mesh_new%ti1, mesh_new%ti2
      ! Horizontal derivatives
      ice%dHib_dx_b( ti) = 0._dp
      ice%dHib_dy_b( ti) = 0._dp
    end do ! do ti = mesh_new%ti1, mesh_new%ti2

    ! Calculate zeta gradients
    call calc_zeta_gradients( mesh_new, ice)

    ! Load target dHi_dt for inversions
    if (C%do_target_dHi_dt) then
      call initialise_dHi_dt_target(mesh_new, ice, region_name)
    else
      ice%dHi_dt_target = 0._dp
    end if

    ! Load target surface ice speed for inversions
    if (C%do_target_uabs_surf) then
      call initialise_uabs_surf_target(mesh_new, ice, region_name)
    else
      ice%uabs_surf_target = 0._dp
    end if

    ! Model states for ice dynamics model
    ice%t_Hi_prev = time
    ice%t_Hi_next = time
    ice%Hi_prev   = ice%Hi
    ice%Hi_next   = ice%Hi

    ! Initialise masks
    ! ================

    ! Calculate the no-ice mask
    call calc_mask_noice( mesh_new, ice)

    ! Remove ice bleed into forbidden areas
    call apply_mask_noice_direct( mesh_new, ice%mask_noice, ice%Hi)
    call apply_mask_noice_direct( mesh_new, ice%mask_noice, ice%dHi_dt)

    ! call it twice so also the "prev" versions are set
    call determine_masks( mesh_new, ice)
    call determine_masks( mesh_new, ice)

    ! ! Smooth the ice at the calving front to improve model stability
    ! call relax_calving_front_after_mesh_update( mesh_new, ice)

    ! Effective ice thickness
    ! =======================

    ! Calculate new effective thickness
    call calc_effective_thickness( mesh_new, ice, ice%Hi, ice%Hi_eff, ice%fraction_margin)

    ! Surface gradients
    ! =================

    ! Calculate absolute surface gradient
    call ddx_a_a_2D( mesh_new, ice%Hs, dHs_dx)
    call ddy_a_a_2D( mesh_new, ice%Hs, dHs_dy)
    ice%Hs_slope = sqrt( dHs_dx**2 + dHs_dy**2)

    ! Sub-grid fractions
    ! ==================

    if (C%choice_subgrid_grounded_fraction == 'bedrock_CDF' .OR. &
        C%choice_subgrid_grounded_fraction == 'bilin_interp_TAF+bedrock_CDF') then
      ! Compute bedrock cumulative density function
      call calc_bedrock_CDFs( mesh_new, refgeo_PD, ice)
    end if
    ! Initialise sub-grid grounded-area fractions
    call calc_grounded_fractions( mesh_new, ice)

    ! Basal conditions
    ! ================

    ! allocate and initialise basal conditions
    call initialise_geothermal_heat_flux(  mesh_new, ice)
    call initialise_basal_hydrology_model( mesh_new, ice, region_name)

    ! FIXME: something should happen here once we start working on remapping of inverted bed roughness!
    call initialise_bed_roughness(         mesh_new, ice, region_name)

    ! Velocities
    ! ==========

    ! Remap data for the chosen velocity solver(s)
    call remap_velocity_solver( mesh_old, mesh_new, ice)

    ! Time stepping
    ! =============

    select case (C%choice_timestepping)
    case default
      call crash('unknown choice_timestepping "' // trim( C%choice_timestepping) // '"!')
    case ('direct')
      ! No need to initialise anything here
    case ('pc')
      call remap_pc_scheme( mesh_old, mesh_new, ice%pc)
    end select

    ! Relax ice geometry around the calving front
    ! ===========================================

    call relax_calving_front( mesh_old, mesh_new, ice, SMB, BMB, LMB, AMB, region_name)

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine remap_ice_dynamics_model

  subroutine remap_basic_ice_geometry( mesh_old, mesh_new, refgeo_PD, GIA, ice)
    !< Remap the basic ice geometry Hi,Hb,Hs,SL.

    ! In/output variables:
    type(type_mesh),               intent(in   ) :: mesh_old
    type(type_mesh),               intent(in   ) :: mesh_new
    type(type_reference_geometry), intent(in   ) :: refgeo_PD
    type(type_GIA_model),          intent(in   ) :: GIA
    type(type_ice_model),          intent(inout) :: ice

    ! Local variables:
    character(len=1024), parameter                  :: routine_name = 'remap_basic_ice_geometry'
    real(dp), dimension( mesh_old%nV)               :: Hi_old_tot
    logical,  dimension( mesh_old%nV)               :: mask_floating_ice_tot
    logical,  dimension( mesh_old%nV)               :: mask_icefree_ocean_tot
    real(dp), dimension( mesh_new%vi1:mesh_new%vi2) :: Hi_new
    real(dp), dimension( mesh_new%vi1:mesh_new%vi2) :: Hs_new
    integer                                         :: vi
    integer                                         :: mi, mi_used
    logical                                         :: found_map
    type(type_sparse_matrix_CSR_dp)                 :: M_CSR
    integer                                         :: vi_new, k1, k2, k, vi_old
    integer                                         :: n_ice, n_nonice
    integer                                         :: n_shelf, n_open_ocean
    real(dp)                                        :: sum_Hi_shelf

    ! Add routine to path
    call init_routine( routine_name)

    ! == Basic: remap surface elevation Hs from the old mesh, remap bedrock elevation Hb
    !    from its (presumably high-resolution) source grid, define remapped ice thickness
    !    as the difference between the two. As surface elevation is typically much smoother
    !    then ice thickness, remapping works much better.
    ! =====================================================================================

    ! Remap bedrock from the original high-resolution grid, and add the (very smooth) modelled deformation to it
    ! Remapping of Hb in the refgeo structure has already happened, only need to copy the data
    if (par%master) call warning('GIA model isnt finished yet - need to include dHb in mesh update!')
    call reallocate_bounds( ice%Hb, mesh_new%vi1, mesh_new%vi2)  ! [m] Bedrock elevation (w.r.t. PD sea level)
    ice%Hb = refgeo_PD%Hb

    ! Remap sea level
    if (par%master) call warning('sea  model isnt finished yet - need to include dSL in mesh update!')
    call reallocate_bounds( ice%SL, mesh_new%vi1, mesh_new%vi2)  ! [m] Sea level (geoid) elevation (w.r.t. PD sea level)
    ice%SL = 0._dp

    ! Gather global ice thickness and masks
    call gather_to_all(      ice%Hi                , Hi_old_tot            )
    call gather_to_all( ice%mask_floating_ice , mask_floating_ice_tot )
    call gather_to_all( ice%mask_icefree_ocean, mask_icefree_ocean_tot)

    ! First, naively remap ice thickness and surface elevation without any restrictions
    call map_from_mesh_to_mesh_2D( mesh_old, mesh_new, ice%Hi, Hi_new, '2nd_order_conservative')
    call map_from_mesh_to_mesh_2D( mesh_old, mesh_new, ice%Hs, Hs_new, '2nd_order_conservative')

    ! Calculate remapped ice thickness as the difference between new bedrock and remapped surface elevation
    do vi = mesh_new%vi1, mesh_new%vi2
      if (Hi_new( vi) > 0._dp) then
        if (Hs_new( vi) <= ice%Hb( vi)) then
          Hi_new( vi) = 0._dp
        else
          Hi_new( vi) = Hi_from_Hb_Hs_and_SL( ice%Hb( vi), Hs_new( vi), ice%SL( vi))
        end if
      else
        Hi_new( vi) = 0._dp
      end if
    end do ! do vi = mesh_new%vi1, mesh_new%vi2

    ! reallocate no-ice mask
    ! T: no ice is allowed here, F: ice is allowed here
    call reallocate_bounds( ice%mask_noice, mesh_new%vi1, mesh_new%vi2)

    ! Apply boundary conditions at the domain border
    call calc_mask_noice( mesh_new, ice)
    call apply_ice_thickness_BC_explicit( mesh_new, ice%mask_noice, ice%Hb, ice%SL, Hi_new)

    ! == Corrections
    ! ==============

    ! Browse the Atlas to find the map between mesh_old and mesh_new
    found_map = .false.
    mi_used   = 0
    do mi = 1, size( Atlas, 1)
      if (Atlas( mi)%name_src == mesh_old%name .and. Atlas( mi)%name_dst == mesh_new%name &
          .and. Atlas( mi)%method == '2nd_order_conservative') then
        found_map = .true.
        mi_used  = mi
        exit
      end if
    end do
    ! Safety
    if (.not. found_map) call crash('couldnt find which map was used')

    ! Convert the mapping matrix to CSR format
    call mat_petsc2CSR( Atlas( mi_used)%M, M_CSR)

    ! == For those vertices of the new mesh that overlap with both old-mesh ice and old-mesh
    !    non-ice, remove very thin remapped ice
    ! ======================================================================================

    do vi_new = mesh_new%vi1, mesh_new%vi2

      k1 = M_CSR%ptr( vi_new)
      k2 = M_CSR%ptr( vi_new+1) - 1

      n_ice    = 0
      n_nonice = 0

      do k = k1, k2

        vi_old = M_CSR%ind( k)

        if     (Hi_old_tot( vi_old) > 1._dp) then
          n_ice = n_ice + 1
        elseif (Hi_old_tot( vi_old) < 1._dp) then
          n_nonice = n_nonice + 1
        end if

      end do ! do k = k1, k2

      if (n_ice > 0 .and. n_nonice > 0) then
        ! This new-mesh vertex overlaps with both old-mesh ice vertices,
        ! and old-mesh non-ice vertices
        if (Hi_new( vi_new) < 1._dp) then
          ! Remove very thin remapped ice

          Hi_new( vi_new) = 0._dp
        end if
      end if

    end do ! do vi_new = mesh_new%vi1, mesh_new%vi2

    ! == For those vertices of the new mesh that overlap with both old-mesh shelf and old-mesh
    !    open ocean, average only over the contributing old-mesh shelf vertices
    ! ======================================================================================

    do vi_new = mesh_new%vi1, mesh_new%vi2

      k1 = M_CSR%ptr( vi_new)
      k2 = M_CSR%ptr( vi_new+1) - 1

      n_shelf      = 0
      n_open_ocean = 0
      sum_Hi_shelf = 0._dp

      do k = k1, k2

        vi_old = M_CSR%ind( k)

        if     (mask_floating_ice_tot(  vi_old)) then
          n_shelf      = n_shelf      + 1
          sum_Hi_shelf = sum_Hi_shelf + Hi_old_tot( vi_old)
        elseif (mask_icefree_ocean_tot( vi_old)) then
          n_open_ocean = n_open_ocean + 1
        end if

      end do ! do k = k1, k2

      if (n_shelf > 0 .and. n_open_ocean > 0) then
        ! This new-mesh vertex overlaps with both old-mesh shelf vertices,
        ! and old-mesh open-ocean vertices
        Hi_new( vi_new) = sum_Hi_shelf / real( n_shelf,dp)
      end if

    end do ! do vi_new = mesh_new%vi1, mesh_new%vi2

    ! Recalculate Hs
    call reallocate_bounds( ice%Hs, mesh_new%vi1, mesh_new%vi2)
    do vi = mesh_new%vi1, mesh_new%vi2
      ice%Hs( vi) = ice_surface_elevation( Hi_new( vi), ice%Hb( vi), ice%SL( vi))
    end do

    ! Move Hi_new to ice%Hi
    call reallocate_bounds( ice%Hi, mesh_new%vi1, mesh_new%vi2)
    ice%Hi = Hi_new

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine remap_basic_ice_geometry

  subroutine relax_calving_front( mesh_old, mesh, ice, SMB, BMB, LMB, AMB, region_name)
    !< Relax ice thickness around the calving front

    ! This routine "steps out of time" for a bit (default dt_relax = 2 yr), where it
    ! lets the ice thickness near the calving front relax for a little bit. To achieve
    ! this, it uses the BC_prescr_mask optional arguments of the velocity solver and the
    ! thickness solver to keep velocities and thickness fixed over parts of the domain.
    ! In this case, it keeps them fixed everywhere except over the open ocean, and the
    ! first 5 shelf vertices inward from the calving front. This allows the little bumps
    ! that appear in the ice thickness after remapping, which seriously slow down the
    ! velocity solution, to relax a little.

    ! In/output variables:
    type(type_mesh),      intent(in   ) :: mesh_old
    type(type_mesh),      intent(inout) :: mesh
    type(type_ice_model), intent(inout) :: ice
    type(type_SMB_model), intent(in   ) :: SMB
    type(type_BMB_model), intent(in   ) :: BMB
    type(type_LMB_model), intent(in   ) :: LMB
    type(type_AMB_model), intent(in   ) :: AMB
    character(len=3),     intent(in   ) :: region_name

    ! Local variables:
    character(len=1024), parameter                 :: routine_name = 'relax_calving_front'
    logical,  dimension(mesh%nV)                   :: mask_icefree_ocean_tot
    logical,  dimension(mesh%nV)                   :: mask_floating_ice_tot
    logical,  dimension(mesh%nV)                   :: mask_cf_fl_tot
    integer,  dimension(mesh%nV)                   :: BC_prescr_mask_tot
    integer,  dimension(mesh%nTri)                 :: BC_prescr_mask_b_tot
    integer,  dimension(mesh%nTri,mesh%nz)         :: BC_prescr_mask_bk_tot
    integer,  dimension(mesh%vi1:mesh%vi2)         :: BC_prescr_mask
    integer,  dimension(mesh%ti1:mesh%ti2)         :: BC_prescr_mask_b
    integer,  dimension(mesh%ti1:mesh%ti2,mesh%nz) :: BC_prescr_mask_bk
    integer,  dimension(mesh%nV)                   :: map
    integer,  dimension(mesh%nV)                   :: stack
    integer                                        :: stackN
    integer,  dimension(mesh%nV)                   :: stack2
    integer                                        :: stack2N
    integer                                        :: vi
    integer                                        :: it
    integer,  parameter                            :: nV_around_calving_front = 5
    integer                                        :: i
    integer                                        :: ci,vj
    integer                                        :: ti,via, vib, vic
    real(dp), dimension(mesh%vi1:mesh%vi2)         :: BC_prescr_Hi
    real(dp), dimension(mesh%ti1:mesh%ti2)         :: BC_prescr_u_b
    real(dp), dimension(mesh%ti1:mesh%ti2)         :: BC_prescr_v_b
    real(dp), dimension(mesh%ti1:mesh%ti2,mesh%nz) :: BC_prescr_u_bk
    real(dp), dimension(mesh%ti1:mesh%ti2,mesh%nz) :: BC_prescr_v_bk
    real(dp), parameter                            :: dt_relax = 2._dp   ! [yr] Time to relax the ice around the calving front
    real(dp)                                       :: t_pseudo
    real(dp), dimension(mesh%vi1:mesh%vi2)         :: SMB_new
    real(dp), dimension(mesh%vi1:mesh%vi2)         :: BMB_new
    real(dp), dimension(mesh%vi1:mesh%vi2)         :: LMB_new
    real(dp), dimension(mesh%vi1:mesh%vi2)         :: AMB_new
    real(dp)                                       :: visc_it_norm_dUV_tol_save
    integer                                        :: visc_it_nit_save
    real(dp)                                       :: visc_it_relax_save
    real(dp)                                       :: visc_eff_min_save
    real(dp)                                       :: vel_max_save
    real(dp)                                       :: stress_balance_PETSc_rtol_save
    real(dp)                                       :: stress_balance_PETSc_abstol_save
    real(dp)                                       :: Glens_flow_law_epsilon_sq_0_save
    real(dp), dimension(mesh%vi1:mesh%vi2)         :: Hi_tplusdt
    real(dp), dimension(mesh%vi1:mesh%vi2)         :: divQ
    integer                                        :: n_visc_its
    integer                                        :: n_Axb_its

    ! Add routine to path
    call init_routine( routine_name)

    ! Gather global masks
    call gather_to_all( ice%mask_icefree_ocean, mask_icefree_ocean_tot)
    call gather_to_all( ice%mask_floating_ice , mask_floating_ice_tot )
    call gather_to_all( ice%mask_cf_fl        , mask_cf_fl_tot        )

    ! == Create the relaxation mask
    ! =============================

    ! Create a mask over the entire ice-free ocean, and 5 vertices into the shelf
    ! NOTE: 0 = velocities and dH/dt are solved
    !       1 = velocities and H are prescribed at remapped values)

    ! Let the master do this (difficult to parallelise)
    if (par%master) then

      ! Initialise mask
      BC_prescr_mask_tot   = 1

      ! First mark the open ocean on the mask
      do vi = 1, mesh%nV
        if (mask_icefree_ocean_tot( vi)) then
          BC_prescr_mask_tot( vi) = 0
        end if
      end do ! do vi = 1, mesh%nV

      ! Initialise flood-fill
      map    = 0
      stackN = 0

      ! Initialise with the floating calving front
      do vi = 1, mesh%nV
        if (mask_cf_fl_tot( vi)) then
          map( vi) = 1
          stackN = stackN + 1
          stack( stackN) = vi
        end if
      end do ! do vi = 1, mesh%nV

      ! Expand into shelf
      do it = 1, nV_around_calving_front

        ! Initialise the second stack
        stack2N = 0

        ! Go over the entire stack
        do i = 1, stackN

          ! Take a vertex from the stack
          vi = stack( i)

          ! Mark it on the mask
          BC_prescr_mask_tot( vi) = 0

          ! Mark it on the map
          map( vi) = 2

          ! Add all its un-mapped, floating neighbours to the new stack
          do ci = 1, mesh%nC( vi)
            vj = mesh%C( vi,ci)
            if (map( vj) == 0 .and. mask_floating_ice_tot( vj)) then
              map( vj) = 1
              stack2N = stack2N + 1
              stack2( stack2N) = vj
            end if
          end do !  do ci = 1, mesh%nC( vi)

        end do ! do i = 1, stackN

        ! Cycle the stacks
        stack( 1:stack2N) = stack2( 1:stack2N)
        stackN = stack2N

      end do ! do it = 1, nV_around_calving_front

      ! Fill in the b-grid masks

      BC_prescr_mask_b_tot  = 0
      BC_prescr_mask_bk_tot = 0

      do ti = 1, mesh%nTri

        ! The three vertices spanning triangle ti
        via = mesh%Tri( ti,1)
        vib = mesh%Tri( ti,2)
        vic = mesh%Tri( ti,3)

        ! Only prescribe velocities on triangles where thickness is prescribed on all three corners
        if (BC_prescr_mask_tot( via) == 1 .and. &
            BC_prescr_mask_tot( vib) == 1 .and. &
            BC_prescr_mask_tot( vic) == 1) then
          BC_prescr_mask_b_tot(  ti  ) = 1
          BC_prescr_mask_bk_tot( ti,:) = 1
        end if

      end do ! do ti = 1, mesh%nTri

    end if ! if (par%master) then

    ! Distribute BC masks to all processes
    call distribute_from_master( BC_prescr_mask_tot   , BC_prescr_mask   )
    call distribute_from_master( BC_prescr_mask_b_tot , BC_prescr_mask_b )
    call distribute_from_master( BC_prescr_mask_bk_tot, BC_prescr_mask_bk)

    ! == Fill in prescribed velocities and thicknesses away from the front
    ! ====================================================================

    BC_prescr_Hi   = ice%Hi
    BC_prescr_u_b  = ice%u_vav_b
    BC_prescr_v_b  = ice%v_vav_b
    BC_prescr_u_bk = ice%u_3D_b
    BC_prescr_v_bk = ice%v_3D_b

    ! == Save proper values of config parameters for the velocity solver
    ! ==================================================================

    visc_it_norm_dUV_tol_save                  = C%visc_it_norm_dUV_tol
    visc_it_nit_save                           = C%visc_it_nit
    visc_it_relax_save                         = C%visc_it_relax
    visc_eff_min_save                          = C%visc_eff_min
    vel_max_save                               = C%vel_max
    stress_balance_PETSc_rtol_save             = C%stress_balance_PETSc_rtol
    stress_balance_PETSc_abstol_save           = C%stress_balance_PETSc_abstol
    Glens_flow_law_epsilon_sq_0_save           = C%Glens_flow_law_epsilon_sq_0

    ! == Set temporary, less strict values of config parameters for the velocity solver
    ! =================================================================================

    C%visc_it_norm_dUV_tol                  = 5E-4_dp                          ! Stop criterion for the viscosity iteration: the L2-norm of successive velocity solutions should be smaller than this number
    C%visc_it_nit                           = 20                               ! Maximum number of effective viscosity iterations
    C%visc_it_relax                         = 0.3_dp                           ! Relaxation parameter for subsequent viscosity iterations (for improved stability)
    C%visc_eff_min                          = 1E5_dp                           ! Minimum value for effective viscosity
    C%vel_max                               = 5000._dp                         ! Velocities are limited to this value
    C%stress_balance_PETSc_rtol             = 1E-3_dp                          ! PETSc solver - stop criterion, relative difference (iteration stops if rtol OR abstol is reached)
    C%stress_balance_PETSc_abstol           = 1E-2_dp                          ! PETSc solver - stop criterion, absolute difference
    C%Glens_flow_law_epsilon_sq_0           = 1E-6_dp                          ! Normalisation term so that zero strain rates produce a high but finite viscosity

    ! == Remap SMB, BMB, LMB, and AMB to get more stable ice thickness
    ! ================================================================

    call map_from_mesh_to_mesh_2D( mesh_old, mesh, SMB%SMB, SMB_new, '2nd_order_conservative')
    call map_from_mesh_to_mesh_2D( mesh_old, mesh, BMB%BMB, BMB_new, '2nd_order_conservative')
    call map_from_mesh_to_mesh_2D( mesh_old, mesh, LMB%LMB, LMB_new, '2nd_order_conservative')
    call map_from_mesh_to_mesh_2D( mesh_old, mesh, AMB%AMB, AMB_new, '2nd_order_conservative')

    ! == Relax the ice thickness for a few time steps
    ! ===============================================

    t_pseudo = 0._dp

    pseudo_time: do while (t_pseudo < dt_relax)

      ! Update velocity solution around the calving front
      call solve_stress_balance( mesh, ice, BMB_new, region_name, &
        n_visc_its, n_Axb_its, &
        BC_prescr_mask_b, BC_prescr_u_b, BC_prescr_v_b, BC_prescr_mask_bk, BC_prescr_u_bk, BC_prescr_v_bk)

      ! Calculate dH/dt around the calving front
      call calc_dHi_dt( mesh, ice%Hi, ice%Hb, ice%SL, ice%u_vav_b, ice%v_vav_b, SMB_new, BMB_new, LMB_new, AMB_new, ice%fraction_margin, ice%mask_noice, C%dt_ice_min, &
        ice%dHi_dt, Hi_tplusdt, divQ, ice%dHi_dt_target, BC_prescr_mask, BC_prescr_Hi)

      ! Update ice thickness and advance pseudo-time
      ice%Hi = Hi_tplusdt
      t_pseudo = t_pseudo + C%dt_ice_min

      ! Update basic geometry
      do vi = mesh%vi1, mesh%vi2
        ice%Hs ( vi) = ice_surface_elevation( ice%Hi( vi), ice%Hb( vi), ice%SL( vi))
        ice%Hib( vi) = ice%Hs( vi) - ice%Hi( vi)
        ice%TAF( vi) = thickness_above_floatation( ice%Hi( vi), ice%Hb( vi), ice%SL( vi))
      end do

    end do pseudo_time ! do while (t_pseudo < dt_relax)

    ! == Reinstate proper values of config parameters for the velocity solver
    ! =======================================================================

    C%visc_it_norm_dUV_tol                  = visc_it_norm_dUV_tol_save
    C%visc_it_nit                           = visc_it_nit_save
    C%visc_it_relax                         = visc_it_relax_save
    C%visc_eff_min                          = visc_eff_min_save
    C%vel_max                               = vel_max_save
    C%stress_balance_PETSc_rtol             = stress_balance_PETSc_rtol_save
    C%stress_balance_PETSc_abstol           = stress_balance_PETSc_abstol_save
    C%Glens_flow_law_epsilon_sq_0           = Glens_flow_law_epsilon_sq_0_save

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine relax_calving_front

  subroutine apply_geometry_relaxation( region)
    !< Relax the initial geometry by running the ice dynamics model
    !< for a certain time without any mass balance terms, inversions
    !< or imposed alterations to the evolution of ice thickness

    use, intrinsic :: ISO_C_BINDING, only: c_carriage_return

    ! In/output variables:
    type(type_model_region),                intent(inout) :: region

    ! Local variables:
    character(len=1024), parameter                        :: routine_name = 'apply_geometry_relaxation'
    integer                                               :: vi
    real(dp)                                              :: t_pseudo, t_step
    real(dp)                                              :: visc_it_norm_dUV_tol_config
    real(dp), dimension(region%mesh%vi1:region%mesh%vi2)  :: Hi_new, dHi_dt_new
    real(dp), dimension(region%mesh%vi1:region%mesh%vi2)  :: SMB_dummy, BMB_dummy, LMB_dummy, AMB_dummy, dHi_dt_target_dummy
    character(len=256)                                    :: t_years, r_time, r_step, r_adv, t_format
    integer                                               :: n_visc_its
    integer                                               :: n_Axb_its

    ! Add routine to path
    call init_routine( routine_name)

    t_pseudo = 0._dp

    ! Print to terminal
    if (par%master .and. C%do_time_display .and. C%geometry_relaxation_t_years > 0._dp) then

      if (C%geometry_relaxation_t_years <= 999._dp .and. &
          C%geometry_relaxation_t_years >=   1._dp) then
        write(*,"(A,F6.2,A)") '   Stepping out of time to relax geometry for ', C%geometry_relaxation_t_years, ' pseudo years...'
      else
        write(*,"(A)") '   Stepping out of time to relax geometry...'
      end if

      ! Display pseudo time
      write( r_time,"(F7.3)") t_pseudo
      write( r_step,"(F5.3)") C%geometry_relaxation_t_years / 100._dp
      write( *,"(A)", ADVANCE = trim( 'no')) c_carriage_return // &
                                             "     t_pseudo = " // trim( r_time) // &
                                             " yr - dt = " // trim( r_step) // " yr"
    end if

    do while (t_pseudo < C%geometry_relaxation_t_years)

      ! Save user-defined viscosity-iteration tolerance to recover it later
      visc_it_norm_dUV_tol_config = C%visc_it_norm_dUV_tol

      ! Set viscosity-iteration to a high enough value, which will allow the
      ! stress balance solver to move forward when given rough initial conditions
      ! NOTE: this is relevant mostly during the first time steps only, since
      ! after a short while the rough surface gradients relax enough to allow
      ! for smaller and smaller residuals during the viscosity iterations
      C%visc_it_norm_dUV_tol = 1._dp

      ! default time step: set so the relaxation takes 100 iterations
      t_step = C%geometry_relaxation_t_years / 100._dp
      ! But prevent time step larger than maximum allowed
      t_step = MIN( t_step, C%dt_ice_max)

      ! Ignore any mass balance terms
      SMB_dummy = 0._dp
      BMB_dummy = 0._dp
      LMB_dummy = 0._dp
      AMB_dummy = 0._dp

      ! Ignore any target thinning rates
      dHi_dt_target_dummy = 0._dp

      region%ice%effective_pressure = MAX( 0._dp, ice_density * grav * region%ice%Hi_eff) * region%ice%fraction_gr

      ! Calculate ice velocities for the predicted geometry
      call solve_stress_balance( region%mesh, region%ice, BMB_dummy, region%name, &
        n_visc_its, n_Axb_its)

      ! Calculate thinning rates for current geometry and velocity
      call calc_dHi_dt( region%mesh, region%ice%Hi, region%ice%Hb, region%ice%SL, region%ice%u_vav_b, region%ice%v_vav_b, SMB_dummy, BMB_dummy, LMB_dummy, AMB_dummy, region%ice%fraction_margin, &
                        region%ice%mask_noice, t_step, dHi_dt_new, Hi_new, region%ice%divQ, dHi_dt_target_dummy)

      ! Set ice model ice thickness to relaxed field
      do vi = region%mesh%vi1, region%mesh%vi2
        ! Apply relaxation over ice shelves and grounding lines
        if (region%ice%mask_floating_ice( vi) .OR. region%ice%mask_gl_gr( vi)) then
          region%ice%Hi( vi) = Hi_new( vi)
          region%ice%dHi_dt( vi) = dHi_dt_new( vi)
        ! Also over steep-sloped interior ice sheet points
        elseif (region%ice%mask_grounded_ice( vi) .and. region%ice%Hs_slope( vi) >= 0.03_dp) then
          region%ice%Hi( vi) = Hi_new( vi)
          region%ice%dHi_dt( vi) = dHi_dt_new( vi)
        end if
      end do

      ! Apply some specific corrections
      do vi = region%mesh%vi1, region%mesh%vi2
        ! don't let grounded ice cross the floatation threshold
        ! if (region%ice%mask_grounded_ice( vi)) then
        !   region%ice%Hi( vi) = MAX( region%ice%Hi( vi), (region%ice%SL( vi) - region%ice%Hb( vi)) * seawater_density/ice_density + .1_dp)
        ! end if
        ! Remove very thin ice
        if (region%ice%Hi( vi) < C%Hi_min) then
          region%ice%Hi( vi) = 0._dp
          region%ice%dHi_dt( vi) = 0._dp
        end if
        ! Remove ice absent at PD
        if (region%refgeo_PD%Hi( vi) == 0._dp) then
          region%ice%Hi( vi) = 0._dp
          region%ice%dHi_dt( vi) = 0._dp
        end if
      end do

      ! Calculate all other ice geometry quantities
      ! ===========================================

      do vi = region%mesh%vi1, region%mesh%vi2

        ! Basic geometry
        region%ice%Hs ( vi) = ice_surface_elevation( region%ice%Hi( vi), region%ice%Hb( vi), region%ice%SL( vi))
        region%ice%Hib( vi) = region%ice%Hs( vi) - region%ice%Hi( vi)
        region%ice%TAF( vi) = thickness_above_floatation( region%ice%Hi( vi), region%ice%Hb( vi), region%ice%SL( vi))

        if (region%ice%TAF( vi) > 0._dp) then
          ! Grounded ice
          region%ice%dHs_dt ( vi) = region%ice%dHb_dt( vi) + region%ice%dHi_dt( vi)
          region%ice%dHib_dt( vi) = region%ice%dHb_dt( vi)
        else
          ! Floating ice
          region%ice%dHs_dt ( vi) = region%ice%dHi_dt( vi) * (1._dp - ice_density / seawater_density)
          region%ice%dHib_dt( vi) = region%ice%dHi_dt( vi) *          ice_density / seawater_density
        end if

      end do

      ! Update masks
      call determine_masks( region%mesh, region%ice)

      ! Calculate new effective thickness
      call calc_effective_thickness( region%mesh, region%ice, region%ice%Hi, region%ice%Hi_eff, region%ice%fraction_margin)

      ! NOTE: as calculating the zeta gradients is quite expensive, only do so when necessary,
      !       i.e. when solving the heat equation or the Blatter-Pattyn stress balance
      ! Calculate zeta gradients
      call calc_zeta_gradients( region%mesh, region%ice)

      ! Calculate sub-grid grounded-area fractions
      call calc_grounded_fractions( region%mesh, region%ice)

      ! Reference geometry
      ! ==================

      region%refgeo_PD%Hi  = region%ice%Hi
      region%refgeo_PD%Hs  = region%ice%Hs
      region%refgeo_PD%Hb  = region%ice%Hb

      ! Differences w.r.t. present-day
      region%ice%dHi  = 0._dp
      region%ice%dHb  = 0._dp
      region%ice%dHs  = 0._dp
      region%ice%dHib = 0._dp

      ! Re-initialise previous and next Hi states
      region%ice%Hi_prev = region%ice%Hi
      region%ice%Hi_next = region%ice%Hi

      ! Advance pesudo time
      ! ===================

      t_pseudo = t_pseudo + t_step

      ! Time display
      if (par%master .and. C%do_time_display) then
        ! Carriage return flag
        r_adv = "no"
        if (t_pseudo >= C%geometry_relaxation_t_years) r_adv = "yes"
        ! Current pseudo time
        write( r_time,"(F7.3)") MIN( t_pseudo,C%geometry_relaxation_t_years)
        ! Current time step
        write( r_step,"(F5.3)") t_step
        ! Time display message
        write( *,"(A)", ADVANCE = trim( r_adv)) c_carriage_return // &
                                                "     t_pseudo = " // trim( r_time) // &
                                                " yr - dt = " // trim( r_step) // " yr"
      end if

      ! Retrieve user-defined viscosity-iteration tolerance
      C%visc_it_norm_dUV_tol = visc_it_norm_dUV_tol_config

    end do

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine apply_geometry_relaxation

! ===== Predictor-corrector scheme =====
! ======================================

  subroutine run_ice_dynamics_model_pc( region, dt_max)
    !< Calculate a new next modelled ice thickness

    ! In/output variables:
    type(type_model_region), intent(inout) :: region
    real(dp),                intent(in   ) :: dt_max

    ! Local variables:
    character(len=1024), parameter                       :: routine_name = 'run_ice_dynamics_model_pc'
    real(dp)                                             :: dt_crit_adv
    integer                                              :: pc_it
    real(dp), dimension(region%mesh%vi1:region%mesh%vi2) :: Hi_dummy, dHi_dt_dummy, LMB_dummy, AMB_dummy
    integer                                              :: vi, n_guilty, n_tot
    integer                                              :: n_visc_its
    integer                                              :: n_Axb_its
    integer                                              :: ierr

    ! Add routine to path
    call init_routine( routine_name)

    ! Store previous ice model state
    region%ice%t_Hi_prev  = region%ice%t_Hi_next
    region%ice%Hi_prev    = region%ice%Hi_next

    ! == Calculate time step ==
    ! =========================

    ! Store previous time step
    region%ice%pc%dt_n = region%ice%pc%dt_np1

    ! Calculate new time step (Robinson et al., 2020, Eq. 33)
    region%ice%pc%dt_np1 = (C%pc_epsilon / region%ice%pc%eta_np1)**(C%pc_k_I + C%pc_k_p) * &
      (C%pc_epsilon / region%ice%pc%eta_n)**(-C%pc_k_p) * region%ice%pc%dt_n

    ! Limit time step to maximum allowed value
    region%ice%pc%dt_np1 = MIN( region%ice%pc%dt_np1, dt_max)

    ! Limit time step to 1.2 times the previous time step
    region%ice%pc%dt_np1 = MIN( region%ice%pc%dt_np1, C%pc_max_time_step_increase * region%ice%pc%dt_n)

    ! Limit time step to minimum allowed value
    region%ice%pc%dt_np1 = MAX( region%ice%pc%dt_np1, C%dt_ice_min)

    ! Limit time step to critical advective time step
    call calc_critical_timestep_adv( region%mesh, region%ice, dt_crit_adv)

    region%ice%pc%dt_np1 = MIN( region%ice%pc%dt_np1, dt_crit_adv)

    ! == Time step iteration: if, at the end of the PC timestep, the truncation error
    !    turns out to be too large, run it again with a smaller dt, until the truncation
    !    decreases to below the specified tolerance
    ! ==================================================================================

    ! Store thinning rates from previous time step
    region%ice%pc%dHi_dt_Hi_nm1_u_nm1 = region%ice%dHi_dt

    ! Store the previous maximum truncation error eta_n
    region%ice%pc%eta_n = region%ice%pc%eta_np1

    pc_it = 0
    iterate_pc_timestep: do while (pc_it < C%pc_nit_max)

      pc_it = pc_it + 1

      ! Calculate time step ratio
      region%ice%pc%zeta_t = region%ice%pc%dt_np1 / region%ice%pc%dt_n

      ! == Predictor step ==
      ! ====================

      ! Invert a basal mass balance field that keeps the ice shelves in equilibrium
      call BMB_inversion( region, region%ice%pc%dt_np1)

      ! Invert a lateral mass balance field that keeps the calving fronts in equilibrium
      call LMB_inversion( region, region%ice%pc%dt_np1)

      ! Invert a surface mass balance field that keeps the ice sheet in check
      call SMB_inversion( region, region%ice%pc%dt_np1)

      ! Calculate thinning rates for current geometry and velocity
      call calc_dHi_dt( region%mesh, region%ice%Hi, region%ice%Hb, region%ice%SL, region%ice%u_vav_b, region%ice%v_vav_b, region%SMB%SMB, region%BMB%BMB, region%LMB%LMB, region%AMB%AMB, region%ice%fraction_margin, &
                        region%ice%mask_noice, region%ice%pc%dt_np1, region%ice%pc%dHi_dt_Hi_n_u_n, Hi_dummy, region%ice%divQ, region%ice%dHi_dt_target)

      ! ! if so desired, alter the computed dH/dt by adjusting dummy mass balance fluxes to get an equilibrium state
      ! call MB_inversion( region%mesh, region%ice, region%refgeo_PD, region%SMB, region%BMB, region%LMB, region%AMB, region%ice%pc%dHi_dt_Hi_n_u_n, Hi_dummy, region%ice%pc%dt_np1, region%time, region%name)

      ! Calculate predicted ice thickness (Robinson et al., 2020, Eq. 30)
      region%ice%pc%Hi_star_np1 = region%ice%Hi_prev + region%ice%pc%dt_np1 * ((1._dp + region%ice%pc%zeta_t / 2._dp) * &
        region%ice%pc%dHi_dt_Hi_n_u_n - (region%ice%pc%zeta_t / 2._dp) * region%ice%pc%dHi_dt_Hi_nm1_u_nm1)

      ! if so desired, modify the predicted ice thickness field based on user-defined settings
      call alter_ice_thickness( region%mesh, region%ice, region%ice%Hi_prev, region%ice%pc%Hi_star_np1, region%refgeo_PD, region%time)

      ! Adjust the predicted dHi_dt to compensate for thickness modifications
      ! This is just Robinson et al., 2020, Eq 30 above rearranged to retrieve
      ! an updated dHi_dt_Hi_n_u_n from the modified Hi_star_np1. if no ice
      ! thickness modifications were applied, then there will be not change.
      region%ice%pc%dHi_dt_Hi_n_u_n = ((region%ice%pc%Hi_star_np1 - region%ice%Hi_prev) / region%ice%pc%dt_np1 + (region%ice%pc%zeta_t / 2._dp) * region%ice%pc%dHi_dt_Hi_nm1_u_nm1) / (1._dp + region%ice%pc%zeta_t / 2._dp)

      ! == Update step ==
      ! =================

      ! Set model geometry to predicted
      region%ice%Hi = region%ice%pc%Hi_star_np1

      ! Set thinning rates to predicted
      region%ice%dHi_dt = (region%ice%Hi - region%ice%Hi_prev) / region%ice%pc%dt_np1

      ! Set model geometry to predicted
      do vi = region%mesh%vi1, region%mesh%vi2
        ! Basic geometry
        region%ice%Hs ( vi) = ice_surface_elevation( region%ice%Hi( vi), region%ice%Hb( vi), region%ice%SL( vi))
        region%ice%Hib( vi) = region%ice%Hs(  vi) - region%ice%Hi( vi)
      end do

      ! Update masks
      call determine_masks( region%mesh, region%ice)

      ! DENK DROM : assess whether this is important for the velocitiy computation below
      ! ! Calculate zeta gradients
      ! call calc_zeta_gradients( region%mesh, region%ice)

      ! Update sub-grid grounded fractions
      call calc_grounded_fractions( region%mesh, region%ice)

      ! DENK DROM : assess whether this is important for the velocitiy computation below
      ! ! Calculate the basal mass balance
      ! call run_BMB_model( region%mesh, region%ice, region%ocean, region%refgeo_PD, region%SMB, region%BMB, region%name, region%time)

      ! Calculate ice velocities for the predicted geometry
      call solve_stress_balance( region%mesh, region%ice, region%BMB%BMB, region%name, &
        n_visc_its, n_Axb_its)

      ! Update stability info
      region%ice%dt_ice     = region%ice%pc%dt_np1
      region%ice%n_visc_its = n_visc_its
      region%ice%n_Axb_its  = n_Axb_its

      ! == Corrector step ==
      ! ====================

      ! Set model geometry back to original
      do vi = region%mesh%vi1, region%mesh%vi2
        region%ice%Hi(  vi) = region%ice%Hi_prev( vi)
        region%ice%Hs(  vi) = ice_surface_elevation( region%ice%Hi( vi), region%ice%Hb( vi), region%ice%SL( vi))
        region%ice%Hib( vi) = region%ice%Hs(  vi) - region%ice%Hi( vi)
      end do

      ! Update masks
      call determine_masks( region%mesh, region%ice)

      ! Update sub-grid grounded fractions
      call calc_grounded_fractions( region%mesh, region%ice)

      ! Update effective ice thickness
      call calc_effective_thickness( region%mesh, region%ice, region%ice%Hi, region%ice%Hi_eff, region%ice%fraction_margin)

      ! Invert a basal mass balance field that keeps the ice shelves in equilibrium
      call BMB_inversion( region, region%ice%pc%dt_np1)

      ! Invert a lateral mass balance field that keeps the calving fronts in equilibrium
      call LMB_inversion( region, region%ice%pc%dt_np1)

      ! Invert a surface mass balance field that keeps the ice sheet in check
      call SMB_inversion( region, region%ice%pc%dt_np1)

      ! Calculate thinning rates for the current ice thickness and predicted velocity
      call calc_dHi_dt( region%mesh, region%ice%Hi, region%ice%Hb, region%ice%SL, region%ice%u_vav_b, region%ice%v_vav_b, region%SMB%SMB, region%BMB%BMB, region%LMB%LMB, region%AMB%AMB, region%ice%fraction_margin, &
                        region%ice%mask_noice, region%ice%pc%dt_np1, region%ice%pc%dHi_dt_Hi_star_np1_u_np1, Hi_dummy, region%ice%divQ, region%ice%dHi_dt_target)

      ! ! if so desired, alter the computed dH/dt by adjusting dummy mass balance fluxes to get an equilibrium state
      ! call MB_inversion( region%mesh, region%ice, region%refgeo_PD, region%SMB, region%BMB, region%LMB, region%AMB, region%ice%pc%dHi_dt_Hi_star_np1_u_np1, Hi_dummy, region%ice%pc%dt_np1, region%time, region%name)

      ! Calculate corrected ice thickness (Robinson et al. (2020), Eq. 31)
      region%ice%pc%Hi_np1 = region%ice%Hi_prev + (region%ice%pc%dt_np1 / 2._dp) * (region%ice%pc%dHi_dt_Hi_n_u_n + region%ice%pc%dHi_dt_Hi_star_np1_u_np1)

      ! Save "raw" thinning rates, as applied after the corrector step
      region%ice%dHi_dt_raw = (region%ice%pc%Hi_np1 - region%ice%Hi_prev) / region%ice%pc%dt_np1

      ! if so desired, modify the corrected ice thickness field based on user-defined settings
      call alter_ice_thickness( region%mesh, region%ice, region%ice%Hi_prev, region%ice%pc%Hi_np1, region%refgeo_PD, region%time)

      ! Adjust the predicted dHi_dt to compensate for thickness modifications
      ! This is just Robinson et al., 2020, Eq 31 above rearranged to retrieve
      ! an updated dHi_dt_Hi_star_np1_u_np1 from the modified Hi_np1. if no ice
      ! thickness modifications were applied, then there will be not change.
      region%ice%pc%dHi_dt_Hi_star_np1_u_np1 = (region%ice%pc%Hi_np1 - region%ice%Hi_prev) / (region%ice%pc%dt_np1 / 2._dp) - region%ice%pc%dHi_dt_Hi_n_u_n

      ! Add difference between raw and applied dHi_dt to residual tracker
      region%ice%dHi_dt_residual = region%ice%dHi_dt_raw - &  ! Raw change
                                   (region%ice%pc%Hi_np1 - region%ice%Hi_prev) / region%ice%pc%dt_np1  ! Minus applied change

      ! == Truncation error ==
      ! ======================

      ! Estimate truncation error
      call calc_pc_truncation_error( region%mesh, region%ice, region%ice%pc)

      ! == Error assessment ==
      ! ======================

      ! Initialise unstable vertex count
      n_tot = 0
      n_guilty = 0

      ! Determine number of unstable vertices
      do vi = region%mesh%vi1, region%mesh%vi2
        ! Only consider fully grounded vertices
        if (region%ice%fraction_gr( vi) < 1._dp) CYCLE
        ! if so, add to total vertex count
        n_tot = n_tot + 1
        ! if this vertex's error is larger than tolerance
        if (region%ice%pc%tau_np1( vi) > C%pc_epsilon) then
          ! Add to total guilty vertex count
          n_guilty = n_guilty + 1
          ! Add to this vertex's guilty record
          region%ice%pc%tau_n_guilty( vi) = region%ice%pc%tau_n_guilty( vi) + 1
        end if
      end do

      ! Add up findings from each process domain
      call MPI_ALLREDUCE( MPI_IN_PLACE, n_tot,    1, MPI_integer, MPI_SUM, MPI_COMM_WORLD, ierr)
      call MPI_ALLREDUCE( MPI_IN_PLACE, n_guilty, 1, MPI_integer, MPI_SUM, MPI_COMM_WORLD, ierr)

      ! Safety
      if (n_tot == 0) n_tot = 1

      ! Check if largest truncation error is small enough; if so, move on
      if (region%ice%pc%eta_np1 < C%pc_epsilon) then
        exit iterate_pc_timestep

      ! if not, check whether that occurs in a significant amount of vertices; if not,
      ! set the truncation error to almost the tolerance (to allow for growth) and move on
      elseif (100._dp * real( n_guilty,dp) / real(n_tot,dp) < C%pc_guilty_max) then
        ! if (par%master) call warning('{dp_01}% of vertices are changing rapidly, ignoring for now', dp_01 = 100._dp * real( n_guilty,dp) / real(n_tot,dp))
        region%ice%pc%eta_np1 = .95_dp * C%pc_epsilon
        exit iterate_pc_timestep

      ! if not, re-do the PC timestep
      else
        !if (par%master) call warning('{dp_01}% of vertices ({int_01}) are changing rapidly (eta = {dp_02}), reducing dt and redoing PC timestep', dp_01 = 100._dp * real( n_guilty,dp) / real(n_tot,dp), int_01 = n_guilty, dp_02 = region%ice%pc%eta_np1)
        region%ice%pc%dt_np1 = region%ice%pc%dt_np1 * 0.8_dp
        ! if the timestep has reached the specified lower limit, stop iterating
        if (region%ice%pc%dt_np1 <= C%dt_ice_min) then
          region%ice%pc%dt_np1 = C%dt_ice_min
          exit iterate_pc_timestep
        end if
      end if

    end do iterate_pc_timestep

    ! == Final quantities
    ! ===================

    ! Set next modelled ice thickness
    region%ice%t_Hi_next = region%ice%t_Hi_prev + region%ice%pc%dt_np1
    region%ice%Hi_next   = region%ice%pc%Hi_np1

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine run_ice_dynamics_model_pc

  subroutine calc_pc_truncation_error( mesh, ice, pc)
    !< Calculate the truncation error tau in the ice thickness
    !< rate of change (Robinson et al., 2020, Eq. 32)

    ! In- and output variables:
    type(type_mesh),      intent(in   ) :: mesh
    type(type_ice_model), intent(in   ) :: ice
    type(type_ice_pc),    intent(inout) :: pc

    ! Local variables:
    character(len=1024), parameter :: routine_name = 'calc_pc_truncation_error'
    integer                        :: vi, ierr

    ! Add routine to path
    call init_routine( routine_name)

    ! Calculate truncation error tau (Robinson et al., 2020, Eq. 32)
    do vi = mesh%vi1, mesh%vi2
      pc%tau_np1( vi) = pc%zeta_t * ABS( pc%Hi_np1( vi) - pc%Hi_star_np1( vi)) / ((3._dp * pc%zeta_t + 3._dp) * pc%dt_n)
    end do

    ! Calculate the maximum truncation error eta over grounded ice only
    pc%eta_np1 = C%pc_eta_min
    do vi = mesh%vi1, mesh%vi2
      if (ice%mask_grounded_ice( vi) .and. .not. ice%mask_gl_gr( vi) .and. ice%fraction_gr( vi) == 1._dp) then
        pc%eta_np1 = MAX( pc%eta_np1, pc%tau_np1( vi))
      end if
    end do
    call MPI_ALLREDUCE( MPI_IN_PLACE, pc%eta_np1, 1, MPI_doUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine calc_pc_truncation_error

  subroutine initialise_pc_scheme( mesh, pc, region_name)
    !< allocate memory and initialise values for the ice thickness predictor/corrector scheme.

    ! In- and output variables
    type(type_mesh),   intent(in   ) :: mesh
    type(type_ice_pc), intent(  out) :: pc
    character(len=3),  intent(in   ) :: region_name

    ! Local variables:
    character(len=1024), parameter :: routine_name = 'initialise_pc_scheme'
    character(len=256)             :: pc_choice_initialise
    character(len=256)             :: filename_pc_initialise
    real(dp)                       :: timeframe_pc_initialise

    ! Add routine to path
    call init_routine( routine_name)

    ! allocate memory
    ! ===============

    allocate( pc%dHi_dt_Hi_nm1_u_nm1(      mesh%vi1:mesh%vi2))   ! [m/yr] Thinning rates from previous time step
    allocate( pc%dHi_dt_Hi_n_u_n(          mesh%vi1:mesh%vi2))   ! [m/yr] Thinning rates for current time step with old geometry
    allocate( pc%Hi_star_np1(              mesh%vi1:mesh%vi2))   ! [m]    Predicted ice thickness
    allocate( pc%dHi_dt_Hi_star_np1_u_np1( mesh%vi1:mesh%vi2))   ! [m/yr] Thinning rates for predicted ice thickness and updated velocity
    allocate( pc%Hi_np1(                   mesh%vi1:mesh%vi2))   ! [m]    Corrected ice thickness
    allocate( pc%tau_np1(                  mesh%vi1:mesh%vi2))   ! [m]    Truncation error
    allocate( pc%tau_n_guilty(             mesh%vi1:mesh%vi2))   ! [-]    Number of PC iterations where vertex had truncation errors above the tolerance

    ! Initialise
    ! ==========

    select case (region_name)
    case default
      call crash('unknown region_name "' // trim( region_name) // '"!')
    case ('NAM')
      pc_choice_initialise    = C%pc_choice_initialise_NAM
      filename_pc_initialise  = C%filename_pc_initialise_NAM
      timeframe_pc_initialise = C%timeframe_pc_initialise_NAM
    case ('EAS')
      pc_choice_initialise    = C%pc_choice_initialise_EAS
      filename_pc_initialise  = C%filename_pc_initialise_EAS
      timeframe_pc_initialise = C%timeframe_pc_initialise_EAS
    case ('GRL')
      pc_choice_initialise    = C%pc_choice_initialise_GRL
      filename_pc_initialise  = C%filename_pc_initialise_GRL
      timeframe_pc_initialise = C%timeframe_pc_initialise_GRL
    case ('ANT')
      pc_choice_initialise    = C%pc_choice_initialise_ANT
      filename_pc_initialise  = C%filename_pc_initialise_ANT
      timeframe_pc_initialise = C%timeframe_pc_initialise_ANT
    end select

    select case (pc_choice_initialise)
    case default
      call crash('unknown pc_choice_initialise "' // trim( pc_choice_initialise) // '"!')
    case ('zero')
      ! Initialise everything from scratch

      pc%dt_n                     = C%dt_ice_min
      pc%dt_np1                   = C%dt_ice_min
      pc%zeta_t                   = 1._dp
      pc%dHi_dt_Hi_nm1_u_nm1      = 0._dp
      pc%dHi_dt_Hi_n_u_n          = 0._dp
      pc%Hi_star_np1              = 0._dp
      pc%dHi_dt_Hi_star_np1_u_np1 = 0._dp
      pc%Hi_np1                   = 0._dp
      pc%tau_np1                  = C%pc_epsilon
      pc%eta_n                    = C%pc_epsilon
      pc%eta_np1                  = C%pc_epsilon

    case ('read_from_file')
      ! Initialise from a (restart) file
      call initialise_pc_scheme_from_file( pc, filename_pc_initialise, timeframe_pc_initialise)
    end select

    ! Initialise the event counter for errors above tolerance
    pc%tau_n_guilty = 0

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine initialise_pc_scheme

  subroutine initialise_pc_scheme_from_file( pc, filename, timeframe)
    !< Initialise values for the ice thickness predictor/corrector scheme from a (restart) file.

    ! In- and output variables
    type(type_ice_pc),  intent(inout) :: pc
    character(len=256), intent(in   ) :: filename
    real(dp),           intent(in   ) :: timeframe

    ! Local variables:
    character(len=1024), parameter :: routine_name = 'initialise_pc_scheme_from_file'

    ! Add routine to path
    call init_routine( routine_name)

    ! write to terminal
    if (par%master) write(0,*) '   Initialising ice thickness predictor/corrector scheme from file "' // colour_string( trim( filename),'light blue') // '"...'

    ! Read values from the file
    if (timeframe == 1E9_dp) then
      ! Assume the file has no time dimension
      call read_field_from_file_0D(         filename, 'dt_n'                    , pc%dt_n                    )
      call read_field_from_file_0D(         filename, 'dt_np1'                  , pc%dt_np1                  )
      call read_field_from_file_0D(         filename, 'zeta_t'                  , pc%zeta_t                  )
      call read_field_from_mesh_file_dp_2D( filename, 'dHi_dt_Hi_nm1_u_nm1'     , pc%dHi_dt_Hi_nm1_u_nm1     )
      call read_field_from_mesh_file_dp_2D( filename, 'dHi_dt_Hi_n_u_n'         , pc%dHi_dt_Hi_n_u_n         )
      call read_field_from_mesh_file_dp_2D( filename, 'Hi_star_np1'             , pc%Hi_star_np1             )
      call read_field_from_mesh_file_dp_2D( filename, 'dHi_dt_Hi_star_np1_u_np1', pc%dHi_dt_Hi_star_np1_u_np1)
      call read_field_from_mesh_file_dp_2D( filename, 'Hi_np1'                  , pc%Hi_np1                  )
      call read_field_from_mesh_file_dp_2D( filename, 'tau_np1'                 , pc%tau_np1                 )
      call read_field_from_file_0D(         filename, 'eta_n'                   , pc%eta_n                   )
      call read_field_from_file_0D(         filename, 'eta_np1'                 , pc%eta_np1                 )
    else
      ! Read specified timeframe
      call read_field_from_file_0D(         filename, 'dt_n'                    , pc%dt_n                    , time_to_read = timeframe)
      call read_field_from_file_0D(         filename, 'dt_np1'                  , pc%dt_np1                  , time_to_read = timeframe)
      call read_field_from_file_0D(         filename, 'zeta_t'                  , pc%zeta_t                  , time_to_read = timeframe)
      call read_field_from_mesh_file_dp_2D( filename, 'dHi_dt_Hi_nm1_u_nm1'     , pc%dHi_dt_Hi_nm1_u_nm1     , time_to_read = timeframe)
      call read_field_from_mesh_file_dp_2D( filename, 'dHi_dt_Hi_n_u_n'         , pc%dHi_dt_Hi_n_u_n         , time_to_read = timeframe)
      call read_field_from_mesh_file_dp_2D( filename, 'Hi_star_np1'             , pc%Hi_star_np1             , time_to_read = timeframe)
      call read_field_from_mesh_file_dp_2D( filename, 'dHi_dt_Hi_star_np1_u_np1', pc%dHi_dt_Hi_star_np1_u_np1, time_to_read = timeframe)
      call read_field_from_mesh_file_dp_2D( filename, 'Hi_np1'                  , pc%Hi_np1                  , time_to_read = timeframe)
      call read_field_from_mesh_file_dp_2D( filename, 'tau_np1'                 , pc%tau_np1                 , time_to_read = timeframe)
      call read_field_from_file_0D(         filename, 'eta_n'                   , pc%eta_n                   , time_to_read = timeframe)
      call read_field_from_file_0D(         filename, 'eta_np1'                 , pc%eta_np1                 , time_to_read = timeframe)
    end if

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine initialise_pc_scheme_from_file

  subroutine write_to_restart_file_pc_scheme( mesh, pc, time)
    !< write to the restart NetCDF file for the ice thickness predictor/corrector scheme

    ! In/output variables:
    type(type_mesh),   intent(in   ) :: mesh
    type(type_ice_pc), intent(in   ) :: pc
    real(dp),          intent(in   ) :: time

    ! Local variables:
    character(len=1024), parameter :: routine_name = 'write_to_restart_file_pc_scheme'
    integer                        :: ncid

    ! Add routine to path
    call init_routine( routine_name)

    ! if no NetCDF output should be created, do nothing
    if (.not. C%do_create_netcdf_output) then
      call finalise_routine( routine_name)
      return
    end if

    ! Print to terminal
    if (par%master) write(0,'(A)') '   Writing to ice dynamics restart file "' // &
      colour_string( trim( pc%restart_filename), 'light blue') // '"...'

    ! Open the NetCDF file
    call open_existing_netcdf_file_for_writing( pc%restart_filename, ncid)

    ! write the time to the file
    call write_time_to_file( pc%restart_filename, ncid, time)

    ! write the data fields to the file
    call write_to_field_multopt_dp_0D(            pc%restart_filename, ncid, 'dt_n'                    , pc%dt_n                    )
    call write_to_field_multopt_dp_0D(            pc%restart_filename, ncid, 'dt_np1'                  , pc%dt_np1                  )
    call write_to_field_multopt_dp_0D(            pc%restart_filename, ncid, 'zeta_t'                  , pc%zeta_t                  )
    call write_to_field_multopt_mesh_dp_2D( mesh, pc%restart_filename, ncid, 'dHi_dt_Hi_nm1_u_nm1'     , pc%dHi_dt_Hi_nm1_u_nm1     )
    call write_to_field_multopt_mesh_dp_2D( mesh, pc%restart_filename, ncid, 'dHi_dt_Hi_n_u_n'         , pc%dHi_dt_Hi_n_u_n         )
    call write_to_field_multopt_mesh_dp_2D( mesh, pc%restart_filename, ncid, 'Hi_star_np1'             , pc%Hi_star_np1             )
    call write_to_field_multopt_mesh_dp_2D( mesh, pc%restart_filename, ncid, 'dHi_dt_Hi_star_np1_u_np1', pc%dHi_dt_Hi_star_np1_u_np1)
    call write_to_field_multopt_mesh_dp_2D( mesh, pc%restart_filename, ncid, 'Hi_np1'                  , pc%Hi_np1                  )
    call write_to_field_multopt_mesh_dp_2D( mesh, pc%restart_filename, ncid, 'tau_np1'                 , pc%tau_np1                 )
    call write_to_field_multopt_dp_0D(            pc%restart_filename, ncid, 'eta_n'                   , pc%eta_n                   )
    call write_to_field_multopt_dp_0D(            pc%restart_filename, ncid, 'eta_np1'                 , pc%eta_np1                 )

    ! Close the file
    call close_netcdf_file( ncid)

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine write_to_restart_file_pc_scheme

  subroutine create_restart_file_pc_scheme( mesh, pc)
    !< Create a restart NetCDF file for the ice thickness predictor/corrector scheme
    !< Includes generation of the procedural filename (e.g. "restart_pc_00001.nc")

    ! In/output variables:
    type(type_mesh),   intent(in   ) :: mesh
    type(type_ice_pc), intent(inout) :: pc

    ! Local variables:
    character(len=1024), parameter :: routine_name = 'create_restart_file_pc_scheme'
    character(len=256)             :: filename_base
    integer                        :: ncid

    ! Add routine to path
    call init_routine( routine_name)

    ! if no NetCDF output should be created, do nothing
    if (.not. C%do_create_netcdf_output) then
      call finalise_routine( routine_name)
      return
    end if

    ! Set the filename
    filename_base = trim( C%output_dir) // 'restart_pc_scheme'
    call generate_filename_XXXXXdotnc( filename_base, pc%restart_filename)

    ! Print to terminal
    if (par%master) write(0,'(A)') '   Creating ice dynamics restart file "' // &
      colour_string( trim( pc%restart_filename), 'light blue') // '"...'

    ! Create the NetCDF file
    call create_new_netcdf_file_for_writing( pc%restart_filename, ncid)

    ! Set up the mesh in the file
    call setup_mesh_in_netcdf_file( pc%restart_filename, ncid, mesh)

    ! Add a time dimension to the file
    call add_time_dimension_to_file( pc%restart_filename, ncid)

    ! Add the data fields to the file
    call add_field_dp_0D(      pc%restart_filename, ncid, 'dt_n'                    , long_name = 'Previous time step', units = 'yr')
    call add_field_dp_0D(      pc%restart_filename, ncid, 'dt_np1'                  , long_name = 'Current time step' , units = 'yr')
    call add_field_dp_0D(      pc%restart_filename, ncid, 'zeta_t'                  , long_name = 'Ratio between previous and new time step')
    call add_field_mesh_dp_2D( pc%restart_filename, ncid, 'dHi_dt_Hi_nm1_u_nm1'     , long_name = 'Thinning rates from previous time step', units = 'm/yr')
    call add_field_mesh_dp_2D( pc%restart_filename, ncid, 'dHi_dt_Hi_n_u_n'         , long_name = 'Thinning rates for current time step with old geometry', units = 'm/yr')
    call add_field_mesh_dp_2D( pc%restart_filename, ncid, 'Hi_star_np1'             , long_name = 'Predicted ice thickness', units = 'm')
    call add_field_mesh_dp_2D( pc%restart_filename, ncid, 'dHi_dt_Hi_star_np1_u_np1', long_name = 'Thinning rates for predicted ice thickness and updated velocity', units = 'm/yr')
    call add_field_mesh_dp_2D( pc%restart_filename, ncid, 'Hi_np1'                  , long_name = 'Corrected ice thickness', units = 'm')
    call add_field_mesh_dp_2D( pc%restart_filename, ncid, 'tau_np1'                 , long_name = 'Truncation error', units = 'm')
    call add_field_dp_0D(      pc%restart_filename, ncid, 'eta_n'                   , long_name = 'Previous maximum truncation error', units = 'm')
    call add_field_dp_0D(      pc%restart_filename, ncid, 'eta_np1'                 , long_name = 'Current maximum truncation error', units = 'm')

    ! Close the file
    call close_netcdf_file( ncid)

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine create_restart_file_pc_scheme

  subroutine remap_pc_scheme( mesh_old, mesh_new, pc)
    !< reallocate memory for the ice thickness predictor/corrector scheme.

    ! In- and output variables
    type(type_mesh),   intent(in   ) :: mesh_old
    type(type_mesh),   intent(in   ) :: mesh_new
    type(type_ice_pc), intent(inout) :: pc

    ! Local variables:
    character(len=1024), parameter :: routine_name = 'remap_pc_scheme'

    ! Add routine to path
    call init_routine( routine_name)

    ! reallocate memory
    call reallocate_bounds( pc%dHi_dt_Hi_nm1_u_nm1     , mesh_new%vi1, mesh_new%vi2)   ! [m/yr] Thinning rates from previous time step
    call reallocate_bounds( pc%dHi_dt_Hi_n_u_n         , mesh_new%vi1, mesh_new%vi2)   ! [m/yr] Thinning rates for current time step with old geometry
    call reallocate_bounds( pc%Hi_star_np1             , mesh_new%vi1, mesh_new%vi2)   ! [m]    Predicted ice thickness
    call reallocate_bounds( pc%dHi_dt_Hi_star_np1_u_np1, mesh_new%vi1, mesh_new%vi2)   ! [m/yr] Thinning rates for predicted ice thickness and updated velocity
    call reallocate_bounds( pc%Hi_np1                  , mesh_new%vi1, mesh_new%vi2)   ! [m]    Corrected ice thickness
    call reallocate_bounds( pc%tau_np1                 , mesh_new%vi1, mesh_new%vi2)   ! [m]    Truncation error
    call reallocate_bounds( pc%tau_n_guilty            , mesh_new%vi1, mesh_new%vi2)   ! [-]    Number of events above error tolerance

    ! Reinitialise everything from scratch
    pc%dt_n                     = C%dt_ice_min
    pc%dt_np1                   = C%dt_ice_min
    pc%zeta_t                   = 1._dp
    pc%dHi_dt_Hi_nm1_u_nm1      = 0._dp
    pc%dHi_dt_Hi_n_u_n          = 0._dp
    pc%Hi_star_np1              = 0._dp
    pc%dHi_dt_Hi_star_np1_u_np1 = 0._dp
    pc%Hi_np1                   = 0._dp
    pc%tau_np1                  = C%pc_epsilon
    pc%tau_n_guilty             = 0
    pc%eta_n                    = C%pc_epsilon
    pc%eta_np1                  = C%pc_epsilon

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine remap_pc_scheme

! ===== Direct scheme =====
! =========================

  subroutine run_ice_dynamics_model_direct( region, dt_max)
    !< Calculate a new next modelled ice thickness

    ! In/output variables:
    type(type_model_region), intent(inout) :: region
    real(dp),                intent(in   ) :: dt_max

    ! Local variables:
    character(len=1024), parameter :: routine_name = 'run_ice_dynamics_model_direct'
    real(dp)                       :: dt_crit_SIA, dt_crit_adv, dt
    integer                        :: vi
    integer                        :: n_visc_its
    integer                        :: n_Axb_its

    ! Add routine to path
    call init_routine( routine_name)

    ! Safety
    if (.not. (C%choice_stress_balance_approximation == 'SIA' .OR. &
               C%choice_stress_balance_approximation == 'SSA' .OR. &
               C%choice_stress_balance_approximation == 'SIA/SSA')) then
      call crash('direct timestepping only works for SIA, SSA, or SIA/SSA ice dynamics!')
    end if

    ! Store previous ice model state
    region%ice%t_Hi_prev  = region%ice%t_Hi_next
    region%ice%Hi_prev    = region%ice%Hi_next

    ! Calculate ice velocities
    call solve_stress_balance( region%mesh, region%ice, region%BMB%BMB, region%name, &
      n_visc_its, n_Axb_its)

    ! Calculate time step

    ! Start with the maximum allowed time step
    dt = dt_max

    ! Limit to the SIA critical time step
    if (C%choice_stress_balance_approximation == 'SIA' .OR. &
        C%choice_stress_balance_approximation == 'SIA/SSA') then
      call calc_critical_timestep_SIA( region%mesh, region%ice, dt_crit_SIA)
      dt = MIN( dt, dt_crit_SIA)
    end if

    ! Limit to the advective critical time step
    if (C%choice_stress_balance_approximation == 'SSA' .OR. &
        C%choice_stress_balance_approximation == 'SIA/SSA') then
      call calc_critical_timestep_adv( region%mesh, region%ice, dt_crit_adv)
      dt = MIN( dt, dt_crit_adv)
    end if

    ! Limit to the smallest allowed time step
    dt = MAX( C%dt_ice_min, dt)

    ! Calculate thinning rates and predicted geometry
    call calc_dHi_dt( region%mesh, region%ice%Hi, region%ice%Hb, region%ice%SL, region%ice%u_vav_b, region%ice%v_vav_b, region%SMB%SMB, region%BMB%BMB, region%LMB%LMB, region%AMB%AMB, region%ice%fraction_margin, &
                      region%ice%mask_noice, dt, region%ice%dHi_dt, region%ice%Hi_next, region%ice%divQ, region%ice%dHi_dt_target)

    ! if so desired, invert/adjust mass balance fluxes to get an equilibrium state
    call MB_inversion( region%mesh, region%ice, region%refgeo_PD, region%SMB, region%BMB, region%LMB, region%AMB, region%ice%dHi_dt, region%ice%Hi_next, dt, region%time, region%name)

    ! Save the "raw" dynamical dH/dt before any alterations
    region%ice%dHi_dt_raw = region%ice%dHi_dt

    ! Modify predicted ice thickness if desired
    call alter_ice_thickness( region%mesh, region%ice, region%ice%Hi_prev, region%ice%Hi_next, region%refgeo_PD, region%time)

    ! Compute residual between the "raw" and final thinning rates
    region%ice%dHi_dt_residual = region%ice%dHi_dt_raw - (region%ice%Hi_next - region%ice%Hi_prev) / dt

    ! Set next modelled ice thickness timestamp
    region%ice%t_Hi_next = region%ice%t_Hi_prev + dt

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine run_ice_dynamics_model_direct

  subroutine calc_critical_timestep_SIA( mesh, ice, dt_crit_SIA)
    !< Calculate the critical time step for advective ice flow (CFL criterion)

    ! In- and output variables:
    type(type_mesh),                     intent(in   ) :: mesh
    type(type_ice_model),                intent(in   ) :: ice
    real(dp),                            intent(  out) :: dt_crit_SIA

    ! Local variables:
    character(len=1024), parameter :: routine_name = 'calc_critical_timestep_SIA'
    real(dp), dimension(mesh%nV)   :: Hi_tot
    integer                        :: ti, via, vib, vic
    real(dp)                       :: d_ab, d_bc, d_ca, d_min, Hi, D_SIA, dt
    real(dp), parameter            :: dt_correction_factor = 0.9_dp ! Make actual applied time step a little bit smaller, just to be sure.
    integer                        :: ierr

    ! Add routine to path
    call init_routine( routine_name)

    ! Gather global ice thickness
    call gather_to_all( ice%Hi, Hi_tot)

    ! Initialise time step with maximum allowed value
    dt_crit_SIA = C%dt_ice_max

    do ti = mesh%ti1, mesh%ti2

      ! Calculate shortest triangle side
      via = mesh%Tri( ti,1)
      vib = mesh%Tri( ti,2)
      vic = mesh%Tri( ti,3)

      d_ab = norm2( mesh%V( vib,:) - mesh%V( via,:))
      d_bc = norm2( mesh%V( vic,:) - mesh%V( vib,:))
      d_ca = norm2( mesh%V( via,:) - mesh%V( vic,:))

      d_min = minval([ d_ab, d_bc, d_ca])

      ! Find maximum diffusivity in the vertical column
      D_SIA = max( 1E2_dp, maxval( abs( ice%SIA%D_3D_b( ti,:))))

      ! Calculate critical timestep
      Hi = maxval( [0.1_dp, Hi_tot( via), Hi_tot( vib), Hi_tot( vic)])
      dt = d_min**2 / (6._dp * Hi * D_SIA) * dt_correction_factor
      dt_crit_SIA = min( dt_crit_SIA, dt)

    end do

    call MPI_ALLREDUCE( MPI_IN_PLACE, dt_crit_SIA, 1, MPI_doUBLE_PRECISION, MPI_MIN, MPI_COMM_WORLD, ierr)
    dt_crit_SIA = min( C%dt_ice_max, dt_crit_SIA)

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine calc_critical_timestep_SIA

  subroutine calc_critical_timestep_adv( mesh, ice, dt_crit_adv)
    !< Calculate the critical time step for advective ice flow (CFL criterion)

    ! In- and output variables:
    type(type_mesh),      intent(in   ) :: mesh
    type(type_ice_model), intent(in   ) :: ice
    real(dp),             intent(  out) :: dt_crit_adv

    ! Local variables:
    character(len=1024), parameter         :: routine_name = 'calc_critical_timestep_adv'
    real(dp), dimension(mesh%nV)           :: Hi_tot
    logical,  dimension(mesh%nV)           :: mask_floating_ice_tot
    real(dp), dimension(mesh%ei1:mesh%ei2) :: u_vav_c, v_vav_c
    real(dp), dimension(mesh%nE)           :: u_vav_c_tot, v_vav_c_tot
    integer                                :: ei, vi, vj
    real(dp)                               :: dist, dt
    real(dp), parameter                    :: dt_correction_factor = 0.9_dp ! Make actual applied time step a little bit smaller, just to be sure.
    integer                                :: ierr

    ! Add routine to path
    call init_routine( routine_name)

    ! Gather global ice thickness
    call gather_to_all( ice%Hi, Hi_tot)

    call gather_to_all( ice%mask_floating_ice, mask_floating_ice_tot)

    ! Calculate vertically averaged ice velocities on the edges
    call map_velocities_from_b_to_c_2D( mesh, ice%u_vav_b, ice%v_vav_b, u_vav_c, v_vav_c)
    call gather_to_all( u_vav_c, u_vav_c_tot)
    call gather_to_all( v_vav_c, v_vav_c_tot)

    ! Initialise time step with maximum allowed value
    dt_crit_adv = C%dt_ice_max

    do ei = mesh%ei1, mesh%ei2

      ! Only check at ice-covered vertices
      vi = mesh%EV( ei,1)
      vj = mesh%EV( ei,2)
      if (Hi_tot( vi) == 0._dp .OR. Hi_tot( vj) == 0._dp) CYCLE

      if (C%do_grounded_only_adv_dt) then
        ! Only check grounded vertices
        if (mask_floating_ice_tot( vi) .OR. mask_floating_ice_tot( vj)) CYCLE
      end if

      dist = norm2( mesh%V( vi,:) - mesh%V( vj,:))
      dt = dist / max( 0.1_dp, abs( u_vav_c_tot( ei)) + abs( v_vav_c_tot( ei))) * dt_correction_factor
      dt_crit_adv = min( dt_crit_adv, dt)

    end do

    call MPI_ALLREDUCE( MPI_IN_PLACE, dt_crit_adv, 1, MPI_doUBLE_PRECISION, MPI_MIN, MPI_COMM_WORLD, ierr)
    dt_crit_adv = min( C%dt_ice_max, dt_crit_adv)

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine calc_critical_timestep_adv

! ===== Inversions =====

  subroutine SMB_inversion( region, dt)
    !< Invert the surface mass balance that would keep the ice sheet in check

    ! In/output variables:
    type(type_model_region), intent(inout) :: region
    real(dp),                intent(in   ) :: dt

    ! Local variables:
    character(len=1024), parameter                       :: routine_name = 'SMB_inversion'
    integer                                              :: vi
    integer,  dimension(region%mesh%vi1:region%mesh%vi2) :: extrapolation_mask
    real(dp)                                             :: dt_dummy
    real(dp), dimension(region%mesh%vi1:region%mesh%vi2) :: SMB_dummy, BMB_dummy, LMB_dummy, AMB_dummy, dHi_dt_dummy, Hi_dummy

    ! Add routine to path
    call init_routine( routine_name)

    ! Check if this inversion is desired
    if (.not. C%do_SMB_removal_icefree_land) then
      ! Finalise routine path
      call finalise_routine( routine_name)
      ! And exit subroutine
      return
    end if

    ! Initialise
    extrapolation_mask = 0

    ! == Equilibrium SMB
    ! ==================

    ! Set dummy mass balance terms to 0
    SMB_dummy    = 0._dp
    BMB_dummy    = 0._dp
    LMB_dummy    = 0._dp
    AMB_dummy    = 0._dp

    ! Copy model time step
    dt_dummy = dt

    ! Use full mass balance to invert SMB values
    call calc_dHi_dt( region%mesh, region%ice%Hi, region%ice%Hb, region%ice%SL, region%ice%u_vav_b, region%ice%v_vav_b, region%SMB%SMB, region%BMB%BMB, region%LMB%LMB, AMB_dummy, region%ice%fraction_margin, &
                      region%ice%mask_noice, dt_dummy, dHi_dt_dummy, Hi_dummy, region%ice%divQ, region%ice%dHi_dt_target)

    ! Compute equilibrium LMB
    do vi = region%mesh%vi1, region%mesh%vi2

      ! Skip vertices where land should not necessarily be ice-free
      if (.not. region%ice%mask_icefree_land( vi) .OR. .not. region%refgeo_PD%Hi( vi) == 0._dp) CYCLE

      ! Equilibrium SMB field
      region%SMB%SMB( vi) = region%SMB%SMB(vi) - dHi_dt_dummy( vi)

    end do

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine SMB_inversion

  subroutine BMB_inversion( region, dt)
    !< Invert the basal mass balance that would keep the ice shelves in equilibrium

    ! In/output variables:
    type(type_model_region), intent(inout) :: region
    real(dp),                intent(in   ) :: dt

    ! Local variables:
    character(len=1024), parameter                       :: routine_name = 'BMB_inversion'
    integer                                              :: vi
    integer,  dimension(region%mesh%vi1:region%mesh%vi2) :: extrapolation_mask
    real(dp)                                             :: dt_dummy
    real(dp), dimension(region%mesh%vi1:region%mesh%vi2) :: SMB_dummy, BMB_dummy, LMB_dummy, AMB_dummy, dHi_dt_dummy, Hi_dummy

    ! Add routine to path
    call init_routine( routine_name)

    ! Check if this inversion is desired
    if (C%do_BMB_inversion .and. &
        region%time >= C%BMB_inversion_t_start .and. &
        region%time <= C%BMB_inversion_t_end) then
      ! Go for it
    else
      ! Finalise routine path
      call finalise_routine( routine_name)
      ! And exit subroutine
      return
    end if

    ! Initialise
    extrapolation_mask = 0

    ! == Equilibrium LMB
    ! ==================

    ! Set dummy mass balance terms to 0
    SMB_dummy    = 0._dp
    BMB_dummy    = 0._dp
    LMB_dummy    = 0._dp
    AMB_dummy    = 0._dp

    ! Copy model time step
    dt_dummy = dt

    ! Use no basal or lateral mass balance to invert BMB values for an ice shelf in equilibrium
    call calc_dHi_dt( region%mesh, region%ice%Hi, region%ice%Hb, region%ice%SL, region%ice%u_vav_b, region%ice%v_vav_b, region%SMB%SMB, BMB_dummy, LMB_dummy, AMB_dummy, region%ice%fraction_margin, &
                      region%ice%mask_noice, dt_dummy, dHi_dt_dummy, Hi_dummy, region%ice%divQ, region%ice%dHi_dt_target)

    ! Initialise
    region%BMB%BMB_inv = 0._dp

    ! Compute equilibrium LMB
    do vi = region%mesh%vi1, region%mesh%vi2

      ! Skip vertices where BMB does not operate
      if (.not. region%ice%mask_gl_gr( vi) .and. &
          .not. region%ice%mask_floating_ice( vi) .and. &
          .not. region%ice%mask_cf_fl( vi)) cycle

      ! Equilibrium BMB field
      region%BMB%BMB_inv( vi) = -dHi_dt_dummy( vi)

      ! Add to extrapolation seeds
      extrapolation_mask( vi) = 2

    end do

    ! == Calving fronts
    ! =================

    do vi = region%mesh%vi1, region%mesh%vi2

      ! Detect shelf fronts where upstream BMB can be extrapolated into
      if (region%ice%mask_cf_fl( vi) .and. .not. region%ice%mask_gl_fl( vi)) then
        extrapolation_mask( vi) = 1
      end if

    end do

    ! Perform the extrapolation - mask: 2 -> use as seed; 1 -> extrapolate; 0 -> ignore
    call extrapolate_Gaussian( region%mesh, extrapolation_mask, region%BMB%BMB_inv, 10000._dp)

    ! == Total BMB
    ! ============

    ! Initialise
    region%BMB%BMB = 0._dp

    ! Compute total BMB
    do vi = region%mesh%vi1, region%mesh%vi2

      ! Skip vertices where BMB does not operate
      if (.not. region%ice%mask_gl_gr( vi) .and. &
          .not. region%ice%mask_floating_ice( vi) .and. &
          .not. region%ice%mask_cf_fl( vi)) cycle

      ! Final BMB field
      region%BMB%BMB( vi) = region%BMB%BMB_inv( vi)

    end do

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine BMB_inversion

  subroutine LMB_inversion( region, dt)
    !< Invert the lateral mass balance that would keep the calving front in equilibrium

    ! In/output variables:
    type(type_model_region), intent(inout) :: region
    real(dp),                intent(in   ) :: dt

    ! Local variables:
    character(len=1024), parameter                       :: routine_name = 'LMB_inversion'
    integer                                              :: vi, vj, ci, ei
    real(dp), dimension(region%mesh%ei1:region%mesh%ei2) :: u_vav_c, v_vav_c
    real(dp), dimension(region%mesh%nE)                  :: u_vav_c_tot, v_vav_c_tot
    real(dp)                                             :: dt_dummy, calving_rate, calving_ratio, calving_perp, L_c, V_calved
    real(dp), dimension(region%mesh%vi1:region%mesh%vi2) :: SMB_dummy, BMB_dummy, LMB_dummy, AMB_dummy, dHi_dt_dummy, Hi_dummy, divQ_eff, LMB_trans
    real(dp), dimension(region%mesh%nV)                  :: Hi_tot, fraction_margin_tot
    logical                                              :: found_advancing_calving_front, found_calving_front_neighbour
    logical,  dimension(region%mesh%vi1:region%mesh%vi2) :: mask_advancing_calving_front
    logical,  dimension(region%mesh%nV)                  :: mask_advancing_calving_front_tot, mask_cf_fl_tot, mask_icefree_ocean_tot

    ! Add routine to path
    call init_routine( routine_name)

    ! Check if this inversion is desired
    if (C%do_LMB_inversion .and. &
        region%time >= C%LMB_inversion_t_start .and. &
        region%time <= C%LMB_inversion_t_end) then
      ! Go for it
    else
      ! Finalise routine path
      call finalise_routine( routine_name)
      ! And exit subroutine
      return
    end if

    ! == Equilibrium LMB
    ! ==================

    ! Set dummy mass balance terms to 0
    SMB_dummy    = 0._dp
    BMB_dummy    = 0._dp
    LMB_dummy    = 0._dp
    AMB_dummy    = 0._dp

    ! Copy model time step
    dt_dummy = dt

    ! Use no lateral mass balance to invert its value for a calving front in equilibrium
    call calc_dHi_dt( region%mesh, region%ice%Hi, region%ice%Hb, region%ice%SL, region%ice%u_vav_b, region%ice%v_vav_b, region%SMB%SMB, region%BMB%BMB, LMB_dummy, AMB_dummy, region%ice%fraction_margin, &
                      region%ice%mask_noice, dt_dummy, dHi_dt_dummy, Hi_dummy, region%ice%divQ, region%ice%dHi_dt_target)

    ! Initialise
    region%LMB%LMB_inv = 0._dp

    ! Compute equilibrium LMB
    do vi = region%mesh%vi1, region%mesh%vi2

      ! Skip vertices where LMB does not operate
      if (.not. region%ice%mask_cf_fl( vi) .and. &
          .not. region%ice%mask_cf_gr( vi) .and. &
          .not. region%ice%mask_icefree_ocean( vi)) CYCLE

      ! Equilibrium LMB field: let positive values remain, since the final goal is
      ! that the *sum* of the equilibrium and transient LMB be equal to the target rate
      region%LMB%LMB_inv( vi) = -dHi_dt_dummy( vi)

    end do

    ! == Transient LMB
    ! ================

    ! Gather data from all processes
    call gather_to_all(      region%ice%Hi_eff, Hi_tot)
    call gather_to_all(      region%ice%fraction_margin, fraction_margin_tot)

    ! Gather masks from all processes
    call gather_to_all( region%ice%mask_cf_fl, mask_cf_fl_tot)
    call gather_to_all( region%ice%mask_icefree_ocean, mask_icefree_ocean_tot)

    ! Calculate vertically averaged ice velocities on the edges
    call map_velocities_from_b_to_c_2D( region%mesh, region%ice%u_vav_b, region%ice%v_vav_b, u_vav_c, v_vav_c)
    call gather_to_all( u_vav_c, u_vav_c_tot)
    call gather_to_all( v_vav_c, v_vav_c_tot)

    ! Initialise
    LMB_trans = 0._dp

    ! ! Translate imposed transient calving rates into LMB
    ! do vi = region%mesh%vi1, region%mesh%vi2

    !   V_calved = 0._dp

    !   ! Skip vertices where LMB does not operate
    !   if (.not. region%ice%mask_cf_fl( vi) .and. .not. region%ice%mask_icefree_ocean( vi)) CYCLE

    !   calving_rate = 0._dp

    !   if (C%choice_refgeo_PD_ANT == 'idealised' .and. region%time > 10000._dp) then
    !     if (C%choice_refgeo_init_idealised == 'calvmip_circular') then
    !       calving_rate = -300._dp * SIN(2._dp * pi * region%time / 1000._dp)
    !     elseif (C%choice_refgeo_init_idealised == 'calvmip_Thule') then
    !       calving_rate = -750._dp * SIN(2._dp * pi * region%time / 1000._dp)
    !     end if
    !   end if

    !   if (region%ice%uabs_vav( vi) > 0._dp) then
    !     calving_ratio = calving_rate / region%ice%uabs_vav( vi)
    !   end if

    !   ! Loop over all connections of vertex vi
    !   do ci = 1, region%mesh%nC( vi)

    !     ! Connection ci from vertex vi leads through edge ei to vertex vj
    !     ei = region%mesh%VE( vi,ci)
    !     vj = region%mesh%C(  vi,ci)

    !     ! The shared Voronoi cell boundary section between the
    !     ! Voronoi cells of vertices vi and vj has length L_c
    !     L_c = region%mesh%Cw( vi,ci)

    !     ! Calculate calving rate component perpendicular to this shared Voronoi cell boundary section
    !     calving_perp = calving_ratio * ABS( u_vav_c_tot( ei) * mesh%D_x( vi, ci)/mesh%D( vi, ci) + v_vav_c_tot( ei) * mesh%D_y( vi, ci)/mesh%D( vi, ci))

    !     ! Calving front vertices: check if neighbour is open ocean
    !     if (region%ice%mask_cf_fl( vi) .and. mask_icefree_ocean_tot( vj)) then

    !       ! Volume calved laterally: perpendicular calving rate times area of the ice front face [m^3/yr]
    !       V_calved = V_calved + L_c * calving_perp * Hi_tot( vi)

    !     ! Ice-free ocean vertices: check if neighbour is a fully advanced calving front
    !     elseif (region%ice%mask_icefree_ocean( vi) .and. mask_cf_fl_tot( vj) .and. fraction_margin_tot( vj) >= 1._dp) then

    !       ! Volume calved laterally: perpendicular calving rate times area of the ice front face [m^3/yr]
    !       V_calved = V_calved + L_c * calving_perp * Hi_tot( vj)

    !     end if

    !   end do

    !   ! Translate lateral volume loss into vertical thinning rate [m/yr]
    !   LMB_trans( vi) = V_calved / region%mesh%A( vi)

    ! end do

    ! == Total LMB
    ! ============

    ! Initialise
    region%LMB%LMB = 0._dp

    ! Compute total LMB
    do vi = region%mesh%vi1, region%mesh%vi2

      ! Skip vertices where LMB does not operate
      if (.not. region%ice%mask_cf_fl( vi) .and. &
          .not. region%ice%mask_cf_gr( vi) .and. &
          .not. region%ice%mask_icefree_ocean( vi)) cycle

      ! Final LMB field: now _this_ one should never be positive
      region%LMB%LMB( vi) = MIN( 0._dp, region%LMB%LMB_inv( vi) + LMB_trans( vi))

    end do

    ! ! == Effective ice divergence
    ! ! ===========================

    ! ! Set dummy mass balance terms to 0
    ! SMB_dummy    = 0._dp
    ! BMB_dummy    = 0._dp
    ! LMB_dummy    = 0._dp
    ! AMB_dummy    = 0._dp

    ! ! Copy model time step
    ! dt_dummy = dt

    ! ! Use no mass balance to get an estimate of the effective flux divergence
    ! call calc_dHi_dt( region%mesh, region%ice%Hi, region%ice%Hb, region%ice%SL, region%ice%u_vav_b, region%ice%v_vav_b, SMB_dummy, BMB_dummy, LMB_dummy, AMB_dummy, region%ice%fraction_margin, &
    !                   region%ice%mask_noice, dt_dummy, dHi_dt_dummy, Hi_dummy, region%ice%divQ, region%ice%dHi_dt_target)

    ! ! Effective flux divergence
    ! divQ_eff = -dHi_dt_dummy

    ! ! Initialise mask of advancing fronts
    ! mask_advancing_calving_front = .false.

    ! ! Identify advancing calving fronts
    ! do vi = region%mesh%vi1, region%mesh%vi2
    !   if (region%ice%mask_icefree_ocean( vi) .and. divQ_eff(vi) < 0._dp) then
    !     mask_advancing_calving_front( vi) = .true.
    !   end if
    ! end do

    ! ! == Valid areas
    ! ! ==============

    ! ! Set dummy mass balance terms to 0
    ! SMB_dummy    = 0._dp
    ! BMB_dummy    = 0._dp
    ! LMB_dummy    = 0._dp
    ! AMB_dummy    = 0._dp

    ! ! Copy model time step
    ! dt_dummy = dt

    ! ! Use total mass balance to check whether the advancing calving front will survive the incoming lateral mass balance
    ! call calc_dHi_dt( region%mesh, region%ice%Hi, region%ice%Hb, region%ice%SL, region%ice%u_vav_b, region%ice%v_vav_b, region%SMB%SMB, region%BMB%BMB, region%LMB%LMB, AMB_dummy, region%ice%fraction_margin, &
    !                   region%ice%mask_noice, dt_dummy, dHi_dt_dummy, Hi_dummy, region%ice%divQ, region%ice%dHi_dt_target)

    ! ! Check predicted dHi/dt
    ! do vi = region%mesh%vi1, region%mesh%vi2
    !   if (region%ice%mask_icefree_ocean( vi) .and. dHi_dt_dummy( vi) <= 0._dp) then

    !     ! It will not, so do not consider this point an advancing front
    !     mask_advancing_calving_front( vi) = .false.

    !     ! Apply only equilibrium LMB here
    !     region%LMB%LMB( vi) = MIN( 0._dp, region%LMB%LMB_inv( vi))

    !   end if
    ! end do

    ! ! Gather advancing and floating calving front masks from all processes
    ! call gather_to_all( mask_advancing_calving_front, mask_advancing_calving_front_tot)

    ! ! Identify vertices where LMB will operate
    ! do vi = region%mesh%vi1, region%mesh%vi2

    !   ! Valid calving front vertices
    !   if (region%ice%mask_cf_fl( vi)) then
    !     ! Initialise flag
    !     found_advancing_calving_front = .false.

    !     ! Check for advancing front neighbours
    !     do ci = 1, region%mesh%nC( vi)
    !       vj = region%mesh%C( vi,ci)
    !       if (mask_advancing_calving_front_tot( vj)) then
    !         found_advancing_calving_front = .true.
    !         exit
    !       end if
    !     end do

    !     if (found_advancing_calving_front) then
    !       ! do not apply LMB here, since it will be applied on its advancing neighbour
    !       region%LMB%LMB( vi) = 0._dp
    !     end if

    !   end if

    !   ! Valid ocean vertices
    !   if (region%ice%mask_icefree_ocean( vi)) then
    !     ! Initialise flag
    !     found_calving_front_neighbour = .false.

    !     ! Check for calving front neighbours
    !     do ci = 1, region%mesh%nC( vi)
    !       vj = region%mesh%C( vi,ci)
    !       if (mask_cf_fl_tot( vj)) then
    !         found_calving_front_neighbour = .true.
    !         exit
    !       end if
    !     end do

    !     if (.not. found_calving_front_neighbour) then
    !       ! No calving front neighbours: apply only equilibrium LMB here
    !       region%LMB%LMB( vi) = MIN( 0._dp, region%LMB%LMB_inv( vi))
    !     end if

    !   end if

    ! end do

    ! == DENK DROM
    ! ============

    ! if (C%choice_refgeo_PD_ANT == 'idealised' .and. &
    !      (C%choice_refgeo_init_idealised == 'calvmip_circular' .OR. &
    !       C%choice_refgeo_init_idealised == 'calvmip_Thule')) then

    !   if (region%time >= 6000._dp) then
    !     if (C%choice_regions_of_interest == 'CalvMIP_quarter') then
    !       C%ROI_maximum_resolution_grounding_line = 5000._dp
    !       ! C%ROI_maximum_resolution_calving_front  = 8000._dp
    !       ! C%ROI_maximum_resolution_floating_ice   = 10000._dp
    !       ! C%ROI_maximum_resolution_grounded_ice   = 20000._dp
    !     else
    !       C%maximum_resolution_grounding_line = 5000._dp
    !       ! C%maximum_resolution_calving_front  = 8000._dp
    !       ! C%maximum_resolution_floating_ice   = 10000._dp
    !     end if
    !   end if

    !   if (region%time >= 7000._dp) then
    !     if (C%choice_regions_of_interest == 'CalvMIP_quarter') then
    !       C%ROI_maximum_resolution_grounding_line = 3000._dp
    !       ! C%ROI_maximum_resolution_calving_front  = 5000._dp
    !       ! C%ROI_maximum_resolution_floating_ice   = 8000._dp
    !       ! C%ROI_maximum_resolution_grounded_ice   = 16000._dp
    !     else
    !       C%maximum_resolution_grounding_line = 3000._dp
    !       ! C%maximum_resolution_calving_front  = 5000._dp
    !       ! C%maximum_resolution_floating_ice   = 8000._dp
    !     end if
    !   end if

    !   if (region%time >= 9000._dp) then
    !     C%allow_mesh_updates = .false.
    !     if (C%choice_refgeo_init_idealised == 'calvmip_circular') then
    !       C%calving_threshold_thickness_shelf = 10._dp
    !     end if
    !   else
    !     do vi = region%mesh%vi1, region%mesh%vi2
    !       if (SQRT(region%mesh%V( vi,1)**2 + region%mesh%V( vi,2)**2) < 750000._dp) then
    !         region%LMB%LMB( vi) = 0._dp
    !       else
    !         region%LMB%LMB( vi) = -100._dp
    !       end if
    !     end do
    !   end if

    ! end if

    ! Finalise routine path
    call finalise_routine( routine_name)

  end subroutine LMB_inversion

end module ice_model_main
