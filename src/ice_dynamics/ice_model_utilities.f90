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
  use masks_mod
  use zeta_gradients
  use subgrid_ice_margin
  use ice_thickness_safeties

  implicit none

contains

! == Masks
! ========

! == Effective ice thickness
! ==========================

! == Zeta gradients
! =================

! == No-ice mask
! ==============

! == Ice thickness modification
! =============================

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
