MODULE geothermal_heat_flux

  ! Subroutines for calculating the geothermal heat flux

! ===== Preamble =====
! ====================

  USE precisions                                             , ONLY: dp
  USE mpi_basic                                              , ONLY: par, sync
  USE control_resources_and_error_messaging                  , ONLY: crash, warning, happy, colour_string, init_routine, finalise_routine
  USE model_configuration                                    , ONLY: C
  USE mesh_types                                             , ONLY: type_mesh
  USE ice_model_types                                        , ONLY: type_ice_model
  USE netcdf_input                                           , ONLY: read_field_from_file_2D

  IMPLICIT NONE

CONTAINS

! ===== Main routines =====
! =========================

  SUBROUTINE initialise_geothermal_heat_flux( mesh, ice)
    ! Initialise the geothermal heat flux

    IMPLICIT NONE

    ! Input variables:
    TYPE(type_mesh),                     INTENT(IN)    :: mesh
    TYPE(type_ice_model),                INTENT(INOUT) :: ice

    ! Local variables:
    CHARACTER(LEN=256), PARAMETER                      :: routine_name = 'initialise_geothermal_heat_flux'

    ! Add routine to path
    CALL init_routine( routine_name)

    SELECT CASE (C%choice_geothermal_heat_flux)

      CASE ('uniform')
        ! Uniform value over whole domain
        ice%geothermal_heat_flux = C%uniform_geothermal_heat_flux

      CASE ('read_from_file')
        ! Spatially variable field

        ! Print to terminal
        IF (par%master) WRITE(0,*) '  Initialising geothermal heat flux from file "' // &
          colour_string( TRIM( C%filename_geothermal_heat_flux),'light blue') // '"...'

        ! Read the data from the provided NetCDF file
        CALL read_field_from_file_2D( C%filename_geothermal_heat_flux, 'hflux', mesh, ice%geothermal_heat_flux)

      CASE DEFAULT
        ! Unknown case
        CALL crash('unknown choice_geothermal_heat_flux "' // TRIM( C%choice_geothermal_heat_flux) // '"')

    END SELECT

    ! Finalise routine path
    CALL finalise_routine( routine_name)

  END SUBROUTINE initialise_geothermal_heat_flux

END MODULE geothermal_heat_flux
