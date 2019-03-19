module module_sf_sfclay_random_forest
    use random_forest
    implicit none

    type random_forests_sfclay
        type(decision_tree), allocatable :: friction_velocity(:)
        type(decision_tree), allocatable :: temperature_scale(:)
        type(decision_tree), allocatable :: moisture_scale(:)
    end type random_forests_sfclay

    contains

    subroutine sfclay_random_forest(ims, ime, jms, jme, kms, kme, &
                                    its, ite, jts, jte, kts, kte, &
                                    n_vertical_layers, &
                                    u_3d, v_3d, t_3d, qv_3d, p_3d, dz_3d, mavail, &
                                    xland, tsk, psfc, swdown, coszen, &
                                    rf_sfc, &
                                    regime, br, ust, zol, mol, &
                                    rmol, hfx, qfx, lh, &
                                    u10, v10, th2, t2, q2, qgh, qsfc, &
                                    chs, chs2, cqs2, flhc, flqc, &
                                    ck, cka, cd, cda)
        ! Definition: This subroutine calculates the surface layer fluxes and associated diagnostics
        ! using random forests to calculate the friction velocity, temperature scale, and moisture scale
        ! from the atmospheric and land surface model fields.
        !
        ! Input fields:
        !   ims, ime (integer):  start and end indices for i (x-direction) in memory
        !   jms, jme (integer):  start and end indices for j (y-direction) in memory
        !   kms, kme (integer):  start and end indices for k (z-direction) in memory
        !   n_vertical_layers (integer): number of vertical layers used by random forest
        !   u_3d (real) : The 3-dimensional U-component of the wind in m s-1
        !   v_3d (real) : The 3-dimensional V-component of the wind in m s-1
        !   t_3d (real): The 3-dimensional temperature in K
        !   qv_3d (real): The 3-dimensional water vapor mixing ratio in kg kg-1
        !   p_3d (real): The 3-dimensional pressure field in Pa
        !   dz_3d (real): The change in height between full model levels in m.
        !   mavail (real): Moisture availability, a scaled measure of soil moisture that ranges from 0 to 1.
        !   xland  (real): Land-sea mask, 1 if by land and 2 if by sea.
        !   tsk (real): Radiative skin temperature from the land surface model in K.
        !   psfc (real): Pressure at the surface in Pa.
        !   swdown (real): Downward shortwave irradiance at the surface in W m-2.
        !   coszen (real): Cosine of the solar zenith angle.
        !   rf_sfc (random_forests_sfclay): Collection of random forests for each surface layer process
        !
        ! Output fields:
        !   regime (real): The stability regime
        !   br (real): Bulk richardson number
        !   ust (real): friction velocity in m s-1
        !   hfx (real):
        implicit none
        integer, intent(in) :: ims, ime, jms, jme, kms, kme, &
                               its, ite, jts, jte, kts, kte, &
                               n_vertical_layers

        real, dimension(ims:ime, kms:kme, jms:jme), &
                intent(in) :: qv_3d, &
                p_3d, &
                t_3d, &
                u_3d, &
                v_3d, &
                dz_3d

        real, dimension(ims:ime, jms:jme), &
                intent(in) :: mavail, &
                xland, &
                tsk, &
                psfc, &
                swdown, &
                coszen

        type(random_forests_sfclay), intent(in) :: rf_sfc

        real, dimension(ims:ime, jms:jme), &
                intent(inout) :: regime, &
                br, &
                hfx, &
                qfx, &
                lh, &
                mol, &
                rmol

        real, dimension(ims:ime, jms:jme), &
                intent(out) :: u10, &
                v10, &
                th2, &
                t2, &
                q2, &
                qsfc

        real, dimension(its:ite, kts:kts+n_vertical_layers) :: u_2d, &
                v_2d, &
                qv_2d, &
                t_2d, &
                dz_2d

        real, dimension(its:ite) :: p_1d


        integer :: i, j

        do j = jts, jte
            do i = its, ite
                do k = kts, kts + n_vertical_layers
                    u_2d(i, k) = u_3d(i, k, j)
                    v_2d(i, k) = v_3d(i, k, j)
                    qv_2d(i, k) = qv_3d(i, k, j)
                    t_2d(i, k) = t_3d(i, k, j)
                    dz_2d(i, k) = dz_3d(i, k, j)
                end do
                p_1d(i) = p_3d(i, kts, j)
            enddo
            call sfclay_random_forest_2d(ims, ime, kms, kme, u_2d, v_2d, qv_2d, p_1d, )
        enddo
    end subroutine sfclay_random_forest

    subroutine sfclay_random_forest_2d(ims, ime, kms, kme, &
                                       its, ite, kts, kte, &
                                       u_2d, v_2d, qv_2d, t_2d, dz_2d, p_1d, &
                                       rf_sfc)
        implicit none
        integer, intent(in) :: ims, ime, kms, kme, its, ite, kts, kte
        real, dimension(its:ite, kts:kte), intent(in) :: u_2d, v_2d, &
                                                         qv_2d, t_2d, dz2d
        real, dimension(its:ite), intent(in) :: p_1d
        type(random_forests_sfclay), intent(in) :: rf_sfc

        ! Calculate derived variables

        ! Input variables into random forest input arrays

        ! Run random forests to get ust, mol, and qstar

        ! Calculate diagnostics
    end subroutine sfclay_random_forest_2d

    subroutine init_sfclay_random_forest(friction_velocity_random_forest_path, &
                                         temperature_scale_random_forest_path, &
                                         moisture_scale_random_forest_path, &
                                         rf_sfc)
        implicit none
        character(len=*), intent(in) :: friction_velocity_random_forest_path
        character(len=*), intent(in) :: temperature_scale_random_forest_path
        character(len=*), intent(in) :: moisture_scale_random_forest_path
        type(random_forests_sfclay), dimension(:), intent(out) :: rf_sfc


        call load_random_forest(friction_velocity_random_forest_path, rf_sfc%friction_velocity)
        call load_random_forest(temperature_scale_random_forest_path, rf_sfc%temperature_scale)
        call load_random_forest(moisture_scale_random_forest_path, rf_sfc%moisture_scale)
    end subroutine init_sfclay_random_forest

end module module_sf_sfclay_random_forest