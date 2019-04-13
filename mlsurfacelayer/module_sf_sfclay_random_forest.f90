module module_sf_sfclay_random_forest
    use random_forest
    implicit none

    type random_forests_sfclay
        type(decision_tree), allocatable :: friction_velocity(:)
        type(decision_tree), allocatable :: temperature_scale(:)
        type(decision_tree), allocatable :: moisture_scale(:)
    end type random_forests_sfclay

    real, parameter :: r_d = 287.0
    real, parameter :: r_v = 461.0
    real, parameter :: e_s_o = 611.0
    real, parameter :: grav = 9.81
    real, parameter :: eps = 0.622
    real, parameter :: cp = 7.0 * r_d / 2.0
    real, parameter :: r_over_cp = r_d / cp
    real, parameter :: p_1000mb = 100000.0
    real, parameter :: xlv = 2.5e6
    real, parameter :: karman = 0.4

    real, dimension(0:1000 ), save :: psim_stab, psim_unstab, psih_stab, psih_unstab

contains

    subroutine sfclay_random_forest(ims, ime, jms, jme, kms, kme, &
            its, ite, jts, jte, kts, kte, &
            n_vertical_layers, &
            u_3d, v_3d, t_3d, qv_3d, p_3d, dz_3d, mavail, &
            xland, tsk, psfc, swdown, coszen, &
            rf_sfc, &
            regime, br, ust, znt, zol, mol, &
            rmol, hfx, qfx, lh, &
            qstar, u10, v10, th2, t2, q2, qgh, qsfc, &
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
                intent(inout) :: regime, br, ust, znt, zol, hfx, qfx, lh, mol, rmol, &
                chs, chs2, cqs2, flhc, flqc

        real, dimension(ims:ime, jms:jme), &
                intent(out) :: qstar, u10, v10, th2, t2, q2, qgh, qsfc, ck, cka, cd, cda

        real, dimension(its:ite, kts:kts+n_vertical_layers) :: u_2d, &
                v_2d, &
                qv_2d, &
                t_2d, &
                dz_2d

        real, dimension(its:ite) :: p_1d


        integer :: i, j, k

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
            call sfclay_random_forest_2d(its, ite, kts, kts + n_vertical_layers, rf_sfc, &
                    u_2d, v_2d, qv_2d, t_2d, dz_2d, p_1d, &
                    mavail(its:ite, j), xland(its:ite, j), tsk(its:ite, j), psfc(its:ite, j), &
                    swdown(its:ite, j), coszen(its:ite, j), &
                    regime(its:ite, j), br(its:ite, j), ust(its:ite, j), znt(its:ite, j), &
                    zol(its:ite, j), hfx(its:ite, j), &
                    qfx(its:ite, j), lh(its:ite, j), mol(its:ite, j), rmol(its:ite, j), &
                    qstar(its:ite, j), u10(its:ite, j), v10(its:ite, j), th2(its:ite, j), t2(its:ite, j), &
                    q2(its:ite, j), qgh(its:ite, j), qsfc(its:ite, j), chs(its:ite, j), &
                    chs2(its:ite, j), cqs2(its:ite, j), flhc(its:ite, j), flqc(its:ite, j), &
                    ck(its:ite, j), cka(its:ite, j), cd(its:ite, j), cda(its:ite, j))
        enddo
    end subroutine sfclay_random_forest

    subroutine sfclay_random_forest_2d(its, ite, kts, kte, rf_sfc, &
            u_2d, v_2d, qv_2d, t_2d, dz_2d, p_1d, &
            mavail, xland, tsk, psfc, swdown, coszen, &
            regime, br, ust, znt, zol, hfx, qfx, lh, mol, rmol, &
            qstar, u10, v10, th2, t2, q2, qgh, qsfc, &
            chs, chs2, cqs2, flhc, flqc, &
            ck, cka, cd, cda)
        integer, intent(in) :: its, ite, kts, kte
        type(random_forests_sfclay), intent(in) :: rf_sfc
        real, dimension(its:ite, kts:kte), intent(in) :: u_2d, v_2d, &
                qv_2d, t_2d, dz_2d
        real, dimension(its:ite), intent(in) :: p_1d
        real, dimension(its:ite), intent(in) :: mavail, xland, tsk, psfc, swdown, coszen
        real, dimension(its:ite), &
                intent(inout) :: regime, br, ust, znt, zol, hfx, qfx, lh, mol, rmol, &
                chs, chs2, cqs2, flhc, flqc
        real, dimension(its:ite), &
                intent(out) :: qstar, u10, v10, th2, t2, q2, qgh, qsfc, ck, cka, cd, cda
        ! declare derived variables
        real, dimension(kts:kte) :: potential_temperature, virtual_potential_temperature, wind_speed
        real :: zenith, d_vpt_d_z, skin_saturation_mixing_ratio, skin_potential_temperature, &
                skin_saturation_vapor_pressure, skin_virtual_potential_temperature, total_dz, z_a
        real, dimension(11) :: ust_rf_inputs
        real, dimension(12) :: tstar_rf_inputs
        integer:: i, k
        do i = its, ite
            ! Calculate derived variables
            total_dz = 0
            z_a = dz_2d(i, 1)
            do k = kts, kte
                potential_temperature(k) = t_2d(i, k) * (psfc(i) / p_1000mb) ** r_over_cp
                virtual_potential_temperature(k) = potential_temperature(k) * (1. + 0.61 * qv_2d(i, k))
                wind_speed(k) = sqrt(u_2d(i, k) ** 2. + v_2d(i, k) ** 2.)
                total_dz = total_dz + dz_2d(i, k)
            end do
            zenith = acos(coszen(i))
            d_vpt_d_z = (virtual_potential_temperature(kte) - virtual_potential_temperature(kts)) / (total_dz - dz_2d(i, kts))
            skin_saturation_vapor_pressure = e_s_o * exp(xlv / r_v * (1.0 / 273.0 - 1. / tsk(i)))
            skin_saturation_mixing_ratio = eps * skin_saturation_vapor_pressure / (psfc(i) - skin_saturation_vapor_pressure)
            skin_potential_temperature = tsk(i) * (psfc(i) / p_1000mb) ** r_over_cp
            skin_virtual_potential_temperature = skin_potential_temperature * (1. + 0.61 * skin_saturation_mixing_ratio)
            br(i) = grav / virtual_potential_temperature(kts) * dz_2d(i, kts) * &
                    (virtual_potential_temperature(kts) - skin_virtual_potential_temperature) / (wind_speed(kts) * wind_speed(kts))

            ! Input variables into random forest input arrays
            ust_rf_inputs = (/ wind_speed(kts), wind_speed(kte), psfc(i), d_vpt_d_z, &
                    skin_virtual_potential_temperature, skin_saturation_mixing_ratio, swdown(i), &
                    virtual_potential_temperature(kts), virtual_potential_temperature(kte), &
                    br(i), zenith /)
            tstar_rf_inputs = (/ wind_speed(kts), wind_speed(kte), psfc(i), d_vpt_d_z, &
                    skin_virtual_potential_temperature, skin_saturation_mixing_ratio, swdown(i), &
                    virtual_potential_temperature(kts), virtual_potential_temperature(kte), &
                    mavail(i), br(i), zenith /)
            ! Run random forests to get ust, mol, and qstar
            ust(i) = random_forest_predict(real(ust_rf_inputs, 8), rf_sfc%friction_velocity)
            mol(i) = random_forest_predict(real(tstar_rf_inputs, 8), rf_sfc%temperature_scale)
            qstar(i) = random_forest_predict(real(tstar_rf_inputs, 8), rf_sfc%moisture_scale)

            ! Calculate diagnostics
            zol(i) = karman * grav / potential_temperature(i, 1) * z_a * mol(i) / (ust(i) * ust(i))

        end do
    end subroutine sfclay_random_forest_2d

    subroutine init_sfclay_random_forest(friction_velocity_random_forest_path, &
            temperature_scale_random_forest_path, &
            moisture_scale_random_forest_path, &
            rf_sfc)
        character(len=*), intent(in) :: friction_velocity_random_forest_path
        character(len=*), intent(in) :: temperature_scale_random_forest_path
        character(len=*), intent(in) :: moisture_scale_random_forest_path
        type(random_forests_sfclay), intent(out) :: rf_sfc


        call load_random_forest(friction_velocity_random_forest_path, rf_sfc%friction_velocity)
        call load_random_forest(temperature_scale_random_forest_path, rf_sfc%temperature_scale)
        call load_random_forest(moisture_scale_random_forest_path, rf_sfc%moisture_scale)
        call sfclayrev_init_table()
    end subroutine init_sfclay_random_forest

    subroutine sfclayrev_init_table()
        ! Initialize lookup table of stability functions.
        integer                   ::      n
        real                      ::      zolf

        do n=0, 1000
            ! stable function tables
            zolf = float(n)*0.01
            psim_stab(n)=psim_stable_full(zolf)
            psih_stab(n)=psih_stable_full(zolf)

            ! unstable function tables
            zolf = -float(n)*0.01
            psim_unstab(n)=psim_unstable_full(zolf)
            psih_unstab(n)=psih_unstable_full(zolf)
        end do

    end subroutine sfclayrev_init_table

    function zolri(ri, z, z0)
        !
        if (ri < 0.)then
            x1 = -5.
            x2 = 0.
        else
            x1 = 0.
            x2 = 5.
        endif
        !
        fx1 = zolri2(x1, ri, z, z0)
        fx2 = zolri2(x2, ri, z, z0)
        do while (abs(x1 - x2) > 0.01)
            if (abs(fx2) < abs(fx1)) then
                x1 = x1 - fx1 / (fx2 - fx1) * (x2 - x1)
                fx1 = zolri2(x1, ri, z, z0)
                zolri = x1
            else
                x2 = x2 - fx2 / (fx2 - fx1) * (x2 - x1)
                fx2 = zolri2(x2, ri, z, z0)
                zolri = x2
            endif
            !
        enddo
        !

        return
    end function

    !
    ! -----------------------------------------------------------------------
    !
    function zolri2(zol2, ri2, z, z0)
        !
        if(zol2 * ri2 < 0.)zol2 = 0.  ! limit zol2 - must be same sign as ri2
        !
        zol20 = zol2 * z0 / z ! z0/L
        zol3 = zol2 + zol20 ! (z+z0)/L
        !
        if (ri2 < 0) then
            psix2 = log((z + z0) / z0) - (psim_unstable(zol3) - psim_unstable(zol20))
            psih2 = log((z + z0) / z0) - (psih_unstable(zol3) - psih_unstable(zol20))
        else
            psix2 = log((z + z0) / z0) - (psim_stable(zol3) - psim_stable(zol20))
            psih2 = log((z + z0) / z0) - (psih_stable(zol3) - psih_stable(zol20))
        endif
        !
        zolri2 = zol2 * psih2 / psix2**2 - ri2
        !
        return
    end function
    !
    ! ... integrated similarity functions ...
    !
    function psim_stable_full(zolf)
        psim_stable_full = -6.1 * log(zolf + (1 + zolf**2.5)**(1. / 2.5))
        return
    end function

    function psih_stable_full(zolf)
        psih_stable_full = -5.3 * log(zolf + (1 + zolf**1.1)**(1. / 1.1))
        return
    end function

    function psim_unstable_full(zolf)
        x = (1. - 16. * zolf)**.25
        psimk = 2 * alog(0.5 * (1 + X)) + alog(0.5 * (1 + X * X)) - 2. * atan(X) + 2. * atan(1.)
        !
        ym = (1. - 10. * zolf)**0.33
        psimc = (3. / 2.) * log((ym**2. + ym + 1.) / 3.) - sqrt(3.) * atan((2. * ym + 1) / sqrt(3.)) + 4. * atan(1.) / sqrt(3.)
        !
        psim_unstable_full = (psimk + zolf**2 * (psimc)) / (1 + zolf**2.)

        return
    end function

    function psih_unstable_full(zolf)
        y = (1. - 16. * zolf)**.5
        psihk = 2. * log((1 + y) / 2.)
        !
        yh = (1. - 34. * zolf)**0.33
        psihc = (3. / 2.) * log((yh**2. + yh + 1.) / 3.) - sqrt(3.) * ATAN((2. * yh + 1) / sqrt(3.)) + 4. * ATAN(1.) / sqrt(3.)
        !
        psih_unstable_full = (psihk + zolf**2 * (psihc)) / (1 + zolf**2.)

        return
    end function

    ! look-up table functions
    function psim_stable(zolf)
        integer :: nzol
        real :: rzol
        nzol = int(zolf * 100.)
        rzol = zolf * 100. - nzol
        if (nzol + 1 <= 1000) then
            psim_stable = psim_stab(nzol) + rzol * (psim_stab(nzol + 1) - psim_stab(nzol))
        else
            psim_stable = psim_stable_full(zolf)
        endif
        return
    end function

    function psih_stable(zolf)
        integer :: nzol
        real :: rzol
        nzol = int(zolf * 100.)
        rzol = zolf * 100. - nzol
        if (nzol + 1 <= 1000) then
            psih_stable = psih_stab(nzol) + rzol * (psih_stab(nzol + 1) - psih_stab(nzol))
        else
            psih_stable = psih_stable_full(zolf)
        endif
        return
    end function

    function psim_unstable(zolf)
        integer :: nzol
        real :: rzol
        nzol = int(-zolf * 100.)
        rzol = -zolf * 100. - nzol
        if (nzol + 1 <= 1000) then
            psim_unstable = psim_unstab(nzol) + rzol * (psim_unstab(nzol + 1) - psim_unstab(nzol))
        else
            psim_unstable = psim_unstable_full(zolf)
        endif
        return
    end function

    function psih_unstable(zolf)
        integer :: nzol
        real :: rzol
        nzol = int(-zolf * 100.)
        rzol = -zolf * 100. - nzol
        if (nzol + 1 <= 1000) then
            psih_unstable = psih_unstab(nzol) + rzol * (psih_unstab(nzol + 1) - psih_unstab(nzol))
        else
            psih_unstable = psih_unstable_full(zolf)
        endif
        return
    end function


end module module_sf_sfclay_random_forest