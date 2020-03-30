module module_sf_sfclay_neural_net
    use module_neural_net
    implicit none

    type neural_net_sfclay
        type(Dense), allocatable :: friction_velocity(:)
        type(Dense), allocatable :: temperature_scale(:)
        type(Dense), allocatable :: moisture_scale(:)
        real(8), allocatable :: friction_velocity_scale_values(:, :)
        real(8), allocatable :: temperature_scale_scale_values(:, :)
        real(8), allocatable :: moisture_scale_scale_values(:, :)
    end type neural_net_sfclay

    real, parameter :: r_dry = 287.0
    real, parameter :: r_v = 461.0
    real, parameter :: e_s_o = 611.0
    real, parameter :: grav = 9.81
    real, parameter :: eps = 0.622
    real, parameter :: c_p = 7.0 * r_dry / 2.0
    real, parameter :: r_over_cp = r_dry / c_p
    real, parameter :: p_1000mb = 100000.0
    real, parameter :: x_lv = 2.5e6
    real, parameter :: vonkarman = 0.4

    real, dimension(0:1000 ), save :: psim_stab, psim_unstab, psih_stab, psih_unstab
    ! Random forest storage type. Initialized in subroutine
    type(neural_net_sfclay), save :: nn_sfc

contains

    subroutine init_sfclay_neural_net(neural_net_path)
        character(len=*), intent(in) :: neural_net_path
        integer :: friction_velocity_num_inputs, temperature_scale_num_inputs, moisture_scale_num_inputs, batch_size

        friction_velocity_num_inputs = 13
        temperature_scale_num_inputs = 15
        moisture_scale_num_inputs = 16
        batch_size = 1

        allocate(nn_sfc%friction_velocity_scale_values(friction_velocity_num_inputs, 2))
        allocate(nn_sfc%temperature_scale_scale_values(temperature_scale_num_inputs, 2))
        allocate(nn_sfc%moisture_scale_scale_values(moisture_scale_num_inputs, 2))

        call load_scale_values(neural_net_path // "friction_velocity_scale_values.csv", &
            friction_velocity_num_inputs, nn_sfc%friction_velocity_scale_values)
        call load_scale_values(neural_net_path // "temperature_scale_scale_values.csv", &
            temperature_scale_num_inputs, nn_sfc%temperature_scale_scale_values)
        call load_scale_values(neural_net_path // "moisture_scale_scale_values.csv", &
            moisture_scale_num_inputs, nn_sfc%moisture_scale_scale_values)
        call init_neural_net(neural_net_path // "friction_velocity-neural_network_fortran.nc", &
            batch_size, nn_sfc%friction_velocity)
        call init_neural_net(neural_net_path // "temperature_scale-neural_network_fortran.nc", &
            batch_size, nn_sfc%temperature_scale)
        call init_neural_net(neural_net_path // "moisture_scale-neural_network_fortran.nc", &
            batch_size, nn_sfc%moisture_scale)
    end subroutine init_sfclay_neural_net

    subroutine sfclay_neural_net(ims, ime, jms, jme, kms, kme, &
            its, ite, jts, jte, kts, kte, &
!            r_d,r_v,cp,xlv, &
!            karman, &
            n_vertical_layers, &
            u_3d, v_3d, t_3d, qv_3d, p_3d, dz_3d, mavail, &
            xland, tsk, psfc, swdown, coszen, wspd, znt, &
            br, ust, mol, qstar, zol, &
            rmol, hfx, qfx, lh, &
            regime, psim, psih, fm, fh, &
            u10, v10, th2, t2, q2, qsfc, &
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
        !   znt (real): roughness length in m.
        !
        ! Output fields:
        !   br (real): Bulk richardson number
        !   ust (real): friction velocity in m s-1
        !   mol (real): temperature scale in K
        !   qstar (real): moisture scale in kg kg-1
        !   zol (real): z over Obukov Length (non-dimensional)
        !   rmol (real): Obukov Length (m)
        !   hfx (real): sensible heat flux in W m-2
        !   qfx (real): moisture flux in
        !   regime (real): The stability regime
        !   psim (real): stability function for momentum
        !   psih (real): stability function for sensible heat
        !   fm (real): integrated stability function for momentum
        !   fh (real): integrated stability function for heat
        !   u10 (real): 10 m zonal wind m/s
        !   v10 (real): 10 m meridional wind m / s
        !   t2 (real): 2 m temperature K
        !   q2 (real): 2 m mixing ratio kg / kg
        !   qsfc (real): saturation mixing ratio at the surface (kg / kg)
        !   chs (real): heat/moisture exchange coefficient for lsm (m /s)
        !   chs2 (real): heat exchange coefficient for lsm at 2 m (m /s)
        !   cqs2 (real): moisture exchange coefficient for lsm at 2 m
        !   flhc (real): exchange coefficient for heat (w/m^2/K)
        !   flqc (real): exchange coefficient for moisture (W/m^2/s)
        integer, intent(in) :: ims, ime, jms, jme, kms, kme, &
                its, ite, jts, jte, kts, kte, &
                n_vertical_layers

!        real, intent(in) :: r_d, cp, xlv, karman

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
                coszen, &
                znt
        real, dimension(ims:ime, jms:jme), &
                intent(inout) :: regime, wspd, br, ust, zol, hfx, qfx, lh, mol, rmol, &
                chs, chs2, cqs2, flhc, flqc, psim, psih, fm, fh

        real, dimension(ims:ime, jms:jme), &
                intent(out) :: qstar, u10, v10, th2, t2, q2, qsfc, ck, cka, cd, cda

        real, dimension(its:ite, kts:kts+n_vertical_layers) :: u_2d, &
                v_2d, &
                qv_2d, &
                t_2d, &
                dz_2d, &
                p_2d

        integer :: i, j, k, kstop

        kstop = kts + n_vertical_layers
        do j = jts, jte
            do i = its, ite
                do k = kts, kstop
                    u_2d(i, k) = u_3d(i, k, j)
                    v_2d(i, k) = v_3d(i, k, j)
                    qv_2d(i, k) = qv_3d(i, k, j)
                    t_2d(i, k) = t_3d(i, k, j)
                    dz_2d(i, k) = dz_3d(i, k, j)
                    p_2d(i, k) = p_3d(i, kts, j)

                end do
            end do
            call sfclay_neuralnet_2d(its, ite, kts, kstop, &
                    u_2d, v_2d, qv_2d, t_2d, dz_2d, p_2d, &
                    mavail(its:ite, j), xland(its:ite, j), tsk(its:ite, j), psfc(its:ite, j), &
                    swdown(its:ite, j), coszen(its:ite, j), wspd(its:ite, j), &
                    regime(its:ite, j), br(its:ite, j), ust(its:ite, j), znt(its:ite, j), &
                    zol(its:ite, j), hfx(its:ite, j), &
                    qfx(its:ite, j), lh(its:ite, j), mol(its:ite, j), rmol(its:ite, j), &
                    qstar(its:ite, j), psim(its:ite, j), psih(its:ite, j), fm(its:ite, j), fh(its:ite, j), &
                    u10(its:ite, j), v10(its:ite, j), th2(its:ite, j), t2(its:ite, j), &
                    q2(its:ite, j), qsfc(its:ite, j), chs(its:ite, j), &
                    chs2(its:ite, j), cqs2(its:ite, j), flhc(its:ite, j), flqc(its:ite, j), &
                    ck(its:ite, j), cka(its:ite, j), cd(its:ite, j), cda(its:ite, j))

        enddo
    end subroutine sfclay_neural_net

    subroutine sfclay_neuralnet_2d(its, ite, kts, kstop, &
                                       u_2d, v_2d, qv_2d, t_2d, dz_2d, p_2d, &
                                       mavail, xland, tsk, psfc, swdown, coszen, wspd, &
                                       regime, br, ust, znt, zol, hfx, qfx, lh, mol, rmol, &
                                       qstar, psim, psih, fm, fh, u10, v10, th2, t2, q2, qsfc, &
                                       chs, chs2, cqs2, flhc, flqc, &
                                       ck, cka, cd, cda)
        real, parameter :: xka = 2.4e-5
        integer, intent(in) :: its, ite, kts, kstop
        real, dimension(its:ite, kts:kstop), intent(in) :: u_2d, v_2d, &
                qv_2d, t_2d, dz_2d, p_2d
        real, dimension(its:ite), intent(in) :: mavail, xland, tsk, psfc, swdown, coszen, znt
        real, dimension(its:ite), intent(inout) :: regime, wspd, br, ust, zol, hfx, qfx, lh, mol, rmol, &
                chs, chs2, cqs2, flhc, flqc
        real, dimension(its:ite), &
                intent(out) :: qstar, psim, psih, fm, fh, u10, v10, th2, t2, q2, qsfc, ck, cka, cd, cda
        ! declare derived variables
        real, dimension(kts:kstop) :: potential_temperature, virtual_potential_temperature, wind_speed, &
                saturation_vapor_pressure, saturation_mixing_ratio, relative_humidity
        real :: zenith, d_vpt_d_z, skin_potential_temperature, virtual_temperature, &
                skin_saturation_vapor_pressure, skin_virtual_potential_temperature, total_dz, z_a, rho, cpm, &
                gz1_o_z0, gz2_o_z0, gz10_o_z0, pq, pq2, pq10, psih2, psim2, psih10, psim10, psiq, psiq2, psiq10, zl, &
                psix, psix2, psix10, psit, psit2, zl2, zl10, zol_0, zol_2, zol_10, zol_z_a
        real :: pot_temp_diff, qv_diff, pot_temp_skin_diff_1, pot_temp_skin_diff_2
        real :: qv_skin_diff_1, qv_skin_diff_2
        integer, parameter :: ust_in_size = 13
        integer, parameter :: tstar_in_size = 15
        integer, parameter :: qstar_in_size = 16
        real(8), dimension(1, ust_in_size) :: ust_inputs, ust_norm_inputs
        real(8), dimension(1, tstar_in_size) :: tstar_inputs, tstar_norm_inputs
        real(8), dimension(1, qstar_in_size) :: qstar_inputs, qstar_norm_inputs
        real(8) :: ust_temp(1, 1), mol_temp(1, 1), qstar_temp(1, 1)
        REAL, PARAMETER :: pi_val = 3.1415927
        integer:: i, k
        do i = its, ite
            ! Calculate derived variables
            total_dz = 0
            z_a = dz_2d(i, 1)
            do k = kts, kstop
                potential_temperature(k) = t_2d(i, k) * (p_2d(i, k) / p_1000mb) ** r_over_cp
                virtual_potential_temperature(k) = potential_temperature(k) * (1. + 0.61 * qv_2d(i, k))
                wind_speed(k) = sqrt(u_2d(i, k) ** 2. + v_2d(i, k) ** 2.)
                saturation_vapor_pressure(k) = e_s_o * exp(x_lv / r_v * (1.0 / 273.0 - 1. / t_2d(i, k)))
                saturation_mixing_ratio(k) = eps * saturation_vapor_pressure(k) / (p_2d(i, k) - saturation_vapor_pressure(k))
                relative_humidity(k) = qv_2d(i, k) / saturation_mixing_ratio(k) * 100.0
                if (wind_speed(k) < 0.1) then
                    wind_speed(k) = 0.1
                endif
                total_dz = total_dz + dz_2d(i, k)
            end do
            virtual_temperature = t_2d(i, kts) * (1. + 0.61 * qv_2d(i, kts))
            wspd(i) = wind_speed(kts)
            zenith = acos(coszen(i)) * 180. / pi_val
            d_vpt_d_z = (virtual_potential_temperature(kstop) - virtual_potential_temperature(kts)) / (total_dz - dz_2d(i, kts))
            skin_saturation_vapor_pressure = e_s_o * exp(x_lv / r_v * (1.0 / 273.0 - 1. / tsk(i)))
            qsfc(i) = eps * skin_saturation_vapor_pressure / (psfc(i) - skin_saturation_vapor_pressure)
            skin_potential_temperature = tsk(i) * (psfc(i) / p_1000mb) ** r_over_cp
            skin_virtual_potential_temperature = skin_potential_temperature * (1. + 0.61 * qsfc(i))
            br(i) = grav / potential_temperature(kts) * dz_2d(i, kts) * &
                    (virtual_potential_temperature(kts) - skin_virtual_potential_temperature) / (wind_speed(kts) * wind_speed(kts))
            pot_temp_skin_diff_1 = (skin_potential_temperature - potential_temperature(kts)) / (dz_2d(i, kts))
            pot_temp_skin_diff_2 = (skin_potential_temperature - potential_temperature(kstop)) / (dz_2d(i, kstop) + dz_2d(i, kts))
            qv_skin_diff_1 = (qsfc(i) - qv_2d(i, kts)) / (dz_2d(i, kts)) * 1000.0
            qv_skin_diff_2 = (qsfc(i) - qv_2d(i, kstop)) / (dz_2d(i, kstop) + dz_2d(i, kts)) * 1000.0
            ust_inputs = reshape((/ wind_speed(kts), wind_speed(kstop), pot_temp_skin_diff_1, qv_skin_diff_1, br(i), zenith, &
                u_2d(i, kts), v_2d(i, kts), u_2d(i, kstop), v_2d(i, kstop), swdown(i), &
                relative_humidity(kts), relative_humidity(kstop) /), (/ 1,  ust_in_size /))
            tstar_inputs = reshape((/ qv_skin_diff_1, qv_skin_diff_2, pot_temp_skin_diff_1, &
                    pot_temp_skin_diff_2, br(i), wspd(kts), wind_speed(kstop), &
                    u_2d(i, kts), v_2d(i, kts), u_2d(i, kstop), v_2d(i, kstop), zenith, swdown(i), &
                    relative_humidity(kts), relative_humidity(kstop) /), (/ 1, tstar_in_size /))
            qstar_inputs = reshape((/ qv_skin_diff_1, qv_skin_diff_2, pot_temp_skin_diff_1, &
                    pot_temp_skin_diff_2, br(i), wspd(kts), wind_speed(kstop), &
                    u_2d(i, kts), v_2d(i, kts), u_2d(i, kstop), v_2d(i, kstop), zenith, swdown(i), &
                    mavail(i), relative_humidity(kts), relative_humidity(kstop) /), (/ 1, qstar_in_size /))
            ! Run neural nets to get ust, mol, and qstar
            call standard_scaler_transform(ust_inputs, nn_sfc%friction_velocity_scale_values, ust_norm_inputs)
            call standard_scaler_transform(tstar_inputs, nn_sfc%temperature_scale_scale_values, tstar_norm_inputs)
            call standard_scaler_transform(qstar_inputs, nn_sfc%moisture_scale_scale_values, qstar_norm_inputs)
            call neural_net_predict(real(ust_norm_inputs, 8), nn_sfc%friction_velocity, ust_temp)
            call neural_net_predict(real(tstar_norm_inputs, 8), nn_sfc%temperature_scale, mol_temp)
            call neural_net_predict(real(qstar_norm_inputs, 8), nn_sfc%moisture_scale, qstar_temp)
            ust(i) = real(ust_temp(1, 1), 4)
            mol(i) = real(mol_temp(1, 1), 4)
            qstar(i) = real(qstar_temp(1, 1) / 1000.0, 4)
            ! Calculate diagnostics
            zol(i) = vonkarman * grav / potential_temperature(kts) * z_a * mol(i) / (ust(i) * ust(i))

            if (zol(i) < -10.) then
                zol(i) = -1.
            endif

            if (zol(i) > 1.) then
                zol(i) = 1.
            endif

            rmol(i) = zol(i) / z_a
            ! We need the zol values at different heights to calculate 2 m t and q and 10 m winds
            zol_z_a = zol(i) * (z_a + znt(i)) / z_a
            zol_10 = zol(i) * (10. + znt(i)) / z_a
            zol_2 = zol(i) * (2. + znt(i)) / z_a
            zol_0 = zol(i) * znt(i) / z_a
            zl2 = 2. / z_a * zol(i)
            zl10 = 10. / z_a * zol(i)

            if ((xland(i) - 1.5) >= 0.) then
                zl = znt(i)  ! (0.01)/l
            else
                zl = 0.01
            endif
            gz1_o_z0 = alog((z_a + znt(i)) / znt(i))
            gz2_o_z0 = alog((2. + znt(i)) / znt(i))
            gz10_o_z0 = alog((10. + znt(i)) / znt(i))
            ! Calculate stability regimes for 2 and 10 m diagnostics
            if (br(i) > 0.) then
                regime(i) = 1.
                psim(i) = psim_stable(zol_z_a) - psim_stable(zol_0)
                psih(i) = psim_stable(zol_z_a) - psih_stable(zol_0)
                psim10 = psim_stable(zol_10) - psim_stable(zol_0)
                psih10 = psih_stable(zol_10) - psih_stable(zol_0)

                psim2 = psim_stable(zol_2) - psim_stable(zol_0)
                psih2 = psih_stable(zol_2) - psih_stable(zol_0)

                pq = psih_stable(zol(i)) - psih_stable(zl)
                pq2 = psih_stable(zl2) - psih_stable(zl)
                pq10 = psih_stable(zl10) - psih_stable(zl)
            elseif (br(i) == 0.) then
                regime(i) = 3.
                psim(i) = 0.
                psih(i) = 0.
                psim10 = 0.
                psih10 = 0.
                psim2 = 0.
                psih2 = 0.
                pq = 0.
                pq2 = 0.
                pq10 = 0.
            else
                regime(i) = 4.
                psim(i) = psim_unstable(zol_z_a) - psim_unstable(zol_0)
                psih(i) = psih_unstable(zol_z_a) - psim_unstable(zol_0)

                psim10 = psim_unstable(zol_10) - psim_unstable(zol_0)
                psih10 = psih_unstable(zol_10) - psih_unstable(zol_0)

                psim2  = psim_unstable(zol_2) - psim_unstable(zol_0)
                psih2 = psih_unstable(zol_2) - psih_unstable(zol_0)

                pq = psih_unstable(zol(i)) - psih_unstable(zl)
                pq2 = psih_unstable(zl2) - psih_unstable(zl)
                pq10 = psih_unstable(zl10) - psih_unstable(zl)
            end if
            psix = gz1_o_z0 - psim(i)
            psix10 = gz10_o_z0 - psim10

            psit = gz1_o_z0 - psih(i)
            psit2 = gz2_o_z0 - psih2
            psiq = alog(vonkarman * ust(i) * z_a / xka + z_a / zl) - pq
            psiq2 = alog(vonkarman * ust(i) * 2. / xka + 2. / zl) - pq2
            ! ahw: mods to compute ck, cd
            psiq10 = alog(vonkarman * ust(i) * 10. / xka + 10. / zl) - pq10
            ! Calculate 10 and 2 m diagnostics
            u10(i) = u_2d(i, kts)
            v10(i) = v_2d(i, kts)
            !q2(i) = qv_2d(i, kts)
            q2(i) = qsfc(i) + (qv_2d(i, kts) - qsfc(i)) * psiq2 / psiq
            fm(i) = psix
            fh(i) = psit
            !chs(i) = ust(i) * vonkarman / psiq
            !cqs2(i) = ust(i) * vonkarman / psiq2
            !chs2(i) = ust(i) * vonkarman / psit2
            ck(i) = (vonkarman / psix10) * (vonkarman / psiq10)
            cd(i) = (vonkarman / psix10) * (vonkarman / psix10)
            cka(i) = (vonkarman / psix) * (vonkarman / psiq)
            cda(i) = (vonkarman / psix) * (vonkarman / psix)
            pot_temp_diff = potential_temperature(kts) - skin_potential_temperature 
            qv_diff = qsfc(i) - qv_2d(i, kts)
            th2(i) = skin_potential_temperature + pot_temp_diff * psit2 / psit
            t2(i) = th2(i) * (psfc(i) / p_1000mb) ** r_over_cp
            chs(i) = abs(ust(i) * mol(i) / pot_temp_diff)
            cqs2(i) = ust(i) * qstar(i) / qv_diff
            chs2(i) = ust(i) * mol(i) / pot_temp_diff
            ! Calculate fluxes and exchange coefficients
            rho = psfc(i) / (r_dry * virtual_temperature)
            cpm = c_p * (1.0 + 0.8 * qv_2d(i, kts))
            if (abs(pot_temp_diff) > 1e-5) then
                flhc(i) = cpm * rho * ust(i) * mol(i) / pot_temp_diff
            else
                flhc(i) = 0
            endif
            hfx(i) = flhc(i) * (-pot_temp_diff)
            flqc(i) = rho * mavail(i) * ust(i) * qstar(i) / qv_diff 
            qfx(i) = flqc(i) * qv_diff
            lh(i) = x_lv * qfx(i)
            !if (i == 1) then
            !    print*, "chs", chs(i), "chs2", chs2(i), "ptd", pot_temp_diff
            !end if
        end do
    end subroutine sfclay_neuralnet_2d


    subroutine sfclayrev_init_table
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
        real, intent(in) :: ri, z, z0
        real :: zolri, fx1, fx2, x1, x2
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
        real, intent(in) :: zol2, ri2, z, z0
        real :: zol2b, zolri2, zol20, zol3, psix2, psih2
        !
        if (zol2 * ri2 < 0.) then
            zol2b = 0.
        else
            zol2b = zol2
        end if  ! limit zol2 - must be same sign as ri2
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
        zolri2 = zol2b * psih2 / psix2**2 - ri2
        !
        return
    end function
    !
    ! ... integrated similarity functions ...
    !
    function psim_stable_full(zolf)
        real, intent(in) :: zolf
        real :: psim_stable_full
        psim_stable_full = -6.1 * log(zolf + (1 + zolf**2.5)**(1. / 2.5))
        return
    end function

    function psih_stable_full(zolf)
        real, intent(in) :: zolf
        real :: psih_stable_full
        psih_stable_full = -5.3 * log(zolf + (1 + zolf**1.1)**(1. / 1.1))
        return
    end function

    function psim_unstable_full(zolf)
        real, intent(in) :: zolf
        real :: x, psimk, ym, psimc, psim_unstable_full
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
        real, intent(in) :: zolf
        real :: y, psihk, yh, psihc, psih_unstable_full
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
        real, intent(in) :: zolf
        integer :: nzol
        real :: rzol, psim_stable
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
        real, intent(in) :: zolf
        integer :: nzol
        real :: rzol, psih_stable
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
        real, intent(in) :: zolf
        integer :: nzol
        real :: rzol, psim_unstable
        nzol = int(abs(-zolf * 100.))
        rzol = -zolf * 100. - nzol
        if (nzol + 1 <= 1000) then
            psim_unstable = psim_unstab(nzol) + rzol * (psim_unstab(nzol + 1) - psim_unstab(nzol))
        else
            psim_unstable = psim_unstable_full(zolf)
        endif
        return
    end function

    function psih_unstable(zolf)
        real, intent(in) :: zolf
        integer :: nzol
        real :: rzol, psih_unstable
        nzol = int(abs(-zolf * 100.))
        rzol = -zolf * 100. - nzol
        if (nzol + 1 <= 1000) then
            psih_unstable = psih_unstab(nzol) + rzol * (psih_unstab(nzol + 1) - psih_unstab(nzol))
        else
            psih_unstable = psih_unstable_full(zolf)
        endif
        return
    end function


end module module_sf_sfclay_neural_net
