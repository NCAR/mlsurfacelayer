!WRF:MODEL_LAYER:PHYSICS
!
module module_sf_sfclayrev

    real, parameter :: vconvc = 1.
    real, parameter :: czo = 0.0185
    real, parameter :: ozo = 1.59e-5

    real, dimension(0:1000), save :: psim_stab, psim_unstab, psih_stab, psih_unstab

contains

    !-------------------------------------------------------------------
    subroutine sfclayrev(u3d, v3d, t3d, qv3d, p3d, dz8w, &
            cp, g, rovcp, r, xlv, psfc, chs, chs2, cqs2, cpm, &
            znt, ust, pblh, mavail, zol, mol, regime, psim, psih, &
            fm, fh, &
            xland, hfx, qfx, lh, tsk, flhc, flqc, qgh, qsfc, rmol, &
            u10, v10, th2, t2, q2, &
            gz1oz0, wspd, br, isfflx, dx, &
            svp1, svp2, svp3, svpt0, ep1, ep2, &
            karman, eomeg, stbolt, &
            p1000mb, &
            ids, ide, jds, jde, kds, kde, &
            ims, ime, jms, jme, kms, kme, &
            its, ite, jts, jte, kts, kte, &
            ustm, ck, cka, cd, cda, isftcflx, iz0tlnd, scm_force_flux)
        !-------------------------------------------------------------------
        implicit none
        !-------------------------------------------------------------------
        !   changes in v3.7 over water surfaces:
        !          1. for znt/cd, replacing constant ozo with 0.11*1.5e-5/ust(i)
        !             the coare 3.5 (edson et al. 2013) formulation is also available
        !          2. for vconv, reducing magnitude by half
        !          3. for ck, replacing carlson-boland with coare 3
        !-------------------------------------------------------------------
        !-- u3d         3d u-velocity interpolated to theta points (m/s)
        !-- v3d         3d v-velocity interpolated to theta points (m/s)
        !-- t3d         temperature (k)
        !-- qv3d        3d water vapor mixing ratio (kg/kg)
        !-- p3d         3d pressure (pa)
        !-- dz8w        dz between full levels (m)
        !-- cp          heat capacity at constant pressure for dry air (j/kg/k)
        !-- g           acceleration due to gravity (m/s^2)
        !-- rovcp       r/cp
        !-- r           gas constant for dry air (j/kg/k)
        !-- xlv         latent heat of vaporization for water (j/kg)
        !-- psfc        surface pressure (pa)
        !-- znt         roughness length (m)
        !-- ust         u* in similarity theory (m/s)
        !-- ustm        u* in similarity theory (m/s) without vconv correction
        !               used to couple with tke scheme
        !-- pblh        pbl height from previous time (m)
        !-- mavail      surface moisture availability (between 0 and 1)
        !-- zol         z/l height over monin-obukhov length
        !-- mol         t* (similarity theory) (k)
        !-- regime      flag indicating pbl regime (stable, unstable, etc.)
        !-- psim        similarity stability function for momentum
        !-- psih        similarity stability function for heat
        !-- fm          integrated stability function for momentum
        !-- fh          integrated stability function for heat
        !-- xland       land mask (1 for land, 2 for water)
        !-- hfx         upward heat flux at the surface (w/m^2)
        !-- qfx         upward moisture flux at the surface (kg/m^2/s)
        !-- lh          net upward latent heat flux at surface (w/m^2)
        !-- tsk         surface temperature (k)
        !-- flhc        exchange coefficient for heat (w/m^2/k)
        !-- flqc        exchange coefficient for moisture (kg/m^2/s)
        !-- chs         heat/moisture exchange coefficient for lsm (m/s)
        !-- qgh         lowest-level saturated mixing ratio
        !-- qsfc        ground saturated mixing ratio
        !-- u10         diagnostic 10m u wind
        !-- v10         diagnostic 10m v wind
        !-- th2         diagnostic 2m theta (k)
        !-- t2          diagnostic 2m temperature (k)
        !-- q2          diagnostic 2m mixing ratio (kg/kg)
        !-- gz1oz0      log(z/z0) where z0 is roughness length
        !-- wspd        wind speed at lowest model level (m/s)
        !-- br          bulk richardson number in surface layer
        !-- isfflx      isfflx=1 for surface heat and moisture fluxes
        !-- dx          horizontal grid size (m)
        !-- svp1        constant for saturation vapor pressure (kpa)
        !-- svp2        constant for saturation vapor pressure (dimensionless)
        !-- svp3        constant for saturation vapor pressure (k)
        !-- svpt0       constant for saturation vapor pressure (k)
        !-- ep1         constant for virtual temperature (r_v/r_d - 1) (dimensionless)
        !-- ep2         constant for specific humidity calculation
        !               (r_d/r_v) (dimensionless)
        !-- karman      von karman constant
        !-- eomeg       angular velocity of earth's rotation (rad/s)
        !-- stbolt      stefan-boltzmann constant (w/m^2/k^4)
        !-- ck          enthalpy exchange coeff at 10 meters
        !-- cd          momentum exchange coeff at 10 meters
        !-- cka         enthalpy exchange coeff at the lowest model level
        !-- cda         momentum exchange coeff at the lowest model level
        !-- isftcflx    =0, (charnock and carlson-boland); =1, ahw ck, cd, =2 garratt
        !-- iz0tlnd     =0 carlson-boland, =1 czil_new
        !-- ids         start index for i in domain
        !-- ide         end index for i in domain
        !-- jds         start index for j in domain
        !-- jde         end index for j in domain
        !-- kds         start index for k in domain
        !-- kde         end index for k in domain
        !-- ims         start index for i in memory
        !-- ime         end index for i in memory
        !-- jms         start index for j in memory
        !-- jme         end index for j in memory
        !-- kms         start index for k in memory
        !-- kme         end index for k in memory
        !-- its         start index for i in tile
        !-- ite         end index for i in tile
        !-- jts         start index for j in tile
        !-- jte         end index for j in tile
        !-- kts         start index for k in tile
        !-- kte         end index for k in tile
        !-------------------------------------------------------------------
        integer, intent(in) :: ids, ide, jds, jde, kds, kde, &
                ims, ime, jms, jme, kms, kme, &
                its, ite, jts, jte, kts, kte
        !
        integer, intent(in) :: isfflx
        real, intent(in) :: svp1, svp2, svp3, svpt0
        real, intent(in) :: ep1, ep2, karman, eomeg, stbolt
        real, intent(in) :: p1000mb
        !
        real, dimension(ims:ime, kms:kme, jms:jme), &
                intent(in) :: dz8w

        real, dimension(ims:ime, kms:kme, jms:jme), &
                intent(in) :: qv3d, &
                p3d, &
                t3d

        real, dimension(ims:ime, jms:jme), &
                intent(in) :: mavail, &
                pblh, &
                xland, &
                tsk
        real, dimension(ims:ime, jms:jme), &
                intent(out) :: u10, &
                v10, &
                th2, &
                t2, &
                q2, &
                qsfc

        !
        real, dimension(ims:ime, jms:jme), &
                intent(inout) :: regime, &
                hfx, &
                qfx, &
                lh, &
                mol, rmol
        !m the following 5 are change to memory size
        !
        real, dimension(ims:ime, jms:jme), &
                intent(inout) :: gz1oz0, wspd, br, &
                psim, psih, fm, fh

        real, dimension(ims:ime, kms:kme, jms:jme), &
                intent(in) :: u3d, &
                v3d

        real, dimension(ims:ime, jms:jme), &
                intent(in) :: psfc

        real, dimension(ims:ime, jms:jme), &
                intent(inout) :: znt, &
                zol, &
                ust, &
                cpm, &
                chs2, &
                cqs2, &
                chs

        real, dimension(ims:ime, jms:jme), &
                intent(inout) :: flhc, flqc

        real, dimension(ims:ime, jms:jme), &
                intent(inout) :: &
                qgh

        real, intent(in) :: cp, g, rovcp, r, xlv, dx

        real, optional, dimension(ims:ime, jms:jme), &
                intent(out) :: ck, cka, cd, cda

        real, optional, dimension(ims:ime, jms:jme), &
                intent(inout) :: ustm

        integer, optional, intent(in) :: isftcflx, iz0tlnd
        integer, optional, intent(in) :: scm_force_flux
        ! local vars

        real, dimension(its:ite) :: u1d, &
                v1d, &
                qv1d, &
                p1d, &
                t1d

        real, dimension(its:ite) :: dz8w1d

        integer :: i, j

        do j = jts, jte
            do i = its, ite
                dz8w1d(i) = dz8w(i, 1, j)
            enddo

            do i = its, ite
                u1d(i) = u3d(i, 1, j)
                v1d(i) = v3d(i, 1, j)
                qv1d(i) = qv3d(i, 1, j)
                p1d(i) = p3d(i, 1, j)
                t1d(i) = t3d(i, 1, j)
            enddo

            !  sending array starting locations of optional variables may cause
            !  troubles, so we explicitly change the call.

            call sfclayrev1d(j, u1d, v1d, t1d, qv1d, p1d, dz8w1d, &
                    cp, g, rovcp, r, xlv, psfc(ims, j), chs(ims, j), chs2(ims, j), &
                    cqs2(ims, j), cpm(ims, j), pblh(ims, j), rmol(ims, j), &
                    znt(ims, j), ust(ims, j), mavail(ims, j), zol(ims, j), &
                    mol(ims, j), regime(ims, j), psim(ims, j), psih(ims, j), &
                    fm(ims, j), fh(ims, j), &
                    xland(ims, j), hfx(ims, j), qfx(ims, j), tsk(ims, j), &
                    u10(ims, j), v10(ims, j), th2(ims, j), t2(ims, j), &
                    q2(ims, j), flhc(ims, j), flqc(ims, j), qgh(ims, j), &
                    qsfc(ims, j), lh(ims, j), &
                    gz1oz0(ims, j), wspd(ims, j), br(ims, j), isfflx, dx, &
                    svp1, svp2, svp3, svpt0, ep1, ep2, karman, eomeg, stbolt, &
                    p1000mb, &
                    ids, ide, jds, jde, kds, kde, &
                    ims, ime, jms, jme, kms, kme, &
                    its, ite, jts, jte, kts, kte                          &
                    #if ( em_core == 1 )
, isftcflx, iz0tlnd, scm_force_flux, &
                    ustm(ims, j), ck(ims, j), cka(ims, j), &
                    cd(ims, j), cda(ims, j)                               &
                    #endif
)
        enddo

    end subroutine sfclayrev


    !-------------------------------------------------------------------
    subroutine sfclayrev1d(j, ux, vx, t1d, qv1d, p1d, dz8w1d, &
            cp, g, rovcp, r, xlv, psfcpa, chs, chs2, cqs2, cpm, pblh, rmol, &
            znt, ust, mavail, zol, mol, regime, psim, psih, fm, fh, &
            xland, hfx, qfx, tsk, &
            u10, v10, th2, t2, q2, flhc, flqc, qgh, &
            qsfc, lh, gz1oz0, wspd, br, isfflx, dx, &
            svp1, svp2, svp3, svpt0, ep1, ep2, &
            karman, eomeg, stbolt, &
            p1000mb, &
            ids, ide, jds, jde, kds, kde, &
            ims, ime, jms, jme, kms, kme, &
            its, ite, jts, jte, kts, kte, &
            isftcflx, iz0tlnd, scm_force_flux, &
            ustm, ck, cka, cd, cda)
        !-------------------------------------------------------------------
        implicit none
        !-------------------------------------------------------------------
        !-------------------------------------------------------------------
        !   changes in v3.7 over water surfaces:
        !          1. for znt/cd, replacing constant ozo with 0.11*1.5e-5/ust(i)
        !             the coare 3.5 (edson et al. 2013) formulation is also available
        !          2. for vconv, reducing magnitude by half
        !          3. for ck, replacing carlson-boland with coare 3
        !-------------------------------------------------------------------
        !-- ux          1d u-velocity interpolated to theta points (m/s)
        !-- vx         1d v-velocity interpolated to theta points (m/s)
        !-- t1d         temperature (k)
        !-- qv1d        3d water vapor mixing ratio (kg/kg)
        !-- p1d         3d pressure (pa)
        !-- dz8w1d        dz between full levels (m)
        !-- cp          heat capacity at constant pressure for dry air (j/kg/k)
        !-- g           acceleration due to gravity (m/s^2)
        !-- rovcp       r/cp
        !-- r           gas constant for dry air (j/kg/k)
        !-- xlv         latent heat of vaporization for water (j/kg)
        !-- psfcpa        surface pressure (pa)
        !-- chs         heat/moisture exchange coefficient for lsm (m/s)
        !-- chs2        ???
        !-- cqs2          ???
        !-- cpm         heat capacity at a constant pressure for moist air cpm(i) = cp * (1. + 0.8 * qx(i))
        !-- pblh        pbl height from previous time (m)
        !-- rmol        1 / monin obukhov length
        !-- znt         z naught, zo, roughness length (m)
        !-- ust         u* in similarity theory (m/s)
        !-- ustm        u* in similarity theory (m/s) without vconv correction
        !               used to couple with tke scheme
        !-- mavail      surface moisture availability (between 0 and 1)
        !-- zol         z/l height over monin-obukhov length
        !-- mol         t* (similarity theory) (k) (Actually this might be monin-obukhov length but am not sure)
        !-- regime      flag indicating pbl regime (stable, unstable, etc.)
        !-- psim        similarity stability function for momentum
        !-- psih        similarity stability function for heat
        !-- fm          integrated stability function for momentum
        !-- fh          integrated stability function for heat
        !-- xland       land mask (1 for land, 2 for water)
        !-- hfx         upward heat flux at the surface (w/m^2)
        !-- qfx         upward moisture flux at the surface (kg/m^2/s)
        !-- lh          net upward latent heat flux at surface (w/m^2)
        !-- tsk         surface temperature (k)
        !-- flhc        exchange coefficient for heat (w/m^2/k)
        !-- flqc        exchange coefficient for moisture (kg/m^2/s)
        !-- qgh         lowest-level saturated mixing ratio
        !-- qsfc        ground saturated mixing ratio
        !-- u10         diagnostic 10m u wind
        !-- v10         diagnostic 10m v wind
        !-- th2         diagnostic 2m theta (k)
        !-- t2          diagnostic 2m temperature (k)
        !-- q2          diagnostic 2m mixing ratio (kg/kg)
        !-- gz1oz0      log(z/z0) where z0 is roughness length
        !-- wspd        wind speed at lowest model level (m/s)
        !-- br          bulk richardson number in surface layer
        !-- isfflx      isfflx=1 for surface heat and moisture fluxes
        !-- dx          horizontal grid size (m)
        !-- svp1        constant for saturation vapor pressure (kpa)
        !-- svp2        constant for saturation vapor pressure (dimensionless)
        !-- svp3        constant for saturation vapor pressure (k)
        !-- svpt0       constant for saturation vapor pressure (k)
        !-- ep1         constant for virtual temperature (r_v/r_d - 1) (dimensionless)
        !-- ep2         constant for specific humidity calculation
        !               (r_d/r_v) (dimensionless)
        !-- karman      von karman constant
        !-- eomeg       angular velocity of earth's rotation (rad/s)
        !-- stbolt      stefan-boltzmann constant (w/m^2/k^4)
        !-- ck          enthalpy exchange coeff at 10 meters
        !-- cd          momentum exchange coeff at 10 meters
        !-- cka         enthalpy exchange coeff at the lowest model level
        !-- cda         momentum exchange coeff at the lowest model level
        !-- isftcflx    =0, (charnock and carlson-boland); =1, ahw ck, cd, =2 garratt
        !-- iz0tlnd     =0 carlson-boland, =1 czil_new
        !-- ids         start index for i in domain
        !-- ide         end index for i in domain
        !-- jds         start index for j in domain
        !-- jde         end index for j in domain
        !-- kds         start index for k in domain
        !-- kde         end index for k in domain
        !-- ims         start index for i in memory
        !-- ime         end index for i in memory
        !-- jms         start index for j in memory
        !-- jme         end index for j in memory
        !-- kms         start index for k in memory
        !-- kme         end index for k in memory
        !-- its         start index for i in tile
        !-- ite         end index for i in tile
        !-- jts         start index for j in tile
        !-- jte         end index for j in tile
        !-- kts         start index for k in tile
        !-- kte         end index for k in tile
        !-------------------------------------------------------------------
        real, parameter :: xka = 2.4e-5
        real, parameter :: prt = 1.

        integer, intent(in) :: ids, ide, jds, jde, kds, kde, &
                ims, ime, jms, jme, kms, kme, &
                its, ite, jts, jte, kts, kte, &
                j
        !
        integer, intent(in) :: isfflx
        real, intent(in) :: svp1, svp2, svp3, svpt0
        real, intent(in) :: ep1, ep2, karman, eomeg, stbolt
        real, intent(in) :: p1000mb

        !
        real, dimension(ims:ime), &
                intent(in) :: mavail, &
                pblh, &
                xland, &
                tsk
        !
        real, dimension(ims:ime), &
                intent(in) :: psfcpa

        real, dimension(ims:ime), &
                intent(inout) :: regime, &
                hfx, &
                qfx, &
                mol, rmol
        !m the following 5 are changed to memory size---
        !
        real, dimension(ims:ime), &
                intent(inout) :: gz1oz0, wspd, br, &
                psim, psih, fm, fh

        real, dimension(ims:ime), &
                intent(inout) :: znt, &
                zol, &
                ust, &
                cpm, &
                chs2, &
                cqs2, &
                chs

        real, dimension(ims:ime), &
                intent(inout) :: flhc, flqc

        real, dimension(ims:ime), &
                intent(inout) :: &
                qgh

        real, dimension(ims:ime), &
                intent(out) :: u10, v10, &
                th2, t2, q2, qsfc, lh

        real, intent(in) :: cp, g, rovcp, r, xlv, dx

        ! module-local variables, defined in subroutine sfclay
        real, dimension(its:ite), intent(in) :: dz8w1d

        real, dimension(its:ite), intent(in) :: ux, &
                vx, &
                qv1d, &
                p1d, &
                t1d

        real, optional, dimension(ims:ime), &
                intent(out) :: ck, cka, cd, cda
        real, optional, dimension(ims:ime), &
                intent(inout) :: ustm

        integer, optional, intent(in) :: isftcflx, iz0tlnd
        integer, optional, intent(in) :: scm_force_flux

        ! local vars

        real, dimension(its:ite) :: za, &
                thvx, zqkl, &
                zqklp1, &
                thx, qx, &
                psih2, &
                psim2, &
                psih10, &
                psim10, &
                denomq, &
                denomq2, &
                denomt2, &
                wspdi, &
                gz2oz0, &
                gz10oz0
        !
        real, dimension(its:ite) :: &
                rhox, govrth, &
                tgdsa
        !
        real, dimension(its:ite) :: scr3, scr4
        real, dimension(its:ite) :: thgb, psfc
        !
        integer :: kl

        integer :: n, i, k, kk, l, nzol, nk, nzol2, nzol10

        real :: pl, thcon, tvcon, e1
        real :: zl, tskv, dthvdz, dthvm, vconv, rzol, rzol2, rzol10, zol2, zol10
        real :: dtg, psix, dtthx, psix10, psit, psit2, psiq, psiq2, psiq10
        real :: fluxc, vsgd, z0q, visc, restar, czil, gz0ozq, gz0ozt
        real :: zw, zn1, zn2
        !
        ! .... paj ...
        !
        real :: zolzz, zol0
        !     real    :: zolri,zolri2
        !     real    :: psih_stable,psim_stable,psih_unstable,psim_unstable
        !     real    :: psih_stable_full,psim_stable_full,psih_unstable_full,psim_unstable_full
        real :: zl2, zl10, z0t
        real, dimension(its:ite) :: pq, pq2, pq10
        ! thgb      potential skin temperature (K)
        ! tskv
        !-------------------------------------------------------------------
        kl = kte

        do i = its, ite
            ! psfc cb
            psfc(i) = psfcpa(i) / 1000.
        enddo
        !
        !----convert ground temperature to potential temperature:
        !
        do 5 i = its, ite
            tgdsa(i) = tsk(i)
            ! psfc cb
            !        thgb(i)=tsk(i)*(100./psfc(i))**rovcp
            thgb(i) = tsk(i) * (p1000mb / psfcpa(i))**rovcp
        5 continue
        !
        !-----decouple flux-form variables to give u,v,t,theta,theta-vir.,
        !     t-vir., qv, and qc at cross points and at ktau-1.
        !
        !     *** note ***
        !         the boundary winds may not be adequately affected by friction,
        !         so use only interior values of ux and vx to calculate
        !         tendencies.
        !
        10 continue

        !     do 24 i=its,ite
        !        ux(i)=u1d(i)
        !        vx(i)=v1d(i)
        !  24 continue

        26 continue

        !.....scr3(i,k) store temperature,
        !     scr4(i,k) store virtual temperature.

        do 30 i = its, ite
            ! pl cb
            pl = p1d(i) / 1000.
            scr3(i) = t1d(i)
            !         thcon=(100./pl)**rovcp
            thcon = (p1000mb * 0.001 / pl)**rovcp
            thx(i) = scr3(i) * thcon
            scr4(i) = scr3(i)
            thvx(i) = thx(i)
            qx(i) = 0.
        30 continue
        !
        do i = its, ite
            qgh(i) = 0.
            flhc(i) = 0.
            flqc(i) = 0.
            cpm(i) = cp
        enddo
        !
        !     if(idry.eq.1)goto 80
        do 50 i = its, ite
            qx(i) = qv1d(i)
            tvcon = (1. + ep1 * qx(i))
            thvx(i) = thx(i) * tvcon
            scr4(i) = scr3(i) * tvcon
        50 continue
        !
        do 60 i = its, ite
            e1 = svp1 * exp(svp2 * (tgdsa(i) - svpt0) / (tgdsa(i) - svp3))
            !  for land points qsfc can come from previous time step
            if(xland(i).gt.1.5.or.qsfc(i).le.0.0)qsfc(i) = ep2 * e1 / (psfc(i) - e1)
            ! qgh changed to use lowest-level air temp consistent with myjsfc change
            ! q2sat = qgh in lsm
            e1 = svp1 * exp(svp2 * (t1d(i) - svpt0) / (t1d(i) - svp3))
            pl = p1d(i) / 1000.
            qgh(i) = ep2 * e1 / (pl - e1)
            cpm(i) = cp * (1. + 0.8 * qx(i))
        60 continue
        80 continue

        !-----compute the height of full- and half-sigma levels above ground
        !     level, and the layer thicknesses.

        do 90 i = its, ite
            zqklp1(i) = 0.
            rhox(i) = psfc(i) * 1000. / (r * scr4(i))
        90 continue
        !
        do 110 i = its, ite
            zqkl(i) = dz8w1d(i) + zqklp1(i)
        110 continue
        !
        do 120 i = its, ite
            za(i) = 0.5 * (zqkl(i) + zqklp1(i))
        120 continue
        !
        do 160 i = its, ite
            govrth(i) = g / thx(i)
        160 continue

        !-----calculate bulk richardson no. of surface layer, according to
        !     akb(1976), eq(12).

        do 260 i = its, ite
            gz1oz0(i) = alog((za(i) + znt(i)) / znt(i))   ! log((z+z0)/z0)
            gz2oz0(i) = alog((2. + znt(i)) / znt(i))      ! log((2+z0)/z0)
            gz10oz0(i) = alog((10. + znt(i)) / znt(i))    ! log((10+z0)z0)
            if((xland(i) - 1.5).ge.0)then
                zl = znt(i)
            else
                zl = 0.01
            endif
            wspd(i) = sqrt(ux(i) * ux(i) + vx(i) * vx(i))

            tskv = thgb(i) * (1. + ep1 * qsfc(i))
            dthvdz = (thvx(i) - tskv)
            !  convective velocity scale vc and subgrid-scale velocity vsg
            !  following beljaars (1994, qjrms) and mahrt and sun (1995, mwr)
            !                                ... hong aug. 2001
            !
            !       vconv = 0.25*sqrt(g/tskv*pblh(i)*dthvm)
            !      use beljaars over land, old mm5 (wyngaard) formula over water
            if (xland(i).lt.1.5) then
                fluxc = max(hfx(i) / rhox(i) / cp                    &
                        + ep1 * tskv * qfx(i) / rhox(i), 0.)
                vconv = vconvc * (g / tgdsa(i) * pblh(i) * fluxc)**.33
            else
                if(-dthvdz.ge.0)then
                    dthvm = -dthvdz
                else
                    dthvm = 0.
                endif
                !       vconv = 2.*sqrt(dthvm)
                ! v3.7: reducing contribution in calm conditions
                vconv = sqrt(dthvm)
            endif
            ! mahrt and sun low-res correction
            vsgd = 0.32 * (max(dx / 5000. - 1., 0.))**.33
            wspd(i) = sqrt(wspd(i) * wspd(i) + vconv * vconv + vsgd * vsgd)
            wspd(i) = amax1(wspd(i), 0.1)
            br(i) = govrth(i) * za(i) * dthvdz / (wspd(i) * wspd(i))
            !  if previously unstable, do not let into regimes 1 and 2
            if(mol(i).lt.0.)br(i) = amin1(br(i), 0.0)
            !jdf
            rmol(i) = -govrth(i) * dthvdz * za(i) * karman
            !jdf

        260 continue

        !
        !-----diagnose basic parameters for the appropriated stability class:
        !
        !
        !     the stability classes are determined by br (bulk richardson no.)
        !     and hol (height of pbl/monin-obukhov length).
        !
        !     criteria for the classes are as follows:
        !
        !        1. br .ge. 0.0;
        !               represents nighttime stable conditions (regime=1),
        !
        !        3. br .eq. 0.0
        !               represents forced convection conditions (regime=3),
        !
        !        4. br .lt. 0.0
        !               represents free convection conditions (regime=4).
        !
        !ccccc

        do 320 i = its, ite
            !
            if (br(i).gt.0) then
                if (br(i).gt.250.0) then
                    zol(i) = zolri(250.0, za(i), znt(i))
                else
                    zol(i) = zolri(br(i), za(i), znt(i))
                endif
            endif
            !
            if (br(i).lt.0) then
                if(ust(i).lt.0.001)then
                    zol(i) = br(i) * gz1oz0(i)
                else
                    if (br(i).lt.-250.0) then
                        zol(i) = zolri(-250.0, za(i), znt(i))
                    else
                        zol(i) = zolri(br(i), za(i), znt(i))
                    endif
                endif
            endif
            !
            ! ... paj: compute integrated similarity functions.
            !
            zolzz = zol(i) * (za(i) + znt(i)) / za(i) ! (z+z0/l
            zol10 = zol(i) * (10. + znt(i)) / za(i)   ! (10+z0)/l
            zol2 = zol(i) * (2. + znt(i)) / za(i)     ! (2+z0)/l
            zol0 = zol(i) * znt(i) / za(i)          ! z0/l
            zl2 = (2.) / za(i) * zol(i)             ! 2/l
            zl10 = (10.) / za(i) * zol(i)           ! 10/l

            if((xland(i) - 1.5).lt.0.)then
                zl = (0.01) / za(i) * zol(i)   ! (0.01)/l
            else
                zl = zol0                     ! z0/l
            endif

            if(br(i).lt.0.)goto 310  ! go to unstable regime (class 4)
            if(br(i).eq.0.)goto 280  ! go to neutral regime (class 3)
            !
            !-----class 1; stable (nighttime) conditions:
            !
            regime(i) = 1.
            !
            ! ... paj: psim and psih. follows cheng and brutsaert 2005 (cb05).
            !
            psim(i) = psim_stable(zolzz) - psim_stable(zol0)
            psih(i) = psih_stable(zolzz) - psih_stable(zol0)
            !
            psim10(i) = psim_stable(zol10) - psim_stable(zol0)
            psih10(i) = psih_stable(zol10) - psih_stable(zol0)
            !
            psim2(i) = psim_stable(zol2) - psim_stable(zol0)
            psih2(i) = psih_stable(zol2) - psih_stable(zol0)
            !
            ! ... paj: preparations to compute psiq. follows cb05+carlson boland jam 1978.
            !
            pq(i) = psih_stable(zol(i)) - psih_stable(zl)
            pq2(i) = psih_stable(zl2) - psih_stable(zl)
            pq10(i) = psih_stable(zl10) - psih_stable(zl)
            !
            !       1.0 over monin-obukhov length
            rmol(i) = zol(i) / za(i)
            !

            goto 320
            !
            !-----class 3; forced convection:
            !
            280   regime(i) = 3.
            psim(i) = 0.0
            psih(i) = psim(i)
            psim10(i) = 0.
            psih10(i) = psim10(i)
            psim2(i) = 0.
            psih2(i) = psim2(i)
            !
            ! paj: preparations to compute psiq.
            !
            pq(i) = psih(i)
            pq2(i) = psih2(i)
            pq10(i) = 0.
            !
            zol(i) = 0.
            rmol(i) = zol(i) / za(i)

            goto 320
            !
            !-----class 4; free convection:
            !
            310   continue
            regime(i) = 4.
            !
            ! ... paj: psim and psih ...
            !
            psim(i) = psim_unstable(zolzz) - psim_unstable(zol0)
            psih(i) = psih_unstable(zolzz) - psih_unstable(zol0)
            !
            psim10(i) = psim_unstable(zol10) - psim_unstable(zol0)
            psih10(i) = psih_unstable(zol10) - psih_unstable(zol0)
            !
            psim2(i) = psim_unstable(zol2) - psim_unstable(zol0)
            psih2(i) = psih_unstable(zol2) - psih_unstable(zol0)
            !
            ! ... paj: preparations to compute psiq
            !
            pq(i) = psih_unstable(zol(i)) - psih_unstable(zl)
            pq2(i) = psih_unstable(zl2) - psih_unstable(zl)
            pq10(i) = psih_unstable(zl10) - psih_unstable(zl)
            !
            !---limiot psih and psim in the case of thin layers and high roughness
            !---  this prevents denominator in fluxes from getting too small
            psih(i) = amin1(psih(i), 0.9 * gz1oz0(i))
            psim(i) = amin1(psim(i), 0.9 * gz1oz0(i))
            psih2(i) = amin1(psih2(i), 0.9 * gz2oz0(i))
            psim10(i) = amin1(psim10(i), 0.9 * gz10oz0(i))
            !
            ! ahw: mods to compute ck, cd
            psih10(i) = amin1(psih10(i), 0.9 * gz10oz0(i))

            rmol(i) = zol(i) / za(i)

        320 continue
        !
        !-----compute the frictional velocity:
        !     za(1982) eqs(2.60),(2.61).
        !
        do 330 i = its, ite
            dtg = thx(i) - thgb(i)
            psix = gz1oz0(i) - psim(i)
            psix10 = gz10oz0(i) - psim10(i)

            !     lower limit added to prevent large flhc in soil model
            !     activates in unstable conditions with thin layers or high z0
            !       psit=amax1(gz1oz0(i)-psih(i),2.)
            psit = gz1oz0(i) - psih(i)
            psit2 = gz2oz0(i) - psih2(i)
            !
            if((xland(i) - 1.5).ge.0)then
                zl = znt(i)
            else
                zl = 0.01
            endif
            !
            psiq = alog(karman * ust(i) * za(i) / xka + za(i) / zl) - pq(i)
            psiq2 = alog(karman * ust(i) * 2. / xka + 2. / zl) - pq2(i)

            ! ahw: mods to compute ck, cd
            psiq10 = alog(karman * ust(i) * 10. / xka + 10. / zl) - pq10(i)

            ! v3.7: using fairall 2003 to compute z0q and z0t over water:
            !       adapted from module_sf_mynn.f
            if ((xland(i) - 1.5).ge.0.) then
                visc = (1.32 + 0.009 * (scr3(i) - 273.15)) * 1.e-5
                restar = ust(i) * znt(i) / visc
                z0t = (5.5e-5) * (restar**(-0.60))
                z0t = min(z0t, 1.0e-4)
                z0t = max(z0t, 2.0e-9)
                z0q = z0t

                psiq = max(alog((za(i) + z0q) / z0q) - psih(i), 2.)
                psit = max(alog((za(i) + z0t) / z0t) - psih(i), 2.)
                psiq2 = max(alog((2. + z0q) / z0q) - psih2(i), 2.)
                psit2 = max(alog((2. + z0t) / z0t) - psih2(i), 2.)
                psiq10 = max(alog((10. + z0q) / z0q) - psih10(i), 2.)
            endif

            if (present(isftcflx)) then
                if (isftcflx.eq.1 .and. (xland(i) - 1.5).ge.0.) then
                    ! v3.1
                    !             z0q = 1.e-4 + 1.e-3*(max(0.,ust(i)-1.))**2
                    ! hfip1
                    !             z0q = 0.62*2.0e-5/ust(i) + 1.e-3*(max(0.,ust(i)-1.5))**2
                    ! v3.2
                    z0q = 1.e-4
                    !
                    ! ... paj: recompute psih for z0q
                    !
                    zolzz = zol(i) * (za(i) + z0q) / za(i)    ! (z+z0q)/l
                    zol10 = zol(i) * (10. + z0q) / za(i)   ! (10+z0q)/l
                    zol2 = zol(i) * (2. + z0q) / za(i)     ! (2+z0q)/l
                    zol0 = zol(i) * z0q / za(i)          ! z0q/l
                    !
                    if (zol(i).gt.0.) then
                        psih(i) = psih_stable(zolzz) - psih_stable(zol0)
                        psih10(i) = psih_stable(zol10) - psih_stable(zol0)
                        psih2(i) = psih_stable(zol2) - psih_stable(zol0)
                    else
                        if (zol(i).eq.0) then
                            psih(i) = 0.
                            psih10(i) = 0.
                            psih2(i) = 0.
                        else
                            psih(i) = psih_unstable(zolzz) - psih_unstable(zol0)
                            psih10(i) = psih_unstable(zol10) - psih_unstable(zol0)
                            psih2(i) = psih_unstable(zol2) - psih_unstable(zol0)
                        endif
                    endif
                    !
                    psiq = alog((za(i) + z0q) / z0q) - psih(i)
                    psit = psiq
                    psiq2 = alog((2. + z0q) / z0q) - psih2(i)
                    psiq10 = alog((10. + z0q) / z0q) - psih10(i)
                    psit2 = psiq2
                endif
                if (isftcflx.eq.2 .and. (xland(i) - 1.5).ge.0.) then
                    ! ahw: garratt formula: calculate roughness reynolds number
                    !        kinematic viscosity of air (linear approc to
                    !                 temp dependence at sea level)
                    ! gz0ozt and gz0ozq are based off formulas from brutsaert (1975), which
                    ! garratt (1992) used with values of k = 0.40, pr = 0.71, and sc = 0.60
                    visc = (1.32 + 0.009 * (scr3(i) - 273.15)) * 1.e-5
                    !!            visc=1.5e-5
                    restar = ust(i) * znt(i) / visc
                    gz0ozt = 0.40 * (7.3 * sqrt(sqrt(restar)) * sqrt(0.71) - 5.)
                    !
                    ! ... paj: compute psih for z0t for temperature ...
                    !
                    z0t = znt(i) / exp(gz0ozt)
                    !
                    zolzz = zol(i) * (za(i) + z0t) / za(i)    ! (z+z0t)/l
                    zol10 = zol(i) * (10. + z0t) / za(i)   ! (10+z0t)/l
                    zol2 = zol(i) * (2. + z0t) / za(i)     ! (2+z0t)/l
                    zol0 = zol(i) * z0t / za(i)          ! z0t/l
                    !
                    if (zol(i).gt.0.) then
                        psih(i) = psih_stable(zolzz) - psih_stable(zol0)
                        psih10(i) = psih_stable(zol10) - psih_stable(zol0)
                        psih2(i) = psih_stable(zol2) - psih_stable(zol0)
                    else
                        if (zol(i).eq.0) then
                            psih(i) = 0.
                            psih10(i) = 0.
                            psih2(i) = 0.
                        else
                            psih(i) = psih_unstable(zolzz) - psih_unstable(zol0)
                            psih10(i) = psih_unstable(zol10) - psih_unstable(zol0)
                            psih2(i) = psih_unstable(zol2) - psih_unstable(zol0)
                        endif
                    endif
                    !
                    !              psit=gz1oz0(i)-psih(i)+restar2
                    !              psit2=gz2oz0(i)-psih2(i)+restar2
                    psit = alog((za(i) + z0t) / z0t) - psih(i)
                    psit2 = alog((2. + z0t) / z0t) - psih2(i)
                    !
                    gz0ozq = 0.40 * (7.3 * sqrt(sqrt(restar)) * sqrt(0.60) - 5.)
                    z0q = znt(i) / exp(gz0ozq)
                    !
                    zolzz = zol(i) * (za(i) + z0q) / za(i)    ! (z+z0q)/l
                    zol10 = zol(i) * (10. + z0q) / za(i)   ! (10+z0q)/l
                    zol2 = zol(i) * (2. + z0q) / za(i)     ! (2+z0q)/l
                    zol0 = zol(i) * z0q / za(i)          ! z0q/l
                    !
                    if (zol(i).gt.0.) then
                        psih(i) = psih_stable(zolzz) - psih_stable(zol0)
                        psih10(i) = psih_stable(zol10) - psih_stable(zol0)
                        psih2(i) = psih_stable(zol2) - psih_stable(zol0)
                    else
                        if (zol(i).eq.0) then
                            psih(i) = 0.
                            psih10(i) = 0.
                            psih2(i) = 0.
                        else
                            psih(i) = psih_unstable(zolzz) - psih_unstable(zol0)
                            psih10(i) = psih_unstable(zol10) - psih_unstable(zol0)
                            psih2(i) = psih_unstable(zol2) - psih_unstable(zol0)
                        endif
                    endif
                    !
                    psiq = alog((za(i) + z0q) / z0q) - psih(i)
                    psiq2 = alog((2. + z0q) / z0q) - psih2(i)
                    psiq10 = alog((10. + z0q) / z0q) - psih10(i)
                    !              psiq=gz1oz0(i)-psih(i)+2.28*sqrt(sqrt(restar))-2.
                    !              psiq2=gz2oz0(i)-psih2(i)+2.28*sqrt(sqrt(restar))-2.
                    !              psiq10=gz10oz0(i)-psih(i)+2.28*sqrt(sqrt(restar))-2.
                endif
            endif
            if(present(ck) .and. present(cd) .and. present(cka) .and. present(cda)) then
                ck(i) = (karman / psix10) * (karman / psiq10)
                cd(i) = (karman / psix10) * (karman / psix10)
                cka(i) = (karman / psix) * (karman / psiq)
                cda(i) = (karman / psix) * (karman / psix)
            endif
            if (present(iz0tlnd)) then
                if (iz0tlnd.eq.1 .and. (xland(i) - 1.5).le.0.) then
                    zl = znt(i)
                    !             czil related changes for land
                    visc = (1.32 + 0.009 * (scr3(i) - 273.15)) * 1.e-5
                    restar = ust(i) * zl / visc
                    !             modify czil according to chen & zhang, 2009

                    czil = 10.0 ** (-0.40 * (zl / 0.07))
                    !
                    ! ... paj: compute phish for z0t over land
                    !
                    z0t = znt(i) / exp(czil * karman * sqrt(restar))
                    !
                    zolzz = zol(i) * (za(i) + z0t) / za(i)    ! (z+z0t)/l
                    zol10 = zol(i) * (10. + z0t) / za(i)   ! (10+z0t)/l
                    zol2 = zol(i) * (2. + z0t) / za(i)     ! (2+z0t)/l
                    zol0 = zol(i) * z0t / za(i)          ! z0t/l
                    !
                    if (zol(i).gt.0.) then
                        psih(i) = psih_stable(zolzz) - psih_stable(zol0)
                        psih10(i) = psih_stable(zol10) - psih_stable(zol0)
                        psih2(i) = psih_stable(zol2) - psih_stable(zol0)
                    else
                        if (zol(i).eq.0) then
                            psih(i) = 0.
                            psih10(i) = 0.
                            psih2(i) = 0.
                        else
                            psih(i) = psih_unstable(zolzz) - psih_unstable(zol0)
                            psih10(i) = psih_unstable(zol10) - psih_unstable(zol0)
                            psih2(i) = psih_unstable(zol2) - psih_unstable(zol0)
                        endif
                    endif
                    !
                    psiq = alog((za(i) + z0t) / z0t) - psih(i)
                    psiq2 = alog((2. + z0t) / z0t) - psih2(i)
                    psit = psiq
                    psit2 = psiq2
                    !
                    !              psit=gz1oz0(i)-psih(i)+czil*karman*sqrt(restar)
                    !              psiq=gz1oz0(i)-psih(i)+czil*karman*sqrt(restar)
                    !              psit2=gz2oz0(i)-psih2(i)+czil*karman*sqrt(restar)
                    !              psiq2=gz2oz0(i)-psih2(i)+czil*karman*sqrt(restar)

                endif
            endif
            ! to prevent oscillations average with old value
            ust(i) = 0.5 * ust(i) + 0.5 * karman * wspd(i) / psix
            ! tke coupling: compute ust without vconv for use in tke scheme
            wspdi(i) = sqrt(ux(i) * ux(i) + vx(i) * vx(i))
            if (present(ustm)) then
                ustm(i) = 0.5 * ustm(i) + 0.5 * karman * wspdi(i) / psix
            endif

            u10(i) = ux(i) * psix10 / psix
            v10(i) = vx(i) * psix10 / psix
            th2(i) = thgb(i) + dtg * psit2 / psit
            q2(i) = qsfc(i) + (qx(i) - qsfc(i)) * psiq2 / psiq
            t2(i) = th2(i) * (psfcpa(i) / p1000mb)**rovcp
            !
            if((xland(i) - 1.5).lt.0.)then
                ust(i) = amax1(ust(i), 0.001)
            endif
            mol(i) = karman * dtg / psit / prt
            denomq(i) = psiq
            denomq2(i) = psiq2
            denomt2(i) = psit2
            fm(i) = psix
            fh(i) = psit
        330 continue
        !
        335 continue

        !-----compute the surface sensible and latent heat fluxes:
        if (present(scm_force_flux)) then
            if (scm_force_flux.eq.1) goto 350
        endif
        do i = its, ite
            qfx(i) = 0.
            hfx(i) = 0.
        enddo
        350 continue

        if (isfflx.eq.0) goto 410

        !-----over water, alter roughness length (znt) according to wind (ust).

        do 360 i = its, ite
            if((xland(i) - 1.5).ge.0)then
                !         znt(i)=czo*ust(i)*ust(i)/g+ozo
                ! since v3.7 (ref: ec physics document for cy36r1)
                znt(i) = czo * ust(i) * ust(i) / g + 0.11 * 1.5e-5 / ust(i)
                ! v3.9: add limit as in isftcflx = 1,2
                znt(i) = min(znt(i), 2.85e-3)
                ! coare 3.5 (edson et al. 2013)
                !         czc = 0.0017*wspd(i)-0.005
                !         czc = min(czc,0.028)
                !         znt(i)=czc*ust(i)*ust(i)/g+0.11*1.5e-5/ust(i)
                ! ahw: change roughness length, and hence the drag coefficients ck and cd
                if (present(isftcflx)) then
                    if (isftcflx.ne.0) then
                        !               znt(i)=10.*exp(-9.*ust(i)**(-.3333))
                        !               znt(i)=10.*exp(-9.5*ust(i)**(-.3333))
                        !               znt(i)=znt(i) + 0.11*1.5e-5/amax1(ust(i),0.01)
                        !               znt(i)=0.011*ust(i)*ust(i)/g+ozo
                        !               znt(i)=max(znt(i),3.50e-5)
                        ! ahw 2012:
                        zw = min((ust(i) / 1.06)**(0.3), 1.0)
                        zn1 = 0.011 * ust(i) * ust(i) / g + ozo
                        zn2 = 10. * exp(-9.5 * ust(i)**(-.3333)) + &
                                0.11 * 1.5e-5 / amax1(ust(i), 0.01)
                        znt(i) = (1.0 - zw) * zn1 + zw * zn2
                        znt(i) = min(znt(i), 2.85e-3)
                        znt(i) = max(znt(i), 1.27e-7)
                    endif
                endif
                zl = znt(i)
            else
                zl = 0.01
            endif
            flqc(i) = rhox(i) * mavail(i) * ust(i) * karman / denomq(i)
            !       flqc(i)=rhox(i)*mavail(i)*ust(i)*karman/(   &
            !               alog(karman*ust(i)*za(i)/xka+za(i)/zl)-psih(i))
            dtthx = abs(thx(i) - thgb(i))
            if(dtthx.gt.1.e-5)then
                flhc(i) = cpm(i) * rhox(i) * ust(i) * mol(i) / (thx(i) - thgb(i))
                !         write(*,1001)flhc(i),cpm(i),rhox(i),ust(i),mol(i),thx(i),thgb(i),i
                1001   format(f8.5, 2x, f12.7, 2x, f12.10, 2x, f12.10, 2x, f13.10, 2x, f12.8, f12.8, 2x, i3)
            else
                flhc(i) = 0.
            endif
        360 continue

        !
        !-----compute surface moist flux:
        !
        !     if(idry.eq.1)goto 390
        !
        if (present(scm_force_flux)) then
            if (scm_force_flux.eq.1) goto 405
        endif

        do 370 i = its, ite
            qfx(i) = flqc(i) * (qsfc(i) - qx(i))
            qfx(i) = amax1(qfx(i), 0.)
            lh(i) = xlv * qfx(i)
        370 continue

        !-----compute surface heat flux:
        !
        390 continue
        do 400 i = its, ite
            if(xland(i) - 1.5.gt.0.)then
                hfx(i) = flhc(i) * (thgb(i) - thx(i))
                !         if ( present(isftcflx) ) then
                !            if ( isftcflx.ne.0 ) then
                ! ahw: add dissipative heating term (commented out in 3.6.1)
                !               hfx(i)=hfx(i)+rhox(i)*ustm(i)*ustm(i)*wspdi(i)
                !            endif
                !         endif
            elseif(xland(i) - 1.5.lt.0.)then
                hfx(i) = flhc(i) * (thgb(i) - thx(i))
                hfx(i) = amax1(hfx(i), -250.)
            endif
        400 continue

        405 continue

        do i = its, ite
            if((xland(i) - 1.5).ge.0)then
                zl = znt(i)
            else
                zl = 0.01
            endif
            !v3.1.1
            !         chs(i)=ust(i)*karman/(alog(karman*ust(i)*za(i) &
            !                /xka+za(i)/zl)-psih(i))
            chs(i) = ust(i) * karman / denomq(i)
            !        gz2oz0(i)=alog(2./znt(i))
            !        psim2(i)=-10.*gz2oz0(i)
            !        psim2(i)=amax1(psim2(i),-10.)
            !        psih2(i)=psim2(i)
            ! v3.1.1
            !         cqs2(i)=ust(i)*karman/(alog(karman*ust(i)*2.0  &
            !               /xka+2.0/zl)-psih2(i))
            !         chs2(i)=ust(i)*karman/(gz2oz0(i)-psih2(i))
            cqs2(i) = ust(i) * karman / denomq2(i)
            chs2(i) = ust(i) * karman / denomt2(i)
        enddo

        410 continue
        !jdf
        !     do i=its,ite
        !       if(ust(i).ge.0.1) then
        !         rmol(i)=rmol(i)*(-flhc(i))/(ust(i)*ust(i)*ust(i))
        !       else
        !         rmol(i)=rmol(i)*(-flhc(i))/(0.1*0.1*0.1)
        !       endif
        !     enddo
        !jdf

        !
    end subroutine sfclayrev1d

    !====================================================================
    subroutine sfclayrevinit

        integer :: n
        real :: zolf

        do n = 0, 1000
            ! stable function tables
            zolf = float(n) * 0.01
            psim_stab(n) = psim_stable_full(zolf)
            psih_stab(n) = psih_stable_full(zolf)

            ! unstable function tables
            zolf = -float(n) * 0.01
            psim_unstab(n) = psim_unstable_full(zolf)
            psih_unstab(n) = psih_unstable_full(zolf)

        enddo

    end subroutine sfclayrevinit

    function zolri(ri, z, z0)
        !
        if (ri.lt.0.)then
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
            if(abs(fx2).lt.abs(fx1))then
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
        if(zol2 * ri2 .lt. 0.)zol2 = 0.  ! limit zol2 - must be same sign as ri2
        !
        zol20 = zol2 * z0 / z ! z0/l
        zol3 = zol2 + zol20 ! (z+z0)/l
        !
        if (ri2.lt.0) then
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
        psimk = 2 * alog(0.5 * (1 + x)) + alog(0.5 * (1 + x * x)) - 2. * atan(x) + 2. * atan(1.)
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
        psihc = (3. / 2.) * log((yh**2. + yh + 1.) / 3.) - sqrt(3.) * atan((2. * yh + 1) / sqrt(3.)) + 4. * atan(1.) / sqrt(3.)
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
        if(nzol + 1 .le. 1000)then
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
        if(nzol + 1 .le. 1000)then
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
        if(nzol + 1 .le. 1000)then
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
        if(nzol + 1 .le. 1000)then
            psih_unstable = psih_unstab(nzol) + rzol * (psih_unstab(nzol + 1) - psih_unstab(nzol))
        else
            psih_unstable = psih_unstable_full(zolf)
        endif
        return
    end function

    !-------------------------------------------------------------------

end module module_sf_sfclayrev

!
! ----------------------------------------------------------
!


