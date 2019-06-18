program test_random_forest_parameterization
    use module_sf_sfclay_random_forest
    implicit none
    integer, parameter :: ims=1, ime=3, jms=1, jme=3, kms=1, kme=3
    integer, parameter :: its=1, ite=3, jts=1, jte=3, kts=1, kte=3
    integer, parameter :: n_vertial_layers = 2
    real, dimension(ims:ime, kms:kme, jms:jme) :: qv_3d, &
                p_3d, &
                t_3d, &
                u_3d, &
                v_3d, &
                dz_3d

    real, dimension(ims:ime, jms:jme) :: mavail, &
                xland, &
                tsk, &
                psfc, &
                swdown, &
                coszen
    real, dimension(ims:ime, jms:jme) :: regime, br, ust, znt, zol, hfx, qfx, lh, mol, rmol, &
            chs, chs2, cqs2, flhc, flqc, psim, psih, fm, fh
    real, dimension(ims:ime, jms:jme) :: qstar, u10, v10, th2, t2, q2, qsfc, ck, cka, cd, cda
    character(len=74) :: friction_velocity_path
    character(len=74) :: temperature_scale_path
    character(len=71) :: moisture_scale_path
    friction_velocity_path = "/Users/dgagne/data/cabauw_surface_layer_models_20190419/friction_velocity/"
    temperature_scale_path = "/Users/dgagne/data/cabauw_surface_layer_models_20190419/temperature_scale/"
    moisture_scale_path = "/Users/dgagne/data/cabauw_surface_layer_models_20190419/moisture_scale/"
    qv_3d(:, 1, :) = 0.005
    qv_3d(:, 2, :) = 0.006
    qv_3d(:, 3, :) = 0.007
    p_3d(:, 1, :) = 101300.0
    p_3d(:, 2, :) = 101000.0
    p_3d(:, 3, :) = 100500.0
    t_3d(:, 1, :) = 280.0
    t_3d(:, 2, :) = 285.0
    t_3d(:, 3, :) = 290.0
    u_3d(:, 1, :) = 5.0
    u_3d(:, 2, :) = 35.0
    u_3d(:, 3, :) = 25.0
    v_3d = 0.0
    dz_3d(:, 1, :) = 10.0
    dz_3d(:, 2, :) = 20.0
    dz_3d(:, 3, :) = 30.0
    mavail = 1.0
    xland = 1.0
    tsk = 275.0
    psfc = 101325.00
    swdown = 500.0
    coszen = 30.0
    regime = 0.0
    br = -1.0
    ust = 5
    znt = 0.1
    zol = 0.0
    hfx = 0.0
    qfx = 0.0
    lh = 0.0
    mol = 0.0
    rmol = 0.0
    chs = 0.0
    chs2  = 0.0
    print *, qv_3d(1, :, 1)
    print *, p_3d(1, :, 1)
    print *, friction_velocity_path
    call init_sfclay_random_forest(friction_velocity_path, temperature_scale_path, moisture_scale_path)
    print *, size(rf_sfc%friction_velocity)
    call sfclay_random_forest(ims, ime, jms, jme, kms, kme, &
                              its, ite, jts, jte, kts, kte, &
                              n_vertial_layers, &
                              u_3d, v_3d, t_3d, qv_3d, p_3d, dz_3d, mavail, &
                              xland, tsk, psfc, swdown, coszen, znt, &
                             br, ust, mol, qstar, zol, &
                             rmol, hfx, qfx, lh, &
                             regime, psim, psih, fm, fh, &
            u10, v10, th2, t2, q2, qsfc, &
            chs, chs2, cqs2, flhc, flqc, &
            ck, cka, cd, cda)
    print *, ust(1, 1), mol(1, 1), qstar(1, 1)
    print *, hfx(1, 1), qfx(1, 1), lh(1, 1)
    print *, t2(1, 1), u10(1, 1), v10(1, 1), q2(1, 1), qsfc(1, 1)
    print *, zol(1, 1), chs(1, 1), chs2(1, 1), cqs2(1, 1)
    print *, flhc(1, 1), flqc(1, 1), ck(1, 1), cka(1, 1), cd(1, 1), cda(1, 1)
    print *, psim(1, 1), psih(1, 1), fm(1, 1), fh(1, 1)
end program test_random_forest_parameterization