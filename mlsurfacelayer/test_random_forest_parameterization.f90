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
    real, dimension(ims:ime, jms:jme) :: qstar, u10, v10, th2, t2, q2, qgh, qsfc, ck, cka, cd, cda
    type(random_forests_sfclay) :: rf_sfc
    character(len=74) :: friction_velocity_path
    character(len=74) :: temperature_scale_path
    character(len=71) :: moisture_scale_path
    friction_velocity_path = "/Users/dgagne/data/cabauw_surface_layer_models_20190417/friction_velocity/"
    temperature_scale_path = "/Users/dgagne/data/cabauw_surface_layer_models_20190417/temperature_scale/"
    moisture_scale_path = "/Users/dgagne/data/cabauw_surface_layer_models_20190417/moisture_scale/"
    qv_3d(:, 1, :) = 0.005
    qv_3d(:, 2, :) = 0.006
    qv_3d(:, 3, :) = 0.007
    p_3d(:, 1, :) = 101300.0
    p_3d(:, 2, :) = 101000.0
    p_3d(:, 3, :) = 100500.0
    t_3d(:, 1, :) = 280.0
    t_3d(:, 2, :) = 278.0
    t_3d(:, 3, :) = 275.0
    u_3d(:, 1, :) = 5.0
    u_3d(:, 2, :) = 7.0
    u_3d(:, 3, :) = 9.0
    v_3d = 0.0
    dz_3d(:, 1, :) = 10.0
    dz_3d(:, 2, :) = 20.0
    dz_3d(:, 3, :) = 30.0
    mavail = 1.0
    xland = 1.0
    tsk = 282.0
    psfc = 101325.00
    swdown = 1003.0
    coszen = 0.5
    regime = 4.0
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
    call init_sfclay_random_forest(friction_velocity_path, temperature_scale_path, moisture_scale_path, rf_sfc)
    print *, size(rf_sfc%friction_velocity)
    call sfclay_random_forest(ims, ime, jms, jme, kms, kme, &
                              its, ite, jts, jte, kts, kte, &
                              n_vertial_layers, &
                              u_3d, v_3d, t_3d, qv_3d, p_3d, dz_3d, mavail, &
                              xland, tsk, psfc, swdown, coszen, &
                              rf_sfc, &
                              regime, psim, psih, fm, fh, br, ust, znt, zol, mol, &
                              rmol, hfx, qfx, lh, qstar, u10, v10, th2, t2, q2, qgh, qsfc, &
                              chs, chs2, cqs2, flhc, flqc, ck, cka, cd, cda)
end program test_random_forest_parameterization