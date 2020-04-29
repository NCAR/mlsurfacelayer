from math import log, atan, sqrt
import numpy as np
from numba import jit

#@jit(nopython=True)
def mo_similarity(u10, v10, tsk, t2, qsfc, q2, psfc, mavail=1, z0=0.01, zt0=0.001, z10=10.0, z2=2.0):
    """
    Calculate flux information based on Monin-Obukhov similarity theory.

    Args:
        u10: 10 m level zonal wind speed [m/s]
        v10: 10 m level meridional wind speed [m/s]
        tsk: surface skin temperature [K]
        t2: 2 m level temperature [K]
        qsfc: ground mixing ratio [kg /kg]
        q2: 2 m level mixing ratio [kg / kg]
        psfc: surface pressure [hPa]
        z0: momentum roughness length
        zt0: heat flux roughness length
        z10: Height of "10 m" layer
        z2: Height of "2 m" layer

    Returns:
        ustar: friction velocity
        tstar: temperature scale
        wthv0: -ustar * tstar
        zeta10: z/L for a z of 10 m. Used for momentum flux
        phim10: momentum universal function at 10 m
        zeta2: z/L for a z of 2 m.
        phih2: sensible heat flux universal function at 2 m.
    """
    if z0 > 0:
        z10oz0 = z10 / z0
        z2oz0 = z2 / z0
        z2ozt0 = z2 / zt0
    else:
        raise ValueError("z0 must be greater than 0")
    # else:
    #    sys.exit("Surface roughnes, z0, must be greter than 0.!")
    #
    # Gravitational acceleration
    g = 9.81
    #
    # Gas constant over spcific heat capacity at constant pressure
    r = 287.058
    cp = 1005.
    rocp = r / cp
    #
    # Reference pressure and temperature
    p0 = 1000.
    t0 = 300.
    #
    # Set M-O parameters based on Dyer 1974 paper
    karman = 0.4
    beta = 5.0
    gamma = 16.0
    #
    # Air density
    rho = psfc / (r * t2)
    #
    # Potential temperature speed at level 1
    th2 = t2 * (p0 / psfc) ** rocp
    th0 = tsk * (p0 / psfc) ** rocp
    #
    # Virtual potential temperature
    thv2 = th2 * (1. + 0.61 * q2)
    thv0 = th0 * (1. + 0.61 * qsfc)
    #
    # Small number
    epsilon = 1.e-6
    #
    # Initial values of drag coefficients - neutrally stratified case
    cd = karman ** 2 / ((log(z10oz0)) ** 2)
    ch = karman ** 2 / ((log(z2ozt0)) ** 2)
    cq = karman ** 2 / ((log(z2ozt0)) ** 2)
    #
    # Initial values of surface friction velocity, temperature scale, and
    # heat flux
    wind_speed = sqrt(u10 * u10 + v10 * v10)
    if wind_speed < 0.1:
        wind_speed = 0.1
        u10 = 0.1
    tauxz = cd * wind_speed * u10
    tauyz = cd * wind_speed * v10
    ustar = (tauxz ** 2 + tauyz ** 2) ** 0.25

    tstar = -ch / ustar * wind_speed * (thv0 - thv2)
    qstar = cq / ustar * mavail * wind_speed * (qsfc - q2)
    wthv0 = -ustar * tstar
    #
    # Set stopping criterion
    diff = 1.
    #
    # Set stability functions
    psim10 = 0.
    psim2 = 0.
    psih2 = 0.
    psiq2 = 0.
    phim10 = 0.
    phih2 = 0
    #
    zeta10 = 0.
    zeta2 = 0.
    count = 0
    while diff > epsilon and count < 100:
        #
        # Surface friction velocity and temperature scale
        tauxz = cd * wind_speed * u10
        tauyz = cd * wind_speed * v10
        ustar = (tauxz ** 2 + tauyz ** 2) ** 0.25
        if ustar < 0.01:
            ustar = 0.01
        wspd2 = ustar / karman * (log(z2oz0) - psim2)
        tstar = -ch / ustar * wspd2 * (thv0 - thv2)
        qstar = cq / ustar * mavail * wspd2 * (qsfc - q2)
        wthv0 = -ustar * tstar
        #
        # Compute drag coefficients
        cdold = cd
        chold = ch
        cqold = cq
        #
        # Neutrally stratified case
        if wthv0 == 0:
            zeta10 = 0.
            zeta2 = 0.
            psim10 = 0.
            psim2 = 0.
            psih2 = 0.
            psiq2 = 0.
            phim10 = 1.
            phih2 = 1.
            cd = karman ** 2 / ((log(z10oz0)) ** 2)
            ch = karman ** 2 / ((log(z2ozt0)) ** 2)
            cq = karman ** 2 / ((log(z2ozt0)) ** 2)
        elif abs(wthv0) > 0:
            #
            # Obukhov length scale
            olength = -ustar ** 3 / (karman * g / t0 * wthv0)
            if abs(olength) < 10 and olength > 0:
                olength = z10
            elif abs(olength) < 10 and olength < 0:
                olength = -z10
            #
            # Free convection
            # if (olength == 0.):
            #    sys.exit("Free convection!")
            #
            # Monin-Obukhov stability parameter
            zeta10 = z10 / olength
            zeta2 = z2 / olength
            #
            # Convective case
            if (zeta2 < -epsilon) & (zeta10 >= -2.):
                xi10 = 1. / ((1. - gamma * zeta10) ** 0.25)
                xi2 = 1. / ((1. - gamma * zeta2) ** 0.25)
                psim10 = log(0.5 * (1.0 + xi10 ** 2) * (0.5 * (1.0 + xi10)) ** 2) \
                    - 2. * atan(xi10) + 0.5 * np.pi
                psim2 = log(0.5 * (1.0 + xi2 ** 2) * (0.5 * (1.0 + xi2)) ** 2) \
                    - 2. * atan(xi2) + 0.5 * np.pi
                psih2 = 2.0 * log(0.5 * (1.0 + xi2 ** 2))
                psiq2 = 2.0 * log(0.5 * (1.0 + xi2 ** 2))
                phim10 = 1. / ((1. - gamma * zeta10) ** 0.25)
                phih2 = 1. / ((1. - gamma * zeta2) ** 0.25)
            #
            # Stably stratified case
            elif (zeta2 > epsilon) & (zeta2 <= 1.):
                psim10 = - beta * zeta10
                psim2 = - beta * zeta2
                psih2 = - beta * zeta2
                psiq2 = - beta * zeta2
                phim10 = (1. + beta * zeta10)
                phih2 = (1. + beta * zeta2)
            #
            # Neutrally stratified case
            elif (zeta2 <= epsilon) & (zeta2 >= -epsilon):
                psim10 = 0.
                psim2 = 0.
                psih2 = 0.
                psiq2 = 0.
                phim10 = 1.
                phih2 = 1.
            #
            cd = karman ** 2 / ((log(z10oz0) - psim10) ** 2)
            ch = karman ** 2 / ((log(z2ozt0) - psim2) * (log(z2ozt0) - psih2))
            cq = karman ** 2 / ((log(z2ozt0) - psim2) * (log(z2ozt0) - psiq2))
        #
        diff = abs(cd - cdold) + abs(ch - chold) + abs(cq - cqold)
        count += 1
    #
    return ustar, tstar, qstar, wthv0, zeta10, phim10, zeta2, phih2


@jit(nopython=True)
def mo_similarity_two_levels(u_low, v_low, u_high, v_high, t_low, t_high, pressure,
                             z_low, z_high):
    """
    Calculate flux information based on Monin-Obukhov similarity theory with instruments at two levels.

    Args:
        u_low: u at lower level in m/s
        v_low: v at lower level in m/s
        u_high: u at higher level in m/s
        v_high: v at higher level in m/s
        t_low: temperature at lower level (K)
        t_high: temperature at higher level (K)
        pressure: surface pressure in hPa
        z_low: height of lower level in m
        z_high: height of higher level in m

    Returns:
        ustar: friction velocity m/s
        tstar: temperature scale K
        wthv0:
        zeta_high: z/L
        phi_m: momentum universal stability function
        phi_h: sensible heat universal stability function
    """
    z_ratio = z_high / z_low
    # else:
    #    sys.exit("Surface roughnes, z0, must be greter than 0.!")
    #
    # Gravitational acceleration
    g = 9.81
    #
    # Gas constant over spcific heat capacity at constant pressure
    r = 287.058
    cp = 1005.
    rocp = r / cp
    #
    # Reference pressure and temperature
    p0 = 1000.
    t0 = 300.
    #
    # Set M-O parameters based on Dyer 1974 paper
    karman = 0.4
    beta = 5.0
    gamma = 16.0
    #
    # Air density
    rho = pressure / (r * t_high)
    #
    # Potential temperature speed at level 1
    th_high = t_high * (p0 / pressure) ** rocp
    th_low = t_low * (p0 / pressure) ** rocp
    #
    # Small number
    epsilon = 1.e-6
    #
    # Initial values of drag coefficients - neutrally stratified case
    cd = karman ** 2 / ((log(z_ratio)) ** 2)
    ch = karman ** 2 / ((log(z_ratio)) ** 2)
    #
    # Initial values of surface friction velocity, temperature scale, and
    # heat flux
    wind_speed_high = sqrt(u_high * u_high + v_high * v_high)
    wind_speed_low = sqrt(u_low * u_low + v_low * v_low)
    if wind_speed_high < 0.1:
        wind_speed_high = 0.1

    if wind_speed_low < 0.01:
        wind_speed_low = 0.01
    # tauxz = cd * (wind_speed_high) * (u_high)
    # tauyz = cd * (wind_speed_high) * (v_high)
    # ustar = (tauxz ** 2 + tauyz ** 2) ** 0.25
    ustar = (cd * (wind_speed_high - wind_speed_low) ** 2) ** 0.5
    if ustar < 0.01:
        ustar = 0.01
    tstar = -ch / ustar * ((wind_speed_high - wind_speed_low) ** 2) ** 0.5 * (th_low - th_high)
    wthv0 = -ustar * tstar
    #
    # Set stopping criterion
    diff = 1.
    #
    # Set stability functions
    psi_m = 0.
    psi_m_low = 0.
    psi_h = 0.
    psi_h_low = 0.
    phi_m = 0.
    phi_h = 0.
    #
    zeta_high = 0.
    zeta_low = 0.
    count = 0
    while diff > epsilon and count < 100:
        #
        # Surface friction velocity and temperature scale
        # tauxz = cd * (wind_speed_high) * (u_high )
        # tauyz = cd * (wind_speed_high) * (v_high)
        # ustar = (tauxz ** 2 + tauyz ** 2) ** 0.25
        ustar = (cd * (wind_speed_high - wind_speed_low) ** 2) ** 0.5
        if ustar < 0.01:
            ustar = 0.01
        tstar = -ch / ustar * ((wind_speed_high - wind_speed_low) ** 2) ** 0.5 * (th_low - th_high)
        wthv0 = -ustar * tstar
        #
        # Compute drag coefficients
        cdold = cd
        chold = ch
        #
        # Neutrally stratified case
        if wthv0 == 0:
            zeta_high = 0.
            zeta_low = 0.
            psi_m = 0.
            psi_m_low = 0.
            psi_h = 0.
            psi_h_low = 0.
            phi_m = 1.
            phi_h = 1.
            cd = karman ** 2 / ((log(z_ratio)) ** 2)
            ch = karman ** 2 / ((log(z_ratio)) ** 2)
        elif abs(wthv0) > 0:
            #
            # Obukhov length scale
            olength = -ustar ** 3 / (karman * g / th_high * wthv0)
            if abs(olength) < (z_high) and olength > 0:
                olength = z_high
            elif abs(olength) < z_high and olength < 0:
                olength = -(z_high)
            #
            # Free convection
            # if (olength == 0.):
            #    sys.exit("Free convection!")
            #
            # Monin-Obukhov stability parameter
            zeta_high = z_high / olength
            zeta_low = z_low / olength
            #
            # Convective case
            if (zeta_high >= -2.) & (zeta_high < -epsilon):
                xi_high = 1. / ((1. - gamma * zeta_high) ** 0.25)
                xi_low = 1. / ((1. - gamma * zeta_low) ** 0.25)

                psi_m = log(0.5 * (1.0 + xi_high ** 2) * (0.5 * (1.0 + xi_high)) ** 2) \
                        - 2. * atan(xi_high) + 0.5 * np.pi

                psi_m_low = log(0.5 * (1.0 + xi_low ** 2) * (0.5 * (1.0 + xi_low)) ** 2) \
                            - 2. * atan(xi_low) + 0.5 * np.pi
                psi_h = 2.0 * log(0.5 * (1.0 + xi_high ** 2))
                psi_h_low = 2.0 * log(0.5 * (1.0 + xi_low ** 2))

                phi_m = 1. / ((1. - gamma * zeta_high) ** 0.25)
                phi_h = 1. / ((1. - gamma * zeta_high) ** 0.25)
            #
            # Stably stratified case
            elif (zeta_high > epsilon) & (zeta_high <= 1.):
                psi_m = - beta * zeta_high
                psi_h = - beta * zeta_high
                psi_m_low = -beta * zeta_low
                psi_h_low = -beta * zeta_low
                phi_m = (1. + beta * zeta_high)
                phi_h = (1. + beta * zeta_high)
            #
            # Neutrally stratified case
            elif (zeta_high <= epsilon) & (zeta_high >= -epsilon):
                psi_m = 0.
                psi_h = 0.
                psi_m_low = 0.
                psi_h_low = 0.
                phi_m = 1.
                phi_h = 1.
            #
            cd = karman ** 2 / ((log(z_ratio) - psi_m + psi_m_low) ** 2)
            ch = karman ** 2 / ((log(z_ratio) - psi_m + psi_m_low) * (log(z_ratio) - psi_h + psi_h_low))
        #
        diff = abs(cd - cdold) + abs(ch - chold)
        count += 1
    #
    return ustar, tstar, wthv0, zeta_high, phi_m, phi_h