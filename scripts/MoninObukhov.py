
"""
This code computes surface friction velocity and surface potential 
temperature scale using Monin-Obukhov Similarity Theory and measurements of
surface temperature as well as potential temperature and wind speed at 
some reference level z1

Written by Branko Kosovic, September 21, 2018
"""
import numpy as np
import scipy
from math import *
import sys
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def mo_similarity(u10,v10,tsk,t2,qsfc,q2,psfc,z0,zt0):
    # u10 - 10 m level zonal wind speed [m/s]
    # v10 - 10 m level meridional wind speed [m/s]
    # tsk - surface skin temperature [K]
    # t2  - 2 m level temperature [K]
    # qsfc - ground mixing ratio
    # q2  - 2 m level mixing ratio
    # psfc - surface pressure
    # z0 - momentum roughness length
    # zt0 - heat flux roughness length
    #
    # Levels z10 and z2 are set to 10 m and 2m, 
    # but they can be any two levels
    z10  = 10.
    z2   = 2.0
    if (z0 > 0.):
        z10oz0 = z10/z0
        z2oz0  = z2/z0
        z2ozt0 = z2/zt0
    #else:
    #    sys.exit("Surface roughnes, z0, must be greter than 0.!")
    #
    # Gravitational acceleration
    g = 9.81
    #
    # Gas constant over spcific heat capacity at constant pressure
    r = 287.058
    cp = 1005.
    rocp = r/cp
    #
    # Reference pressure and temperature
    p0  = 1000.
    t0  = 300.
    #
    # Set M-O parameters based on Dyer 1974 paper
    karman = 0.4
    beta   = 5.0
    gamma  = 16.0
    #
    # Air density
    rho = psfc/(r*t2)
    #
    # Potential temperature speed at level 1
    th2 = t2*(p0/psfc)**rocp
    th0 = tsk*(p0/psfc)**rocp
    #
    # Virtual potential temperature
    thv2 = th2*(1.+0.61*q2)
    thv0 = th0*(1.+0.61*qsfc)
    #
    # Small number
    epsilon = 1.e-6
    #
    # Initial values of drag coefficients - neutrally stratified case
    cd = karman**2/((log(z10oz0))**2)
    ch = karman**2/((log(z2ozt0))**2)
    cq = karman**2/((log(z2ozt0))**2)
    # 
    # Initial values of surface friction velocity, temperature scale, and
    # heat flux
    tauxz = cd*sqrt(u10*u10+v10*v10)*u10
    tauyz = cd*sqrt(u10*u10+v10*v10)*v10
    ustar = (tauxz**2+tauyz**2)**0.25
    tstar = -ch/ustar*sqrt(u10*u10+v10*v10)*(th0-th2)
    wthv0 = -ustar*tstar
    #
    # Set stopping criterion
    diff = 1.
    # 
    # Set stability functions
    psim10 = 0.
    psim2  = 0.
    psih2  = 0.
    psiq2  = 0.
    phim10 = 0.
    phih2 = 0
    #
    while (diff > epsilon):
        #
        # Surface friction velocity and temperature scale
        tauxz = cd*sqrt(u10*u10+v10*v10)*u10
        tauyz = cd*sqrt(u10*u10+v10*v10)*v10
        ustar = (tauxz**2+tauyz**2)**0.25
        wspd2 = ustar/karman*(log(z2oz0)-psim2)
        tstar = -ch/ustar*sqrt(u10*u10+v10*v10)*(thv0-thv2)
        wthv0 = -ustar*tstar
        #
        # Compute drag coefficients
        cdold = cd
        chold = ch
        cqold = cq
        #
        # Neutrally stratified case
        if (wthv0 == 0.): 
            zeta10 = 0.
            zeta2  = 0.
            psim10 = 0.
            psim2  = 0.
            psih2  = 0.
            psiq2  = 0.
            phim10 = 1.
            phih2  = 1.
            cd = karman**2/((log(z10oz0))**2)
            ch = karman**2/((log(z2ozt0))**2)
            cq = karman**2/((log(z2ozt0))**2)
        #
        if (abs(wthv0) > 0.):
            #
            # Obukhov length scale 
            olength = -ustar**3/(karman*g/t0*wthv0)
            #
            # Free convection
            #if (olength == 0.):
            #    sys.exit("Free convection!")
            #
            # Monin-Obukhov stability parameter
            zeta10 = z10/olength
            zeta2 = z2/olength
            #
            # Convective case
            if ((zeta2  < -epsilon) & (zeta10 >= -2.)):
                xi10 = 1./((1.-gamma*zeta10)**0.25)
                xi2  = 1./((1.-gamma*zeta2)**0.25)
                psim10 = log(0.5*(1.0+xi10**2)*(0.5*(1.0+xi10))**2) \
                       -2.*atan(xi10)+0.5*np.pi 
                psim2  = log(0.5*(1.0+xi2**2)*(0.5*(1.0+xi2))**2) \
                        -2.*atan(xi2)+0.5*np.pi 
                psih2  = 2.0*log(0.5*(1.0+xi2**2))
                psiq2  = 2.0*log(0.5*(1.0+xi2**2))
                phim10 =1./((1.-gamma*zeta10)**0.25)
                phih2  =1./((1.-gamma*zeta2)**0.25)
            #
            # Stably stratified case
            if ((zeta2  > +epsilon) & (zeta10 <= 1.)):
                psim10 = - beta*zeta10
                psim2 = - beta*zeta2
                psih2 = - beta*zeta2
                psiq2 = - beta*zeta2
                phim10 =(1.+beta*zeta10) 
                phih2  =(1.+beta*zeta2) 
            #
            # Neutrally stratified case
            if ((zeta2  <= +epsilon) & (zeta2 >= -epsilon)):
                psim10 = 0.
                psim2  = 0.
                psih2  = 0.
                psiq2  = 0.
                phim10 = 1.
                phih2  = 1.
            #
            cd = karman**2/((log(z10oz0)-psim10)**2)
            ch = karman**2/((log(z2ozt0)-psim2)*(log(z2ozt0)-psih2))
            cq = karman**2/((log(z2ozt0)-psim2)*(log(z2ozt0)-psiq2))
        #
        diff = abs(cd-cdold)+abs(ch-chold)+abs(cq-cqold)
    #
    return ustar, tstar, wthv0, zeta10, phim10, zeta2, phih2
#
########################
#
#--- Inputs
# Wind velocity at 10 m
u10 = 5.
v10 = 0.
#
# Skin temperature and temperature at 2 m
tsk = 280.
t2  = 290.
#
# Ground mixing ratio and mixing ration at 2 m
qsfc = 0.01
q2   = 0.01
#
# Surface pressure 
psfc = 1005.
#
# Momentum roughness length, heat flux roughness and reference level
z0  = 0.1
zt0 = z0
#
phim=[]
zolm=[]
phih=[]
zolh=[]
ts = []
#
for tsk in np.arange(289.,300.,0.01):
    ustar,tstar,wthv0,zeta10,phim10,zeta2,phih2 = \
                       mo_similarity(u10,v10,tsk,t2,qsfc,q2,psfc,z0,zt0)
    if ((zeta10 < -2.) | (zeta10 > 1.)):
        print(" zeta10 = ",zeta10)
        #sys.exit("Stability parameter z/L outside the range of validiti of Monin-Obukhov similarity theory!")
    phim.append(phim10)
    zolm.append(zeta10)
    phih.append(phih2)
    zolh.append(zeta2)
    ts.append(tstar)
    #
    print("ustar = ",ustar,"  tstar = ",tstar," wthv0 = ",wthv0)
#
p=plt.plot(zolm,phim,'ro')
plt.xlabel("z/L")
plt.ylabel("$\Phi_m$")
plt.show()
q=plt.plot(zolh,phih,'ro')
plt.xlabel("z/L")
plt.ylabel("$\Phi_h$")
plt.show()
w = plt.plot(zolh, ts)
plt.show()

