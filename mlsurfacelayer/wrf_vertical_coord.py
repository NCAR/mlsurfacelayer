#!/usr/bin/python
"""
 B. Kosovic, NCAR, March 8, 2016

 wrf_vertical_coord.py

 This program computes eta_levels for WRF namelist.input file
 based on user specified parameters:

 nz  - number of vertical levels
 nzc1 - vertical level at which uniform vertical spacing ends
 nzc2 - vertical level at which the top uniform vertical spacing starts
 dz0 - initial uniform vertical spacing
 pb  - base pressure 
 hb  - base elevation
 eps - tolerance for the iterative process of computing eta_levels
 ztop - top of the domain
 ptop - pressure at the top of the domain
 amp1 - =0.0 - lower bound on the stretching amplitude
 amp2 - upper bound on the stretching amplitude
        notice: stretching factor is cf=1.+amp*f(z) where amp is
                stretching amplitude

"""

import numpy as np

# function standard_atm 
# computes an array of pressure levels - p 
# given an array of vertical coordinates - z 
# and base pressure and elevation pb and hb
#
def standard_atm(z,pb,hb):

   R=8.31447  # J/(K*mol) - universal gas constant

   g0=9.80665 # m/s^2     - standard gravity

   M=0.0289644 # kg/mol   - molar mass of Earth's air

   Tb=290.0      # K        - standard temperature

#   pb=100000.d0   # Pa       - static pressure

#   hb=0.d0        # m        - height of the bottom layer

   p=pb*np.exp(g0*M*(hb-z)/(R*Tb))

   return p
#
####
# function create_eta
# computes an array of eta levels - eta
# given an array of pressure levels - p
# and base pressure and elevation pb and hb
#
def create_eta(z,pb,hb):

    p=standard_atm(z,pb,hb)
 
    s=np.size(p)
    n=s

    eta=-(p-p[n-1])/(p[n-1]-p[0])

    return eta
#
####
# function stretch_coefficient
#
def stretch_coefficient(amp,nz):
 
    cf=np.zeros(nz)+1.


    x=1.0*np.arange(nzc2-nzc1)

    x=x/max(x)
 
    pi=np.arccos(-1.)

    cf[nzc1:nzc2]= 1. + amp * np.abs( ( ( 1. + np.cos(2.*pi*x) )/2. )**8 - 1. ) 

    return cf
#
###
# User specified input parameters
#

# base pressure and height (above sea surface) - number of vertical grid points
pb=100000.0
hb=0.0

# nz - number of vertical grid points
# nzc1 - vertical level at which uniform vertical spacing ends
# nzc2 - vertical level at which the top uniform vertical spacing starts
nz=50
nzc1=10
nzc2=45

# dz0 - grid spacing at the bottom
# ptop - pressure at the top of the domain (set negative to use ztop)
# ztop - height of the domain (set negative to use ptop)
dz0=10.0
ptop=10000.0
ztop=-15000.0

# amp1 - =0.0 - lower bound on the stretching amplitude
# amp2 - upper bound on the stretching amplitude
amp1=0.
amp2=0.23

# eps - tolerance criteria for convergence when computing stretching amplitude 
eps=0.01

#
# End of user specified input parameters
####

z=np.zeros(nz)

kk = [n+1 for n in range(nzc1)]

ll = [n+nzc1 for n in range(nzc2-nzc1)]

mm = [n+nzc2 for n in range(nz-nzc2)]

dz=dz0

for k in kk:
    z[k]=z[k-1]+dz

amp=amp2
cf=stretch_coefficient(amp2,nz)

for l in ll:
    dz=cf[l]*dz
    z[l]=z[l-1]+dz

dz=z[nzc2-1]-z[nzc2-2]

for m in mm:
    z[m]=z[m-1]+dz
   
p=standard_atm(z,pb,hb)
p1=p[nz-1]
z1=z[nz-1]

if ptop > 0. and p1 > ptop:
    print('p1 greater than ptop, increase amp2')
    print('p1 = ',p1,'     ptop = ',ptop)
    quit()

if ztop > 0. and z1 < ztop:
    print('z1 less than ztop, increase amp2')
    print('z1 = ',z1,'     ztop = ',ztop)
    quit()

if ptop > 0.:
    diff=np.abs(p1-ptop)

if ztop > 0.:
    diff=np.abs(z1-ztop)

for k in range(nz):
    print(cf[k],z[k],p[k])

#quit()

it=0

while diff > eps:

     dz=dz0

     amp=amp1+0.5*(amp2-amp1)

     cf=stretch_coefficient(amp,nz)

     for l in ll:
         dz=cf[l]*dz
         z[l]=z[l-1]+dz

     dz = z[nzc2-1]-z[nzc2-2]

     for m in mm:
         z[m]=z[m-1]+dz

     if ztop > 0.:
         if z[nz-1] > ztop:
             amp2=amp
         if z[nz-1] < ztop:
             amp1=amp
         diff=np.abs(z[nz-1]-ztop)
         if amp==0.:
             print('no convergence - change parameters')
             quit()

     p=standard_atm(z,pb,hb)

     if ptop > 0.:
         if p[nz-1] > ptop:
             amp1=amp
         if p[nz-1] < ptop:
             amp2=amp
         diff=np.abs(p[nz-1]-ptop)
         if amp==0.:
             print('no convergence - change parameters')
             quit()

     it=it+1
     print(it,diff,z[nz-1],p[nz-1],amp)


###

print(' level     height      pressure')
for k in range(nz):
   print('{0:6d},{1:12.5f},{2:14.5f}'.format(k,z[k],p[k]))

eta=create_eta(z,pb,hb)

kk = [l+1 for l in range((nz-4)//4)]

print(' eta_levels =                         {0:.5f}, {1:.5f}, {2:.5f}, {3:.5f},'.format(eta[0],eta[1],eta[2],eta[3]))

for k in kk:
   print('                                      {0:.5f}, {1:.5f}, {2:.5f}, {3:.5f},'.format(eta[4*k],eta[4*k+1],eta[4*k+2],eta[4*k+3]))

if nz % 4 == 3:
     print('                                      {0:.5f}, {1:.5f}, {2:.5f},'.format(eta[nz-3],eta[nz-2],eta[nz-1]))

if nz % 4 == 2:
     print('                                      {0:.5f}, {1:.5f},'.format(eta[nz-2],eta[nz-1]))

if nz % 4 == 1:
     print('                                      {0:.5f},'.format(eta[nz-1]))
