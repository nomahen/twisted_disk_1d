"""
1D evolution of twisted accretion disks
"""

import numpy as np
from scipy.interpolate import interp1d
from params import *

## Helper functions ##

def load_Q(path):
    # Builds Q1,Q2 or Q3 from path. Currently assumes 1D file. 
    data = open(path,"r")
    parsed = []
    for line in data:
        parsed.append(np.array(line.split()).astype(float))
    return np.array(parsed)[:,0]

#########

## Build interpolation functions 

# Parse data 
Q1_parsed = load_Q(Q1_path)
Q2_parsed = load_Q(Q2_path)
Q3_parsed = load_Q(Q3_path)
ng_Q = len(Q1_parsed) # Get length of Q

# Psi array that Q1/Q2/Q3 correspond to
s_arr = np.logspace(0,np.log10(smax+1),ng_Q) - 1

# Do interpolation with extrapolation for psi > smax
Q1_func = interp1d(s_arr,np.log10(-Q1_parsed + 1e-30),fill_value='extrapolate')
Q2_func = interp1d(s_arr,np.log10(Q2_parsed + 1e-30),fill_value='extrapolate')
Q3_func = interp1d(s_arr,np.log10(Q3_parsed + 1e-30),fill_value='extrapolate')

########

## Build arrays

# r + dr arrays can be linear or log10 spaced
if dolog:
    r = np.logspace(np.log10(rmin),np.log10(rmax),ngrid)
    dr = r[1:-1]*np.log(10)*(np.log10(r)[2:] - np.log10(r)[:-2])
else:
    r = np.linspace(rmin,rmax,ngrid)
    dr = r[2:] - r[:-2]

# orbital frequency is Keplerian
omega = r**(-3./2.)

# change tilt to radians
tilt *= np.pi/180.

# density distribution can be "flat" or "gaussian"
# note: probably most intelligent to normalize rho such that total disk mass
# is constant regardless of distribution
if rho_type == "flat":
    density = np.ones(ngrid)
elif rho_type == "gauss":
    density = (1.0/(rw*np.sqrt(2.0*np.pi))) * np.exp(-0.5*((r - r0)/rw)**2.0)
    density /= np.average(density)
else:
    print "Error! rho_type needs to be set to \"gauss\" or \"flat\"! Exiting"
    exit()

# build angular momentum quantities
amom_mag     = density * omega * r * r # angular momentum magnitude
amom_unit    = np.array([np.sin(tilt),0.0,np.cos(tilt)]) # single amom unit vector
amom_uvector = np.array([amom_unit]*ngrid) # amom unit vector extended to radial grid
amom_vector  = np.copy(amom_uvector) # building [Lx,Ly,Lz] for each radial grid element
for i in range(3): amom_vector[:,i] *= amom_mag

# for Lense-Thirring source term
omega_p = np.zeros(3*ngrid)
omega_p = np.reshape(omega_p, (ngrid,3))
omega_p[:,2] = 2.0 * bhspin / r**3.0 # x/y components are zero, z component is LT precession frequency

# calculate (approximate) viscous time (t_visc = r0**2./nu1(psi=0))
nu_ref    = (-2.0/3.0)*(-1.0*10**(Q1_func(0)))*((HoR**2.0)*r0**0.5)
t_viscous = r0*r0/nu_ref

# convert tmax, dt_init from t_viscous units to code units
tmax    = tmax*t_viscous
dt_init = dt_init*t_viscous
io_freq = io_freq*t_viscous

########

## Do initial input/output

print "#### Parameters ####\n"
print "alpha     = %s\n" % alpha
print "gamma     = %s\n" % gamma
print "HoR       = %s\n" % HoR
print "tilt      = %s [deg]\n" % tilt
print "bhspin    = %s\n" % bhspin
print "r0        = %s [r_g]\n" % r0
print "rw        = %s [r_g]\n" % rw
print "rmin      = %s [r_g]\n" % rmin
print "rmax      = %s [r_g]\n" % rmax
print "rho_type  = %s \n" % rho_type
print "tmax      = %s [t_viscous]\n" % (tmax/t_viscous)
print "dt_init   = %s [t_viscous]\n" % (dt_init/t_viscous)
print "dolog     = %s\n" % dolog
print "bc        = %s\n" % bc
print "io_freq   = %s [t_viscous]\n" % io_freq
print "io_prefix = %s\n" % io_prefix
print "Q_dim     = %s\n" % Q_dim
print "smax      = %s\n" % smax
print "Q1_path   = %s\n" % Q1_path
print "Q2_path   = %s\n" % Q2_path
print "Q3_path   = %s\n" % Q3_path
print "####################\n\n"
print "Beginning simulation...\n\n"

########

#############
## Evolve! ##
#############

## Prepare initial conditions first

# Get initial timestep
dt = np.copy(dt_init)

# Initialize angular momentum array to initial condition
L = np.copy(amom_vector)

# Initialize arrays
psi = np.zeros(len(L))
Q1  = np.zeros(len(L))
Q2  = np.zeros(len(L))
Q3  = np.zeros(len(L))
nu1 = np.zeros(len(L))
nu2 = np.zeros(len(L))
nu3 = np.zeros(len(L))
prec_old = np.zeros(len(L))

# Initialize time and output counter
t = 0.0
io_cnt = 0

# iterate!
while (t < tmax):
    # initialize derivatives to zero at the beginning of every iteration
    dLxdt = np.zeros(len(L))
    dLydt = np.zeros(len(L))
    dLzdt = np.zeros(len(L))

    # we will evolve the cartesian components of the angular momentum vectors
    Lx = L[:,0]
    Ly = L[:,1]
    Lz = L[:,2]
    Lmag = np.sqrt(Lx**2.0 + Ly**2.0 + Lz**2.0)

    # now construct the components of the angular momentum unit vectors
    lx = Lx/Lmag
    ly = Ly/Lmag
    lz = Lz/Lmag
    l = np.array(zip(lx,ly,lz))

    # calculate warp parameter
    psi_x = (0.5*r[1:-1]/dr)*(l[2:,0]-l[:-2,0])
    psi_y = (0.5*r[1:-1]/dr)*(l[2:,1]-l[:-2,1])
    psi_z = (0.5*r[1:-1]/dr)*(l[2:,2]-l[:-2,2])
    psi[1:-1] = np.sqrt(psi_x**2.0 + psi_y**2.0 + psi_z**2.0)

    # calculate nu1,nu2,nu3
    nu1[1:-1] = (-2.0/3.0)*(-1.0*10**(Q1_func(psi[1:-1])))*((HoR**2.0)*r[1:-1]**0.5)
    nu2[1:-1] = 2.0*10**(Q2_func(psi[1:-1]))*((HoR**2.0)*r[1:-1]**0.5)
    nu3[1:-1] = 10**(Q3_func(psi[1:-1]))*((HoR**2.0)*r[1:-1]**0.5)

    # fill guard cells for derivative quantities
    if   (bc=="sink"): #### Apply sink boundary conditions
        psi[0] = 1e-10 * psi[1]
        psi[-1] = 1e-10 * psi[-2]
        nu1[0] = 1e-10 * nu1[1]
        nu1[-1] = 1e-10 * nu1[-2]
        nu2[0] = 1e-10 * nu2[1]
        nu2[-1] = 1e-10 * nu2[-2]
        nu3[0] = 1e-10 * nu3[1]
        nu3[-1] = 1e-10 * nu3[-2]

    elif (bc=="outflow"): #### Apply outflow boundary conditions
        psi[0] = psi[1]
        psi[-1] = psi[-2]
        nu1[0] = nu1[1]
        nu1[-1] = nu1[-2]
        nu2[0] = nu2[1]
        nu2[-1] = nu2[-2]
        nu3[0] = nu3[1]
        nu3[-1] = nu3[-2]


    #### Lets begin constructing the terms to evolve Lx, Ly, and Lz

    ## f1
    f1_x = (3.0/(4.0*r[1:-1]))*(1.0/dr**2.0)*((r[2:]+r[1:-1])*(l[2:,0]+l[1:-1,0])*(nu1[2:]*Lmag[2:]-nu1[1:-1]*Lmag[1:-1]) - (r[1:-1]+r[:-2])*(l[1:-1,0]+l[:-2,0])*(nu1[1:-1]*Lmag[1:-1]-nu1[:-2]*Lmag[:-2]))
    f1_y = (3.0/(4.0*r[1:-1]))*(1.0/dr**2.0)*((r[2:]+r[1:-1])*(l[2:,1]+l[1:-1,1])*(nu1[2:]*Lmag[2:]-nu1[1:-1]*Lmag[1:-1]) - (r[1:-1]+r[:-2])*(l[1:-1,1]+l[:-2,1])*(nu1[1:-1]*Lmag[1:-1]-nu1[:-2]*Lmag[:-2]))
    f1_z = (3.0/(4.0*r[1:-1]))*(1.0/dr**2.0)*((r[2:]+r[1:-1])*(l[2:,2]+l[1:-1,2])*(nu1[2:]*Lmag[2:]-nu1[1:-1]*Lmag[1:-1]) - (r[1:-1]+r[:-2])*(l[1:-1,2]+l[:-2,2])*(nu1[1:-1]*Lmag[1:-1]-nu1[:-2]*Lmag[:-2]))

    ## f2
    f2_x = (1.0/(16.0*r[1:-1]))*(1.0/dr**2.0)*((nu2[2:]+nu2[1:-1])*(r[2:]+r[1:-1])*(Lmag[2:]+Lmag[1:-1])*(l[2:,0]-l[1:-1,0]) - (nu2[1:-1]+nu2[:-2])*(r[1:-1]+r[:-2])*(Lmag[1:-1]+Lmag[:-2])*(l[1:-1,0]-l[:-2,0]))
    f2_y = (1.0/(16.0*r[1:-1]))*(1.0/dr**2.0)*((nu2[2:]+nu2[1:-1])*(r[2:]+r[1:-1])*(Lmag[2:]+Lmag[1:-1])*(l[2:,1]-l[1:-1,1]) - (nu2[1:-1]+nu2[:-2])*(r[1:-1]+r[:-2])*(Lmag[1:-1]+Lmag[:-2])*(l[1:-1,1]-l[:-2,1]))
    f2_z = (1.0/(16.0*r[1:-1]))*(1.0/dr**2.0)*((nu2[2:]+nu2[1:-1])*(r[2:]+r[1:-1])*(Lmag[2:]+Lmag[1:-1])*(l[2:,2]-l[1:-1,2]) - (nu2[1:-1]+nu2[:-2])*(r[1:-1]+r[:-2])*(Lmag[1:-1]+Lmag[:-2])*(l[1:-1,2]-l[:-2,2]))

    ## f3
    f3_x = (1.0/(8.0*r[1:-1]))*(1.0/dr**3.0)*((0.5*(nu2[2:]+nu2[1:-1])*((r[2:]+r[1:-1])**2.0)*((l[2:,0]-l[1:-1,0])**2.0 + (l[2:,1]-l[1:-1,1])**2.0 + (l[2:,2]-l[1:-1,2])**2.0) - 3.0*(nu1[2:]+nu1[1:-1]))*(L[2:,0] + L[1:-1,0]) - (0.5*(nu2[1:-1]+nu2[:-2])*((r[1:-1]+r[:-2])**2.0)*((l[1:-1,0]-l[:-2,0])**2.0 + (l[1:-1,1]-l[:-2,1])**2.0 + (l[1:-1,2]-l[:-2,2])**2.0) - 3.0*(nu1[1:-1]+nu1[:-2]))*(L[1:-1,0] + L[:-2,0]))
    f3_y = (1.0/(8.0*r[1:-1]))*(1.0/dr**3.0)*((0.5*(nu2[2:]+nu2[1:-1])*((r[2:]+r[1:-1])**2.0)*((l[2:,0]-l[1:-1,0])**2.0 + (l[2:,1]-l[1:-1,1])**2.0 + (l[2:,2]-l[1:-1,2])**2.0) - 3.0*(nu1[2:]+nu1[1:-1]))*(L[2:,1] + L[1:-1,1]) - (0.5*(nu2[1:-1]+nu2[:-2])*((r[1:-1]+r[:-2])**2.0)*((l[1:-1,0]-l[:-2,0])**2.0 + (l[1:-1,1]-l[:-2,1])**2.0 + (l[1:-1,2]-l[:-2,2])**2.0) - 3.0*(nu1[1:-1]+nu1[:-2]))*(L[1:-1,1] + L[:-2,1]))
    f3_z = (1.0/(8.0*r[1:-1]))*(1.0/dr**3.0)*((0.5*(nu2[2:]+nu2[1:-1])*((r[2:]+r[1:-1])**2.0)*((l[2:,0]-l[1:-1,0])**2.0 + (l[2:,1]-l[1:-1,1])**2.0 + (l[2:,2]-l[1:-1,2])**2.0) - 3.0*(nu1[2:]+nu1[1:-1]))*(L[2:,2] + L[1:-1,2]) - (0.5*(nu2[1:-1]+nu2[:-2])*((r[1:-1]+r[:-2])**2.0)*((l[1:-1,0]-l[:-2,0])**2.0 + (l[1:-1,1]-l[:-2,1])**2.0 + (l[1:-1,2]-l[:-2,2])**2.0) - 3.0*(nu1[1:-1]+nu1[:-2]))*(L[1:-1,2] + L[:-2,2]))

    ## f4
    f4_x = (1.0/(8.0*r[1:-1]))*(1.0/dr**2.0)*((nu3[2:]+nu3[1:-1])*(r[2:]+r[1:-1])*((L[2:,1]+L[1:-1,1])*(l[2:,2]-l[1:-1,2]) - (L[2:,2]+L[1:-1,2])*(l[2:,1]-l[1:-1,1])) - (nu3[1:-1]+nu3[:-2])*(r[1:-1]+r[:-2])*((L[1:-1,1]+L[:-2,1])*(l[1:-1,2]-l[:-2,2]) - (L[1:-1,2]+L[:-2,2])*(l[1:-1,1]-l[:-2,1])))
    f4_y = (1.0/(8.0*r[1:-1]))*(1.0/dr**2.0)*((nu3[2:]+nu3[1:-1])*(r[2:]+r[1:-1])*((L[2:,2]+L[1:-1,2])*(l[2:,0]-l[1:-1,0]) - (L[2:,0]+L[1:-1,0])*(l[2:,2]-l[1:-1,2])) - (nu3[1:-1]+nu3[:-2])*(r[1:-1]+r[:-2])*((L[1:-1,2]+L[:-2,2])*(l[1:-1,0]-l[:-2,0]) - (L[1:-1,0]+L[:-2,0])*(l[1:-1,2]-l[:-2,2])))
    f4_z = (1.0/(8.0*r[1:-1]))*(1.0/dr**2.0)*((nu3[2:]+nu3[1:-1])*(r[2:]+r[1:-1])*((L[2:,0]+L[1:-1,0])*(l[2:,1]-l[1:-1,1]) - (L[2:,1]+L[1:-1,1])*(l[2:,0]-l[1:-1,0])) - (nu3[1:-1]+nu3[:-2])*(r[1:-1]+r[:-2])*((L[1:-1,0]+L[:-2,0])*(l[1:-1,1]-l[:-2,1]) - (L[1:-1,1]+L[:-2,1])*(l[1:-1,0]-l[:-2,0])))

    ## f5
    f5_x = omega_p[1:-1,2]*(-1.0)*L[1:-1,1]
    f5_y = omega_p[1:-1,2]*L[1:-1,0]


    #### Save before updates
    if ((t%io_freq < dt)):
        print "t/tmax = %s, dt/tmax = %s\n" % (t/tmax,dt/tmax)
        np.savetxt(io_prefix + str(io_cnt) + ".csv", np.array(zip(Lx,Ly,Lz,r)))
        io_cnt += 1

    #### Apply updates!
    dLxdt[1:-1] = dLxdt[1:-1] + f1_x + f2_x + f3_x + f4_x + f5_x
    dLydt[1:-1] = dLydt[1:-1] + f1_y + f2_y + f3_y + f4_y + f5_y
    dLzdt[1:-1] = dLzdt[1:-1] + f1_z + f2_z + f3_z + f4_z

    Lx = Lx + dt*dLxdt
    Ly = Ly + dt*dLydt
    Lz = Lz + dt*dLzdt

    #### Update timestep
    t += dt

    if   (bc=="sink"): #### Apply sink boundary conditions
        Lx[0] = 1e-10 * Lx[1]
        Lx[-1] = 1e-10 * Lx[-2]
        Ly[0] = 1e-10 * Ly[1]
        Ly[-1] = 1e-10 * Ly[-2]
        Lz[0] = 1e-10 * Lz[1]
        Lz[-1] = 1e-10 * Lz[-2]
    elif (bc=="outflow"): #### Apply outflow boundary conditions
        Lx[0] = Lx[1];
        Lx[-1] = Lx[-2];
        Ly[0] = Ly[1];
        Ly[-1] = Ly[-2];
        Lz[0] = Lz[1];
        Lz[-1] = Lz[-2];

    #### Repackage array (but why?)
    L = np.array(zip(Lx,Ly,Lz))
