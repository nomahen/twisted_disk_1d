### Set parameters for evolve.py here. ###

## Physics parameters
alpha    = 0.2      # alpha viscosity [HoR < alpha < 1]
gamma    = 1.0      # adiabatic index [1. < gamma < 5./3.]
HoR      = 1.e-3    # disk scale height [HoR < alpha < 1]
tilt     = 45.      # initial disk tilt [degrees; 0 < tilt < 90]
bhspin   = 0.0      # black hole spin [0 < bhspin < 1]
r0       = 500.     # midpoint of density distribution [r_g; rmin < r0 < rmax]
rw       = 60.     # Gaussian width of density distribution [r_g; rw > 0]
rmin     = 60.0     # Inner radius of disk [r_g]
rmax     = 1000.0   # Outer radius of disk [r_g]
rho_type = "flat"  # Type of density distribution ["gauss" or "flat"]

# Numerical parameters
tmax    = 0.01    # Maximum simulation time [t_visc = r0*r0/nu1(psi=0)]
cfl     = 0.04    # Courant-Friedrichs-Lewy number
ngrid   = 100     # num grid points
bc      = "outflow"  # boundary condition ["sink" or "outflow" or "mix"]

# Output
io_freq   = 20                # how many number of steps to plot at
io_prefix = "./outputs/evolve_rmin60_rad_" # prefix for output files

# Q1, Q2, Q3
Q_dim = "1d" # Dimension of Q tables ["1d" or "2d"]
smax   = 10. # Max psi value from Q table. Min is always 0. 
Q1_path = "./tables/Q1_1d_a0.2_p0.0_g1.0_np30_ng10000.txt" 
Q2_path = "./tables/Q2_1d_a0.2_p0.0_g1.0_np30_ng10000.txt"
Q3_path = "./tables/Q3_1d_a0.2_p0.0_g1.0_np30_ng10000.txt"

#### package all params to send to evolve
import time
from evolve_fvm import *
t1 = time.time()
evolve(alpha,gamma,HoR,tilt,bhspin,r0,rw,rmin,rmax,rho_type,tmax,cfl,ngrid,bc,io_freq,io_prefix,Q_dim,smax,Q1_path,Q2_path,Q3_path)
print "time in seconds: ", time.time() - t1
