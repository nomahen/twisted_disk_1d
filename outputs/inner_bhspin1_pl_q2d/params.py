### Set parameters for evolve.py here. ###

## Physics parameters
alpha    = 0.5      # alpha viscosity [HoR < alpha < 1]
gamma    = 1.0      # adiabatic index [1. < gamma < 5./3.]
HoR      = 0.02    # disk scale height [HoR < alpha < 1]
tilt     = 45.      # initial disk tilt [degrees; 0 < tilt < 90]
bhspin   = 1.    # black hole spin [0 < bhspin < 1]
r0       = 100.     # midpoint of density distribution [r_g; rmin < r0 < rmax]
rw       = 60.     # Gaussian width of density distribution [r_g; rw > 0]
rmin     = 6.0     # Inner radius of disk [r_g]
rmax     = 100.0  # Outer radius of disk [r_g]
rho_type = "flat"  # Type of density distribution ["gauss" or "flat"]

# Numerical parameters
tmax    = 0.1    # Maximum simulation time [t_visc = r0*r0/nu1(psi=0)]
cfl     = 0.1    # Courant-Friedrichs-Lewy number
ngrid   = 100     # num grid points
bc      = "mix"  # boundary condition ["sink" or "outflow" or "mix" or "infinite"]

# Output
io_freq   = 100              # how many number of steps to plot at
io_prefix = "./outputs/evolve_rmin60_rad_" # prefix for output files

# Q1, Q2, Q3
Q_dim = "2d" # Dimension of Q tables ["1d" or "2d"]
smax   = 10. # Max psi value from Q table. Min is always 0. 
rmax_Q = 100. # If Q tables are 2d, this is maximum radius to which the tables were generated. Results will be off if this doesn't equal rmax.
Q1_path = "./tables/Q1_2d_a0.9_p-0.33_g1.0_np32_ng10000" 
Q2_path = "./tables/Q2_2d_a0.9_p-0.33_g1.0_np32_ng10000"
Q3_path = "./tables/Q3_2d_a0.9_p-0.33_g1.0_np32_ng10000"

#### package all params to send to evolve
import time
from evolve_fvm import *
t1 = time.time()
evolve(alpha,gamma,HoR,tilt,bhspin,r0,rw,rmin,rmax,rho_type,tmax,cfl,ngrid,bc,io_freq,io_prefix,Q_dim,smax,rmax_Q,Q1_path,Q2_path,Q3_path)
print "time in seconds: ", time.time() - t1
