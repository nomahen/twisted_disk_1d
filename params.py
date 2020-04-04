### Set parameters for evolve.py here. ###

## Physics parameters
alpha    = 0.2      # alpha viscosity [HoR < alpha < 1]
gamma    = 1.0      # adiabatic index [1. < gamma < 5./3.]
HoR      = 1.e-3    # disk scale height [HoR < alpha < 1]
tilt     = 0.       # initial disk tilt [degrees; 0 < tilt < 90]
bhspin   = 0.0      # black hole spin [0 < bhspin < 1]
r0       = 500.     # midpoint of density distribution [r_g; rmin < r0 < rmax]
rw       = 60.     # Gaussian width of density distribution [r_g; rw > 0]
rmin     = 60.0     # Inner radius of disk [r_g]
rmax     = 1000.0   # Outer radius of disk [r_g]
rho_type = "gauss"  # Type of density distribution ["gauss" or "flat"]

# Numerical parameters
tmax    = 0.1    # Maximum simulation time [t_visc = r0*r0/nu1(psi=0)]
cfl     = 0.5     # Courant-Friedrichs-Lewy number
ngrid   = 50     # num grid points
bc      = "mix"  # boundary condition ["sink" or "outflow"]

# Output
io_freq   = 1e-5                # frequency of outputs [t_viscous]
io_prefix = "./outputs/evolve_rmin60_" # prefix for output files

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
