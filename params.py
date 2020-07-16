### Set parameters for evolve.py here. ###

## Physics parameters
alpha    = 0.2      # alpha viscosity [HoR < alpha < 1]
gamma    = 1.0      # adiabatic index [1. < gamma < 5./3.]
r0       = 600.     # midpoint of density distribution [r_g; rmin < r0 < rmax]
rw       = 100.      # Gaussian width of density distribution [r_g; rw > 0]

#################################

# Q1, Q2, Q3
Q_dim = "1d" # Dimension of Q tables ["1d" or "2d"]
smax   = 10. # Max psi value from Q table. Min is always 0. 
rmax_Q = 100. # If Q tables are 2d, this is maximum radius to which the tables were generated. Results will be off if this doesn't equal rmax.
Q1_path = "./tables/Q1_1d_a0.2_p0.0_g1.0_np30_ng10000.txt" 
Q2_path = "./tables/Q2_1d_a0.2_p0.0_g1.0_np30_ng10000.txt"
Q3_path = "./tables/Q3_1d_a0.2_p0.0_g1.0_np30_ng10000.txt"

# Output
io_freq   = 1e-2             # Frequency
io_prefix = "./outputs/t65/output_" # prefix for output files

# Numerical parameters
tmax    = 1e-3   # Maximum simulation time [t_visc = r0*r0/nu1(psi=0)]
cfl     = 0.8   # Courant-Friedrichs-Lewy number
soft    = 0.0  # softening parameter for warp amplitude
ngrid   = 512    # num grid points
bc      = "outflow_alt"  # boundary condition ["sink" or "outflow" or "mix" or "infinite"]

### for test problems
distr_type = "gauss_rho"   # Type of density distribution ["gauss_rho", "gauss_am", "flat_rho","flat_am"]
tilt     = 65.      # initial disk tilt [degrees; 0 < tilt < 90]
bhspin   = 1.      # black hole spin [0 < bhspin < 1]
HoR      = 1e-3     # disk scale height [HoR < alpha < 1]
rmin     = 2.0     # Inner radius of disk [r_g]
rmax     = 100.0   # Outer radius of disk [r_g]
space_order = 2 # 1 or 2. For heat equation, does nothing, since central differencing is automatically 2nd order. 
time_order  = 2 # 1 or 2. 1: Forward Euler; 2: RK2/Predictor-Corrector (Midpoint Method) (General class is RK2/Predictor-Corrector, Midpoint Method gives coefficients of RK2 class)
which_problem = "flat" # "discontinuity", "flat", "twist"

#### package all params to send to evolve
import time
from evolve_fvm import *
t1 = time.time()
evolve(alpha,gamma,HoR,tilt,bhspin,r0,rw,rmin,rmax,distr_type,tmax,cfl,ngrid,bc,io_freq,io_prefix,Q_dim,smax,rmax_Q,Q1_path,Q2_path,Q3_path,space_order,time_order,which_problem,soft)
print "time in seconds: ", time.time() - t1
