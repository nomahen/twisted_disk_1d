import argparse
p = argparse.ArgumentParser()
p.add_argument('-n',"--ngrid",dest="ngrid",type=int,default=100, help="Number of grid points.")
p.add_argument('-f',"--file",dest="file",type=str,default="./outputs/output_", help="Path to put output files.")

args = p.parse_args()
ngrid = args.ngrid
io_prefix = args.file

### Set parameters for evolve.py here. ###
## this version is for loading certain parameters from command line so i can run scripts

## Physics parameters
gamma    = 1.0      # adiabatic index [1. < gamma < 5./3.]
tilt     = 0.      # initial disk tilt [degrees; 0 < tilt < 90]
bhspin   = 0.      # black hole spin [0 < bhspin < 1]
r0       = 600.     # midpoint of density distribution [r_g; rmin < r0 < rmax]
rw       = 100.      # Gaussian width of density distribution [r_g; rw > 0]
rmin     = 100.0     # Inner radius of disk [r_g]
rmax     = 1000.0   # Outer radius of disk [r_g]
rho_type = "gauss"   # Type of density distribution ["gauss" or "flat"]

# Numerical parameters
tmax    = 1e-6   # Maximum simulation time [t_visc = r0*r0/nu1(psi=0)]
cfl     = 0.8   # Courant-Friedrichs-Lewy number
#ngrid   = 300    # num grid points
bc      = "outflow"  # boundary condition ["sink" or "outflow" or "mix" or "infinite"]

# Output
io_freq   = 1e-2             # Frequency
#io_prefix = "./outputs/output_" # prefix for output files

# Q1, Q2, Q3
Q_dim = "1d" # Dimension of Q tables ["1d" or "2d"]
smax   = 10. # Max psi value from Q table. Min is always 0. 
rmax_Q = 100. # If Q tables are 2d, this is maximum radius to which the tables were generated. Results will be off if this doesn't equal rmax.
Q1_path = "./tables/Q1_1d_a0.2_p0.0_g1.0_np30_ng10000.txt" 
Q2_path = "./tables/Q2_1d_a0.2_p0.0_g1.0_np30_ng10000.txt"
Q3_path = "./tables/Q3_1d_a0.2_p0.0_g1.0_np30_ng10000.txt"

### for test problems
alpha    = 0.2      # alpha viscosity [HoR < alpha < 1]
HoR      = 1e-3     # disk scale height [HoR < alpha < 1]
space_order = 1 # 1 or 2. For heat equation, does nothing, since central differencing is automatically 2nd order. 
time_order  = 1 # 1 or 2. 1: Forward Euler; 2: RK2/Predictor-Corrector (Midpoint Method) (General class is RK2/Predictor-Corrector, Midpoint Method gives coefficients of RK2 class)
which_problem = "twist_smalltilt" # "pulse", "discontinuity", "steady", "steady_smalltilt"    

#### package all params to send to evolve
import time
from evolve_fvm import *
t1 = time.time()
evolve(alpha,gamma,HoR,tilt,bhspin,r0,rw,rmin,rmax,rho_type,tmax,cfl,ngrid,bc,io_freq,io_prefix,Q_dim,smax,rmax_Q,Q1_path,Q2_path,Q3_path,space_order,time_order,which_problem)
print "time in seconds: ", time.time() - t1
