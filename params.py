### Set parameters for evolve.py here. ###

## Physics parameters
alpha    = 0.2      # alpha viscosity [HoR < alpha < 1]
gamma    = 1.0      # adiabatic index [1. < gamma < 5./3.]
HoR      = 1.e-3    # disk scale height [HoR < alpha < 1]
tilt     = 10.      # initial disk tilt [degrees; 0 < tilt < 90]
bhspin   = 1.0      # black hole spin [0 < bhspin < 1]
r0       = 3000.    # midpoint of density distribution [r_g; rmin < r0 < rmax]
rw       = 300.     # Gaussian width of density distribution [r_g; rw > 0]
rmin     = 60.0     # Inner radius of disk [r_g]
rmax     = 6000.0   # Outer radius of disk [r_g]
rho_type = "gauss"  # Type of density distribution ["gauss" or "flat"]

# Numerical parameters
tmax    = 1.      # Maximum simulation time [t_visc = r0*r0/nu1(psi=0)]
dt_init = 1.e-9   # initial timestep [t_visc]
ngrid   = 100     # num grid points
dolog   = True    # whether to logarithmically space grid
bc      = "sink"  # boundary condition ["sink" or "outflow"]

# Output
io_freq   = 1e-6                # frequency of outputs [t_viscous]
io_prefix = "./outputs/evolve_" # prefix for output files

# Q1, Q2, Q3
Q_dim = "1d" # Dimension of Q tables ["1d" or "2d"]
smax   = 10. # Max psi value from Q table. Min is always 0. 
Q1_path = "./tables/Q1_1d_a0.2_p0.0_g1.0_np30_ng10000.txt" 
Q2_path = "./tables/Q2_1d_a0.2_p0.0_g1.0_np30_ng10000.txt"
Q3_path = "./tables/Q3_1d_a0.2_p0.0_g1.0_np30_ng10000.txt"
