### Set parameters for evolve.py here. ###

## for counter rotation driven accretion
rc       = 50       # critical radius at which prec difference is [r_g]
prec_c   = 95.      # initial precession angle beyond r_c [deg]

## Physics parameters
alpha    = 0.2      # alpha viscosity [HoR < alpha < 1]
gamma    = 1.0      # adiabatic index [1. < gamma < 5./3.]
HoR      = 1.e-3    # disk scale height [HoR < alpha < 1]
tilt     = 45.      # initial disk tilt [degrees; 0 < tilt < 90]
bhspin   = 0.0      # black hole spin [0 < bhspin < 1]
r0       = 50.      # midpoint of density distribution [r_g; rmin < r0 < rmax]
rw       = 10.      # Gaussian width of density distribution [r_g; rw > 0]
rmin     = 10.0     # Inner radius of disk [r_g]
rmax     = 100.0    # Outer radius of disk [r_g]
rho_type = "gauss"  # Type of density distribution ["gauss" or "flat"]

# Numerical parameters
tmax    = 0.4     # Maximum simulation time [t_visc = r0*r0/nu1(psi=0)]
dt_init = 5.e-9   # initial timestep [t_visc]
ngrid   = 100     # num grid points
dolog   = True    # whether to logarithmically space grid
bc      = "sink"  # boundary condition ["sink" or "outflow"]

# Output
io_freq   = 1e-4                # frequency of outputs [t_viscous]
io_prefix = "./outputs/evolve_ctr_" # prefix for output files

# Q1, Q2, Q3
Q_dim = "1d" # Dimension of Q tables ["1d" or "2d"]
smax   = 10. # Max psi value from Q table. Min is always 0. 
Q1_path = "./tables/Q1_1d_a0.2_p0.0_g1.0_np30_ng10000.txt" 
Q2_path = "./tables/Q2_1d_a0.2_p0.0_g1.0_np30_ng10000.txt"
Q3_path = "./tables/Q3_1d_a0.2_p0.0_g1.0_np30_ng10000.txt"
