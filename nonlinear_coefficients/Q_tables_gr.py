import numpy as np
import sys
import argparse
from mpi4py import MPI
from scipy.integrate import odeint
from scipy.optimize import least_squares

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

p = argparse.ArgumentParser()
p.add_argument('-a',"--alpha",dest="alpha",type=float,default=0.2, help="Alpha viscosity coefficient.")
p.add_argument('-g',"--gamma",dest="gamma",type=float,default=5./3., help="Adiabatic index. Set between 1 and 5/3.")
p.add_argument('-p',"--alpha_exp",dest="alpha_exp",type=float,default=0., help="Exponent for radially dependent alpha viscosity. alpha = a*r**p.")
p.add_argument('-rmax',"--rmax",dest="rmax",type=float,default=100., help = "Maximum radius in units of r_g. Minimum radius is always 1.")
p.add_argument('-smax',"--smax",dest="smax",type=float,default=10., help = "Maximum warp parameter. Minimum warp parameter is always 0.")
p.add_argument('-np',"--npts",dest="npts",type=int,default=10, help = "Length of r (and s arrays, if 2D).")
p.add_argument('-ng',"--ngrid",dest="ngrid",type=int,default=10000, help = "Number of grid points in call to least squares.")
p.add_argument('-2d',"--dim",dest="dim",type=int,default=1, help = "Whether to produce 1D or 2D table. 0 ~ 1D, 1 ~ 2D. This script is 2D only.")
p.add_argument('-b',"--bhspin",dest="bhspin",type=float,default=1, help = "Dimensionless black hole spin; modifies radial epicyclic frequency. ")
args = p.parse_args()

a = args.alpha
g = args.gamma
p = args.alpha_exp
rmax = args.rmax
smax = args.smax
npts = args.npts
ngpts = args.ngrid
dim = args.dim
bhspin = args.bhspin

#if (dim != 0):
#    print "Only dim = 0 supported! Exiting"
#    sys.exit()

# Filenames reflect parameters chosen, and assumes you are inside the git repository; this sends the generated coefficients 
# to the tables directory of the git repository. 
fn1 = "../tables/Q1_%sd_a%s_p%s_g%s_np%s_ng%s_bhspin%s" % (dim+1,a,p,g,npts,ngpts,bhspin)
fn2 = "../tables/Q2_%sd_a%s_p%s_g%s_np%s_ng%s_bhspin%s" % (dim+1,a,p,g,npts,ngpts,bhspin)
fn3 = "../tables/Q3_%sd_a%s_p%s_g%s_np%s_ng%s_bhspin%s" % (dim+1,a,p,g,npts,ngpts,bhspin)

s_arr = np.logspace(0,np.log10(smax+1),npts) - 1
r_arr = np.logspace(0,np.log10(rmax),npts)

## Define functions ##

def kappa(r):
    """
    dimensionless epicyclic frequency
    k^2 = 4*Omega^2 +2*r*Omega*Omega'
    return k/Omega    

    In Keplerian case this is always 1

    Here we've written GR case from Torok + Stuchlik 2005 Equation 4
    """
    return np.sqrt(1. - 6./r + 8.*bhspin/r**(1.5) -3.*(bhspin**2.)/r**2. + 1e-10)

def alpha(a,r,p):
    "Calculates radially dependent alpha viscosity"
    return a*r**p

def Q(x,sol,a,r,s,p):
    "Calculates Q coefficients given f_i(x) (i=1..6), alpha, radius array, warp parameters. See Ogilvie 1999 Equations 112 and 120."
    f1,f2,f3,f4,f5,f6 = sol[:,0],sol[:,1],sol[:,2],sol[:,3],sol[:,4],sol[:,5]
    k = kappa(r)
    a = alpha(a,r,p)
    Q1_integrand = f6*[-0.5*(4-k**2.0)*a*f2-f3*f5+a*f2*f5*s*np.cos(x)]
    Q4_integrand = np.exp(1j*x)*f6*[f3 - 1j*f3*(f4+f3*s*np.cos(x)) + 1j*a*f2*(f4+f3*s*np.cos(x))*s*np.cos(x) - 1j*a*f2*f3 - 1j*a*f2*s*np.sin(x)]
    Q2_integrand = Q4_integrand.real
    Q3_integrand = Q4_integrand.imag
    
    Q1 = np.average(Q1_integrand)
    Q2 = np.average(Q2_integrand)/(s + int(s==0.0)*1e-16)
    Q3 = np.average(Q3_integrand)/(s + int(s==0.0)*1e-16)
    return Q1,Q2,Q3

def f_appx(a,r,s,g,p):
    # Calculate Taylor expanded initial condition for fi(x) arrays (i=1,...6). See appendix of Ogilvie 1999.
    k = kappa(r)
    a = alpha(a,r,p)
    npts =1000
    x = np.linspace(0,2.*np.pi,npts)
    
    ## 0-th order
    f10 = np.ones(npts) # Unsure, this isn't listed, I'm guessing!
    f20 = np.ones(npts)
    f60 = np.ones(npts)
    
    ## 1st order
    Zr1 = (1j - (4. - k*k)*a + 1j*a*a)/((1.-k*k) + 2j*a - a*a)
    Cr1 = Zr1.real
    Sr1 = Zr1.imag
    Zp1 = 0.5*((k*k + 2j*(2.-k*k)*a - (4-k*k)*a*a)/((1-k*k) + 2j*a - a*a))
    Cp1 = Zp1.real
    Sp1 = Zp1.imag
    f31 = Cr1*np.cos(x) + Sr1*np.sin(x)
    f51 = Cp1*np.cos(x) + Sp1*np.sin(x)
    
    ## 2nd order
    Zt2 = (-3j + 2*(6-k*k)*a - (1+k*k)*1j*a*a + 2*a**3.)/(((3.-g)+2j*(4./3.)*a)*((1-k*k)+2j*a-a*a))
    Ct2 = Zt2.real
    St2 = Zt2.imag
    
    f12 = -0.5*(g-1.)*f10*(St2*np.cos(2*x) - Ct2*np.sin(2*x))
    f22 = -0.5*(g+1.)*(St2*np.cos(2*x) - Ct2*np.sin(2*x)) + 0.5*Sr1 - 0.5*a*Cr1
    f42 = Ct2*np.cos(2*x) + St2*np.sin(2*x)
    f62 = St2*np.cos(2*x) - Ct2*np.sin(2*x)
    
    ## Calculate
    f1 = f10 + s*s*f12
    f2 = f20 + s*s*f22
    f3 = s*f31
    f4 = s*s*f42
    f5 = s*f51
    f6 = f60 + s*s*f62
        
    return [f1[0],f2[0],f3[0],f4[0],f5[0],f6[0]]
    
    
    
def f(y,x,a,r,s,g,p):
    # Calculates dfi/dx(x) for (i=1...6); see equations 105-109 + 117 of Ogilvie 1999
    f1,f2,f3,f4,f5,f6 = y[0],y[1],y[2],y[3],y[4],y[5]
    k = kappa(r)
    a = alpha(a,r,p)

    # Equations 105-109 Ogilvie 1999
    df1_dx = (g - 1.0)*f4*f1
    df2_dx = (g + 1.0)*f4*f2
    df3_dx = f4*f3 + 2*f5 + (1.0 + (1.0/3.0)*a*f4)*f2*s*np.cos(x) - a*f2*f3*(1.0+s*s*np.cos(x)**2) - a*f2*s*np.sin(x)
    df4_dx = -df3_dx*s*np.cos(x) + 2.0*f3*s*np.sin(x) + f4*(f4+f3*s*np.cos(x)) + 1.0 - (1 + (1./3.)*a*f4)*f2 - a*f2*(f4+f3*s*np.cos(x))*(1+s*s*np.cos(x)**2.0) + a*f2*s*s*np.cos(x)*np.sin(x)
    df5_dx = f4*f5 - 0.5*k*k*f3 - a*f2*f5*(1.0+s*s*np.cos(x)**2.0) + 0.5*(4-k*k)*a*f2*s*np.cos(x)
    df6_dx = -2.0*f4*f6

    return [df1_dx,df2_dx,df3_dx,df4_dx,df5_dx,df6_dx]

def objective(ic,x,a,r,s,g,p):
    dx = np.average(x[1:] - x[:-1])
    
    f1a,f2a,f3a,f4a,f5a,f6a = ic[:6]
    df1_dxa,df2_dxa,df3_dxa,df4_dxa,df5_dxa,df6_dxa = f(ic,0.,a,r,s,g,p)
    sol = odeint(f,ic[:6],x,args=(a,r,s,g,p),atol=1e-18,rtol=1e-13)
    outer_bc = np.array([sol[:,0][-1],sol[:,1][-1],sol[:,2][-1],sol[:,3][-1],sol[:,4][-1],sol[:,5][-1]])
    df1_dxb,df2_dxb,df3_dxb,df4_dxb,df5_dxb,df6_dxb = f(outer_bc,2.0*np.pi ,a,r,s,g,p)
    
    # Minimizing derivative of f forces both f and f' to be continuous. We do compare first and last elements because
    # each function is periodic. 
    condition = [df1_dxb-df1_dxa, df2_dxb-df2_dxa, df3_dxb-df3_dxa, df4_dxb-df4_dxa, df5_dxb-df5_dxa, df6_dxb-df6_dxa]  

    return condition


def f_soln(a,r,s,g,p,ngrid=100,method='lm',ic_guess=[],bounds=([-np.infty]*6,[np.inf]*6)):
    
    # If we haven't supplied a guess for the inital conditions, use the Taylor expanded formulae
    if len(ic_guess) == 0: ic_guess = f_appx(a,r,s,g,p)

    # Calculate Epicyclic Frequency (trivially 1 for Keplerian orbits)
    k = kappa(r)
    
    # f1,...,f6 are defined from 0 to 2pi
    x = np.linspace(0, 2.0*np.pi, ngrid)
    dx = np.average(x[1:] - x[:-1])
    
    # Find correct ICs to satisfy BC condition by minimizing least squares condition.
    ya = least_squares(objective,ic_guess,args=(x,a,r,s,g,p),bounds=bounds,method=method,xtol=1e-11,ftol=1e-11)
    ic = np.copy(ya.x)
    
    # Calculate integrated equations f1,f2,f3,f4,f5,f6 as a function of x (phi)
    sol = odeint(f,ic[:6],x,args=(a,r,s,g,p),atol=1e-11,rtol=1e-11)
    
    # Normalization condition is <f_6> = 1
    if (np.abs(np.average(sol[:,5]))==0.0):
	print "Avg f6: ", np.average(sol[:,5])
    else:
	sol[:,5] = sol[:,5]/np.average(sol[:,5])
    
    return x,sol

## Make empty arrays to load Q1,Q2,Q3

Q1_send = np.zeros((npts,npts))
Q2_send = np.zeros((npts,npts))
Q3_send = np.zeros((npts,npts))

if (rank == 0):
    Q1_recv = np.zeros((npts,npts))
    Q2_recv = np.zeros((npts,npts))
    Q3_recv = np.zeros((npts,npts))
else:
    Q1_recv = None
    Q2_recv = None
    Q3_recv = None

## Do MPI stuff

perrank = npts//size
comm.Barrier()

## Iterate


# 1D: iterate only through psi array
for i in range(rank*perrank, (rank+1)*perrank):
    r = r_arr[i]
    print "rank = ", rank, "r = ", r, "\n"
    ic_guess = []
    for j,s in enumerate(s_arr):

        # Get solution from kernel for current psi value
        x, sol = f_soln(a,r,s,g,p,ngrid=ngpts,method='lm',ic_guess=ic_guess)

        # f1(0)...f6(0) form our guess for the initial conditions of the next iteration
        f10,f20,f30,f40,f50,f60 = sol[:,0][0],sol[:,1][0],sol[:,2][0],sol[:,3][0],sol[:,4][0],sol[:,5][0]
        ic_guess = [f10,f20,f30,f40,f50,f60]

        # Calculate Q1,Q2,Q3 for the given value of psi
        Q1_send[i][j],Q2_send[i][j],Q3_send[i][j] = Q(x,sol,a,r,s,p)

comm.Reduce([Q1_send,MPI.DOUBLE],[Q1_recv,MPI.DOUBLE],op=MPI.SUM,root=0)
comm.Reduce([Q2_send,MPI.DOUBLE],[Q2_recv,MPI.DOUBLE],op=MPI.SUM,root=0)
comm.Reduce([Q3_send,MPI.DOUBLE],[Q3_recv,MPI.DOUBLE],op=MPI.SUM,root=0)

# Currently, Q1,Q2,Q3 are saved only as numpy arrays. 
if rank == 0:
    np.savetxt(fn1 + ".txt",Q1_recv)
    np.savetxt(fn2 + ".txt",Q2_recv)
    np.savetxt(fn3 + ".txt",Q3_recv)
