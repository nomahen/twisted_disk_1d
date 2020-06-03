"""
1D evolution of twisted accretion disks

Finite volume method
"""

import matplotlib.pyplot as plt
import numpy as np
cimport numpy as np
cimport cython
from libc.stdio cimport printf,fopen,fclose,fprintf,FILE,sprintf
from libc.stdlib cimport malloc
from libc.math cimport fmin, fmax, fabs, isnan, sin, cos
from libc.string cimport memcpy

# If I want to time things
cdef extern from "time.h":
    ctypedef unsigned long clock_t
    cdef clock_t clock()
    cdef enum:
        CLOCKS_PER_SEC

## Helper functions ##

def load_Q(path,dim="1d"):
    data = open(path,"r")
    parsed = []
    if (dim == "1d"):
        for line in data:
            parsed.append(np.array(line.split()).astype(float))
        return np.array(parsed)[:,0]
    elif (dim == "2d"):
        for row in data:
            cols = row.split()
            row_parsed = []
            for col in cols:
                row_parsed.append(float(col))
            row_parsed = np.array(row_parsed)
            parsed.append(row_parsed)
        return np.array(parsed)
    else:
        print "Error! dim has to equal \"1d\" or \"2d\"!"

## Functions used in iteration ##

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double[:] apply_outflow(double[:] vec, int ngrid):
    vec[0] = 1.0 * vec[2]
    vec[1] = 1.0 * vec[2]
    vec[ngrid-1] = 1.0 * vec[ngrid-3]
    vec[ngrid-2] = 1.0 * vec[ngrid-3]
    return vec

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double[:] apply_zero(double[:] vec, int ngrid):
    vec[0] = 0.0
    vec[1] = 0.0
    vec[ngrid-1] = 0.0
    vec[ngrid-2] = 0.0
    return vec

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double minmod(double a, double b):
    if   ( (fabs(a) < fabs(b)) and (a*b > 0.) ):
            return a
    elif ( (fabs(b) < fabs(a)) and (a*b > 0.) ):
            return b
    else: 
            return 0.

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double interp_1d(double[:] x, double[:] y, double new_x, int nx):
    cdef int i
    cdef int ind=nx-1
    cdef double new_y
    for i in range(nx):
        if (new_x == x[i]):
            new_y = y[i]
            return new_y
        if (new_x < x[i]):
            ind = i
            break
    new_y = (y[ind-1]*(x[ind]-new_x) + y[ind]*(new_x - x[ind-1]))/(x[ind]-x[ind-1])
    return new_y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double interp_2d(double[:] x, double[:] y, double[:,:] z, double new_x, double new_y, int n):
    cdef int i,xind=n-1,yind=n-1
    cdef double new_z,z11,z12,z21,z22
    cdef double x1,x2,y1,y2

    # get x1,x2,y1,y2 which define the "box" around our x,y
    # if x, y is exactly an x,y value from our array, we can skip the rest and do a 1d interpolation
    for i in range(n):
        if (new_x == x[i]):
            return interp_1d(y,z[i,:],new_y,n) 
        elif (new_x <= x[i]):
            xind = i
            break
    for i in range(n):
        if (new_y == y[i]):
            return interp_1d(x,z[:,i],new_x,n) 
        elif (new_y <= y[i]):
            yind = i
            break
    # Get z values corresponding to each pair (x_i, y_i) that defines our box
    x1 = x[xind-1]
    x2 = x[xind]
    y1 = y[yind-1]
    y2 = y[yind]
    z22 = z[xind,yind]
    z12 = z[xind-1,yind]
    z21 = z[xind,yind-1]
    z11 = z[xind-1,yind-1]

    # Now apply formula for bilinear interpolation
    return (1./((x2-x1)*(y2-y1)))*(z11*(x2-new_x)*(y2-new_y) + z21*(new_x-x1)*(y2-new_y) + z12*(x2-new_x)*(new_y-y1) + z22*(new_x-x1)*(new_y-y1))


## main code body

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def evolve(*p):
    ## Get params 
    cdef double alpha, gamma, HoR, tilt, bhspin, r0, rw, rmin, rmax, tmax, smax, cfl, io_freq
    cdef int    ngrid, bc_type, dim_type
    cdef int    ngc = 2 # number of guard cells
    cdef bint   dolog
    cdef char*  io_prefix
    # for simple setup
    cdef int    space_order, time_order, eq_type
    alpha,gamma,HoR,tilt,bhspin,r0,rw,rmin,rmax,rho_type,tmax,cfl,ngrid,bc,io_freq,io_prefix,Q_dim,smax,rmax_Q,Q1_path,Q2_path,Q3_path,space_order,time_order,which_problem = p
 
    xmin = 2.0
    xmax = 3.0
    _dx  = (xmax-xmin)/ngrid
    _x   = np.linspace(xmin-1.5*_dx,xmax+1.5*_dx,ngrid+2*ngc) 

    _Lz = np.zeros(ngrid+2*ngc)
    if (which_problem=="pulse"):
        # width 0.2, centered at 1.5
        tmax = 0.1 # for outflow BCs, want to lower tmax, otherwise Lz just leaves domain by tmax 
        _Lz = (1.0/(0.2*np.sqrt(2.0*np.pi))) * (np.exp(-0.5*((_x - 2.5)**2./0.2)) + 1.0)
        bc = "outflow"
    if (which_problem=="discontinuity"):
        tmax = 0.1 # for outflow BCs, want to lower tmax, otherwise Lz just leaves domain by tmax 
        _Lz = np.ones(ngrid+2*ngc)
        _Lz[_x>2.5] += 1.0
        bc = "outflow"
    if (which_problem=="steady"):
        # width 0.2, centered at 1.5
        tmax = 1.0 # evolve to full viscous time
        _Lz = (1.0/(0.2*np.sqrt(2.0*np.pi))) * (np.exp(-0.5*((_x - 2.5)**2./0.2)) + 1.0)
        bc = "infinite"  
 
 
    # make bc_type
    if (bc == 'sink'):
        bc_type = 0
    elif (bc == 'outflow'):
        bc_type = 1
    elif (bc == 'mix'):
        bc_type = 2
    elif (bc == 'infinite'):
        bc_type = 3 
    else:
        print "Error! bc needs to be set to \"sink\", \"outflow\", \"mix\" or \"infinite\"! Exiting"
        exit()
  
    ########
 
    ## Do initial input/output
 
    print "#### Parameters ####\n"
    print "tmax        = %s \n" % tmax
    print "cfl         = %s\n" % cfl
    print "bc          = %s\n" % bc
    print "io_freq     = %s [tmax]\n" % io_freq
    print "io_prefix   = %s\n" % io_prefix
    print "space_order = %d\n" % space_order
    print "time_order  = %d\n" % time_order
    print "####################\n\n"
    print "Beginning simulation...\n\n"
 
    ########
 
    #############
    ## Evolve! ##
    #############

    # Initialize Cython stuff for iteration
    cdef double[:] L  = np.zeros(ngrid+2*ngc)
    cdef double[:] Lx   = np.zeros(ngrid+2*ngc)
    cdef double[:] Ly   = np.zeros(ngrid+2*ngc)
    cdef double[:] Lz   = _Lz 

    # Cell interfaces 
    cdef double[:] Lx_L      = np.zeros(ngrid+2*ngc)
    cdef double[:] Lx_R      = np.zeros(ngrid+2*ngc)
    cdef double[:] Ly_L      = np.zeros(ngrid+2*ngc)
    cdef double[:] Ly_R      = np.zeros(ngrid+2*ngc)
    cdef double[:] Lz_L      = np.zeros(ngrid+2*ngc)
    cdef double[:] Lz_R      = np.zeros(ngrid+2*ngc)
    cdef double[:] L_L       = np.zeros(ngrid+2*ngc)
    cdef double[:] L_R       = np.zeros(ngrid+2*ngc)

    # Fluxes (interior cell interfaces only only)
    cdef double[:] F_z       = np.zeros(ngrid+1)

    # miscellaneous variables
    cdef double tmp_slope # for data reconstruction
    cdef int i
    cdef double[:] Lz_inf = np.copy(_Lz[ngrid+2*ngc-2:ngrid+2*ngc])
    cdef double[:] Lz_old = np.copy(_Lz)

    # io variables
    cdef FILE      *f_out
    cdef char[40]  io_fn
    cdef int       io_cnt = 0, nstep = 0
    cdef int       predictor = 1
    cdef double    t = 0.

    # for simple problem
    cdef double[:] x = _x
    cdef double[:] Q1 = np.ones(ngrid+2*ngc) * (-1.5*alpha) # this is the taylor expansion for zero warp
    cdef double[:] Q2 = np.zeros(ngrid+2*ngc)
    cdef double[:] Q3 = np.zeros(ngrid+2*ngc)


    # for grid
    cdef double dx = _dx
    cdef double dt = 1000000000000000000.
    cdef double nu = fabs(2.*Q1[0])*HoR**2.
    cdef double vel = fabs(Q1[0])*HoR**2.
    cdef double t_viscous = (xmax-xmin)**2. / nu
    dt  = fabs(cfl*(dx/vel)/(1. + 2*nu/dx/vel))
    tmax *= t_viscous

    ### Do initial outputs
    printf("Doing initial output...\n")
    sprintf(io_fn,"%s%d.csv",io_prefix,io_cnt)
    f_out = fopen(io_fn,"w")
    for i in range(ngrid+2*ngc):
        fprintf(f_out, "%e ", Lx[i])
        fprintf(f_out, "%e ", Ly[i])
        fprintf(f_out, "%e ", Lz[i])
        fprintf(f_out, "%e ", x[i])#r[i])
        fprintf(f_out, "%e ", Q1[i])
        fprintf(f_out, "%e ", Q2[i])
        fprintf(f_out, "%e ", Q3[i])
        fprintf(f_out, "%e ", t)
        fprintf(f_out, "\n")
    fclose(f_out)
    io_cnt += 1

    # iterate!
    while (t < tmax):
 
        ### Reconstruct, Evolve, Average algorithm
        ## reconstruct
        for i in range(ngrid+2*ngc):
            if ((space_order==1) or (i==0) or (i==(ngrid+2*ngc-1))): # do piecewise constant reconstruction for outer-most guard cells
                tmp_slope = 0.0
            if (space_order==2):
                tmp_slope = minmod((Lz[i]-Lz[i-1])/dx,(Lz[i+1]-Lz[i])/dx)
            Lz_L[i] = Lz[i] - 0.5*dx*tmp_slope
            Lz_R[i] = Lz[i] + 0.5*dx*tmp_slope


            L_L[i]  = (Lx_L[i]**2. + Ly_L[i]**2. + Lz_L[i]**2.)**0.5
            L_R[i]  = (Lx_R[i]**2. + Ly_R[i]**2. + Lz_R[i]**2.)**0.5


        ## evolve (get fluxes)
        for i in range(ngrid+1):
            # advective Q1 term
            F_z[i] = (-1.0) * Q1[i] * 0.5*(Lz_L[i+2]+Lz_R[i+1])

            # diffusive Q1 term
            F_z[i] += 2.*Q1[i] * ( (Lz[i+2] - Lz[i+1])/dx ) * 0.5*(Lz_L[i+2]/L_L[i+2] + Lz_L[i+1]/L_L[i+2])  

            # Apply HoR factor to everything
            F_z[i] *= (HoR)**2.

        ## update
        for i in range(2,ngrid+ngc):
            if (time_order==2):
                if predictor:
                    Lz[i] = Lz_old[i] - 0.5*(dt/dx)*(F_z[i-1] - F_z[i-2])
                else:
                    Lz[i] = Lz_old[i] - (dt/dx)*(F_z[i-1] - F_z[i-2])
            elif (time_order==1):
                Lz[i] = Lz_old[i] - (dt/dx)*(F_z[i-1] - F_z[i-2])
                


        ## apply boundary conditions
        ## Weight by R so sigma is continuous! L should have profile at boundaries
        ## Two guard cells on each end
        if (bc_type==1): #### Apply outflow boundary conditions
            ## Lz
            Lz[ngrid+2*ngc-1] = Lz[ngrid-1+ngc]
            Lz[ngrid+2*ngc-2] = Lz[ngrid-1+ngc]
            Lz[0] = Lz[2]
            Lz[1] = Lz[2]

        elif (bc_type==3): #### Infinite disk
            ### Outer boundary conditions are Dirichlet (set by user):
            ## Lz
            Lz[ngrid+2*ngc-1] = Lz_inf[1]
            Lz[ngrid+2*ngc-2] = Lz_inf[0]
            Lz[0] = Lz[2]
            Lz[1] = Lz[2]
            
        if (predictor and (time_order==2)): # predictor step
            predictor = 0 # next while loop iteration due to the full update
        else: # corrector step
            #### Update timestep and do outputs
            t += dt
            nstep += 1

            # Update old quantities; these dont change during predictor stage
            for i in range(ngrid+2*ngc):
                Lz_old[i] = Lz[i]
            predictor = 1 # next while loop iteration do predictor half time step update

            #### Print outputs
            if ((t%(io_freq*t_viscous) < dt)):
                printf("t/tmax = %e, dt/tmax = %e, io_cnt = %d\n",t/tmax,dt/tmax,io_cnt)
                sprintf(io_fn,"%s%d.csv",io_prefix,io_cnt)

                f_out = fopen(io_fn,"w")
                for i in range(ngrid+2*ngc):
                    fprintf(f_out, "%e ", Lx[i])
                    fprintf(f_out, "%e ", Ly[i])
                    fprintf(f_out, "%e ", Lz[i])
                    fprintf(f_out, "%e ", x[i])
                    fprintf(f_out, "%e ", Q1[i])
                    fprintf(f_out, "%e ", Q2[i])
                    fprintf(f_out, "%e ", Q3[i])
                    fprintf(f_out, "%e ", t)
                    fprintf(f_out, "\n")
                fclose(f_out)
                io_cnt += 1 

    ### Do final outputs
    printf("Doing final output...\n")
    sprintf(io_fn,"%s%d.csv",io_prefix,io_cnt)
    f_out = fopen(io_fn,"w")
    for i in range(ngrid+2*ngc):
        fprintf(f_out, "%e ", Lx[i])
        fprintf(f_out, "%e ", Ly[i])
        fprintf(f_out, "%e ", Lz[i])
        fprintf(f_out, "%e ", x[i])#r[i])
        fprintf(f_out, "%e ", Q1[i])
        fprintf(f_out, "%e ", Q2[i])
        fprintf(f_out, "%e ", Q3[i])
        fprintf(f_out, "%e ", t)
        fprintf(f_out, "\n")
    fclose(f_out)
