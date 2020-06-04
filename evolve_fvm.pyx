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
 
    ## currently, the following doesnt do anything
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
    ## Lets read in our Q coefficients
    Q1_parsed = load_Q(Q1_path)
    Q2_parsed = load_Q(Q2_path)
    Q3_parsed = load_Q(Q3_path)
    cdef int ng_Q = len(Q1_parsed) # Get length of Q
    ## Psi array that Q1/Q2/Q3 correspond to
    _s_arr = np.logspace(0,np.log10(smax+1),ng_Q) - 1
    ## Now make cython arrays for each Q coefficient and the corresponding psi array so we can interpolate from it in the code. 
    cdef double[:] Q1_arr = np.log10(-Q1_parsed + small)
    cdef double[:] Q2_arr = np.log10(Q2_parsed + small)
    cdef double[:] Q3_arr = np.log10(Q3_parsed + small)
    cdef double[:] s_arr = _s_arr
    ########

  
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
    cdef double[:] Lx   = np.zeros(ngrid+2*ngc)
    cdef double[:] Ly   = np.zeros(ngrid+2*ngc)
    cdef double[:] Lz   = _Lz 
    cdef double[:] L  = np.copy(_Lz)
    cdef double[:] lx   = np.zeros(ngrid+2*ngc)
    cdef double[:] ly   = np.zeros(ngrid+2*ngc)
    cdef double[:] lz   = _Lz/np.copy(_Lz) 

    # Cell interface reconstructions
    cdef double[:] Lx_L      = np.zeros(ngrid+2*ngc)
    cdef double[:] Lx_R      = np.zeros(ngrid+2*ngc)
    cdef double[:] Ly_L      = np.zeros(ngrid+2*ngc)
    cdef double[:] Ly_R      = np.zeros(ngrid+2*ngc)
    cdef double[:] Lz_L      = np.zeros(ngrid+2*ngc)
    cdef double[:] Lz_R      = np.zeros(ngrid+2*ngc)
    cdef double[:] L_L       = np.zeros(ngrid+2*ngc)
    cdef double[:] L_R       = np.zeros(ngrid+2*ngc)

    # Gradients or gradient-dependent quantities
    cdef double[:] psi       = np.zeros(ngrid+2*ngc)
    cdef double[:] Q1        = np.zeros(ngrid+2*ngc)#np.ones(ngrid+2*ngc) * (-1.5*alpha) # this is the taylor expansion for zero warp
    cdef double[:] Q2        = np.zeros(ngrid+2*ngc)
    cdef double[:] Q3        = np.zeros(ngrid+2*ngc)

    # Fluxes; only interior cell faces are used. Here, F_[i] corresponds to the *left* side. 
    cdef double[:] F_x       = np.zeros(ngrid+2*ngc)
    cdef double[:] F_y       = np.zeros(ngrid+2*ngc)
    cdef double[:] F_z       = np.zeros(ngrid+2*ngc)

    # miscellaneous variables
    cdef double tmp_slope # for data reconstruction
    cdef double small = 1e-10 # just a real small number! for avoiding 1/0 errors. 
    cdef int i
    cdef double[:] Lx_old = np.zeros(ngrid+2*ngc)#np.copy(_Lx)
    cdef double[:] Ly_old = np.zeros(ngrid+2*ngc)#np.copy(_Ly)
    cdef double[:] Lz_old = np.copy(_Lz)

    # io variables
    cdef FILE      *f_out
    cdef char[40]  io_fn
    cdef int       io_cnt = 0, nstep = 0
    cdef int       predictor = 1
    cdef double    t = 0.

    # for grid
    cdef double[:] x = _x
    cdef double dx = _dx
    cdef double dt = 1000000000000000000.

    # for dt and tmax determination. Calculate zero-warp viscosity coefficient to determine "viscous time" of disk. 
    Q1_tmp = -1.0*10**(interp_1d(s_arr,Q1_arr,0,ng_Q))
    cdef double nu = fabs(2.*Q1_tmp)*HoR**2.
    cdef double vel = fabs(Q1_tmp)*HoR**2.
    cdef double t_viscous = (xmax-xmin)**2. / nu

    # determine initial dt from wave velocity and viscosity
    dt  = fabs(cfl*(dx/vel)/(1. + 2*nu/dx/vel))
    tmax *= t_viscous


    #################################
    #################################
    ########## simulate :) ##########
    #################################
    #################################

    ########################
    ## do initial outputs ##
    ########################
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

    ######################
    ## begin iteration! ##
    ######################
    while (t < tmax):
 
        #########################
        ## data reconstruction ##
        #########################
        for i in range(ngrid+2*ngc):

            ## For Lx/Ly/Lz, calculate the cell slopes based on the desired spatial order of the scheme, and reconstruct the corresponding structure in the cell
            # Lx
            if ((space_order==1) or (i==0) or (i==(ngrid+2*ngc-1))): # do piecewise constant reconstruction for outer-most guard cells
                tmp_slope = 0.0
            if (space_order==2):
                tmp_slope = minmod((Lx[i]-Lx[i-1])/dx,(Lx[i+1]-Lx[i])/dx)
            Lx_L[i] = Lx[i] - 0.5*dx*tmp_slope
            Lx_R[i] = Lx[i] + 0.5*dx*tmp_slope
            # Ly
            if ((space_order==1) or (i==0) or (i==(ngrid+2*ngc-1))): # do piecewise constant reconstruction for outer-most guard cells
                tmp_slope = 0.0
            if (space_order==2):
                tmp_slope = minmod((Ly[i]-Ly[i-1])/dx,(Ly[i+1]-Ly[i])/dx)
            Ly_L[i] = Ly[i] - 0.5*dx*tmp_slope
            Ly_R[i] = Ly[i] + 0.5*dx*tmp_slope
            # Lz
            if ((space_order==1) or (i==0) or (i==(ngrid+2*ngc-1))): # do piecewise constant reconstruction for outer-most guard cells
                tmp_slope = 0.0
            if (space_order==2):
                tmp_slope = minmod((Lz[i]-Lz[i-1])/dx,(Lz[i+1]-Lz[i])/dx)
            Lz_L[i] = Lz[i] - 0.5*dx*tmp_slope
            Lz_R[i] = Lz[i] + 0.5*dx*tmp_slope
            # Get magnitude of L
            L_L[i]  = (Lx_L[i]**2. + Ly_L[i]**2. + Lz_L[i]**2.)**0.5
            L_R[i]  = (Lx_R[i]**2. + Ly_R[i]**2. + Lz_R[i]**2.)**0.5

            ## For any gradient quantities, evaluate at cell interfaces here. 
            ## Only need to calculate for the ngrid+1 interfaces of the cell interfaces
            if not((i==0) or (i==1) or (i==(ngrid+2*ngc-1))):
                # warp parameter psi
                psi[i] = ((lx[i]-lx[i-1])**2. + (ly[i]-ly[i-1])**2. + (lz[i]-lz[i-1])**2.)**(0.5) / dx

                # Given warmp amplitude psi[i], Q coefficients must be interpolated from arrays for psi ("s_arr") and Q ("Q1,2,3_arr"), which are of length "ng_Q"
                Q1[i]      = (-1.0*10**(interp_1d(s_arr,Q1_arr,psi[i],ng_Q)))
                Q2[i]      = 10**(interp_1d(s_arr,Q2_arr,psi[i],ng_Q))
                Q3[i]      = 10**(interp_1d(s_arr,Q3_arr,psi[i],ng_Q))



        ######################
        ## calculate fluxes ##
        ######################

        for i in range(ngc,ngrid+ngc+1):
            # advective Q1 term
            F_z[i] = (-1.0) * Q1[i]

            # diffusive Q1 term
            F_z[i] += 2.*Q1[i] * ( (Lz[i] - Lz[i-1])/dx ) / (0.5 * (L[i] + L[i-1])) 

            # upwind the fluxes
            if (F_z[i] >= 0.): 
                F_z[i] *= Lz_R[i-1]
            else:
                F_z[i] *= Lz_L[i]

            # Apply HoR factor to everything
            F_z[i] *= (HoR)**2.

        #####################
        ## update solution ##
        #####################

        for i in range(ngc,ngrid+ngc):
            if (time_order==2):
                if predictor:
                    Lz[i] = Lz_old[i] - 0.5*(dt/dx)*(F_z[i+1] - F_z[i])
                else:
                    Lz[i] = Lz_old[i] - (dt/dx)*(F_z[i+1] - F_z[i])
            elif (time_order==1):
                Lz[i] = Lz_old[i] - (dt/dx)*(F_z[i+1] - F_z[i])

            L[i] = (Lx[i]**2. + Ly[i]**2. + Lz[i]**2.)**0.5
            lx[i] = Lx[i]/L[i]
            ly[i] = Ly[i]/L[i]
            lz[i] = Lz[i]/L[i]
                

        ###############################
        ## apply boundary conditions ##
        ###############################

        ## Two guard cells on each end
        if (bc_type==1): #### Apply outflow boundary conditions
            ## Lx
            Lx[ngrid+2*ngc-1] = Lx[ngrid-1+ngc]
            Lx[ngrid+2*ngc-2] = Lx[ngrid-1+ngc]
            Lx[0] = Lx[2]
            Lx[1] = Lx[2]
            ## Ly
            Ly[ngrid+2*ngc-1] = Ly[ngrid-1+ngc]
            Ly[ngrid+2*ngc-2] = Ly[ngrid-1+ngc]
            Ly[0] = Ly[2]
            Ly[1] = Ly[2]
            ## Lz
            Lz[ngrid+2*ngc-1] = Lz[ngrid-1+ngc]
            Lz[ngrid+2*ngc-2] = Lz[ngrid-1+ngc]
            Lz[0] = Lz[2]
            Lz[1] = Lz[2]

        elif (bc_type==3): #### Infinite disk
            ### Outer boundary conditions are unchanged from initial conditions; inner BCs are outflow.
            ## Lx
            Lx[0] = Lx[2]
            Lx[1] = Lx[2]
            ## Ly
            Ly[0] = Ly[2]
            Ly[1] = Ly[2]
            ## Lz
            Lz[0] = Lz[2]
            Lz[1] = Lz[2]

        ## Also, we want magnitude and unit vectors calculated in our BC, but these are independent of bc_type.
        # L
        L[0] = (Lx[0]**2. + Ly[0]**2. + Lz[0]**2.)**0.5
        L[1] = (Lx[1]**2. + Ly[1]**2. + Lz[1]**2.)**0.5
        L[ngrid+2*ngc-1] = (Lx[ngrid+2*ngc-1]**2. + Ly[ngrid+2*ngc-1]**2. + Lz[ngrid+2*ngc-1]**2.)**0.5
        L[ngrid+2*ngc-2] = (Lx[ngrid+2*ngc-2]**2. + Ly[ngrid+2*ngc-2]**2. + Lz[ngrid+2*ngc-2]**2.)**0.5
        # lx
        lx[0] = Lx[0]/L[0]
        lx[1] = Lx[1]/L[1]
        lx[ngrid+2*ngc-1] = Lx[ngrid+2*ngc-1]/L[ngrid+2*ngc-1]
        lx[ngrid+2*ngc-2] = Lx[ngrid+2*ngc-2]/L[ngrid+2*ngc-2]
        # ly
        ly[0] = Ly[0]/L[0]
        ly[1] = Ly[1]/L[1]
        ly[ngrid+2*ngc-1] = Ly[ngrid+2*ngc-1]/L[ngrid+2*ngc-1]
        ly[ngrid+2*ngc-2] = Ly[ngrid+2*ngc-2]/L[ngrid+2*ngc-2]
        # lz
        lz[0] = Lz[0]/L[0]
        lz[1] = Lz[1]/L[1]
        lz[ngrid+2*ngc-1] = Lz[ngrid+2*ngc-1]/L[ngrid+2*ngc-1]
        lz[ngrid+2*ngc-2] = Lz[ngrid+2*ngc-2]/L[ngrid+2*ngc-2]

        ############################
        ## end of iteration calls ##
        ############################
            
        if (predictor and (time_order==2)): # predictor step
            predictor = 0 # next while loop iteration due to the full update
        else: # corrector step
            #### Update timestep and do outputs
            t += dt
            nstep += 1

            # Update old quantities; these dont change during predictor stage
            for i in range(ngrid+2*ngc):
                Lx_old[i] = Lx[i]
                Ly_old[i] = Ly[i]
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

    ######################
    ## do final outputs ##
    ######################
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
