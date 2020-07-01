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
cdef inline double mc(double a, double b):
    if (a*b > 0.):
        return fmin(fmin(fabs(a),fabs(b)), 0.25*fabs(a+b)) 
    else:
        return 0.

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double interp_1d(double[:] x, double[:] y, double new_x, int nx):
    cdef int i
    cdef int ind=nx-1
    cdef double new_y
    if (new_x >= x[nx-1]): # If new_x is outside of bounds, extrapolate.  
        return y[nx-1] + ((y[nx-1]-y[nx-2])/(x[nx-1]-x[nx-2])) * (new_x - x[nx-1])

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
    cdef double alpha, gamma, HoR, tilt, bhspin, r0, rw, rmin, rmax, tmax, smax, cfl, io_freq, soft
    cdef int    ngrid, bc_type, dim_type
    cdef int    ngc = 2 # number of guard cells
    cdef bint   dolog
    cdef char*  io_prefix
    # for simple setup
    cdef int    space_order, time_order, eq_type
    alpha,gamma,HoR,tilt,bhspin,r0,rw,rmin,rmax,distr_type,tmax,cfl,ngrid,bc,io_freq,io_prefix,Q_dim,smax,rmax_Q,Q1_path,Q2_path,Q3_path,space_order,time_order,which_problem,soft = p
 
    xmin = np.log(rmin)
    xmax = np.log(rmax)
    _dx  = (xmax-xmin)/ngrid
    _x   = np.linspace(xmin-1.5*_dx,xmax+1.5*_dx,ngrid+2*ngc) 
    _r   = np.exp(_x)
    tilt *= np.pi/180. 



    _Lx = np.zeros(ngrid+2*ngc)
    _Ly = np.zeros(ngrid+2*ngc)
    # calculate kernel for density distruption
    if (distr_type == "flat_rho"):          # Flat distribution in surface density (L/r ~ sigma)
        _Lz = _r * np.ones(ngrid+2*ngc)  
    if (distr_type == "flat_am"):          # Flat distribution in code angular momentum
        _Lz = np.ones(ngrid+2*ngc)  
    elif (distr_type == "gauss_rho"):   # Gaussian in surface density (L/r ~ Sigma), centered in r
        midpoint = np.exp(np.median(_x))
        width    = 0.1 * (rmax-rmin)
        _Lz = _r * (1.0/(width*np.sqrt(2.0*np.pi))) * (np.exp(-0.5*((_r -  midpoint)**2./width)) + 1.0)
    elif (distr_type == "gauss_am"):    # Gaussian in code angular momentum, centered in _x
        midpoint = np.median(_x)
        width    = 0.1 * (xmax-xmin)
        _Lz = (1.0/(width*np.sqrt(2.0*np.pi))) * (np.exp(-0.5*((_x - midpoint)**2./width)) + 1.0)
    elif (distr_type == "nixon_king"): # NOTE: This one is special! it overwrites xmin,xmax,etc to reproduce nixon and king 2012 setup. 
        rmin = 60.0   
        rmax = 6000.0 
        xmin = np.log(rmin)
        xmax = np.log(rmax)
        _dx  = (xmax-xmin)/ngrid
        _x   = np.linspace(xmin-1.5*_dx,xmax+1.5*_dx,ngrid+2*ngc) 
        _r   = np.exp(_x)
        midpoint = 3000.0
        width    = 300.0
        _Lz = _r * (1.0/(width*np.sqrt(2.0*np.pi))) * (np.exp(-0.5*((_r -  midpoint)**2./width)) + 1.0)
    else:
        print "Error! distr_type needs to be set to \"flat_rho\", \"flat_am\", \"gauss_rho\" or \"gauss_am\"! Exiting"



    if (which_problem=="discontinuity"):
        _Lz[_x>np.median(_x)] += 1.0
    if (which_problem=="discontinuity_tilt"):
        tilt_profile = np.zeros(ngrid+2*ngc)
        tilt_profile[_x>np.median(_x)] += tilt
        _L  = np.copy(_Lz)
        _Lz = np.cos(tilt_profile) * _L
        _Lx = np.sin(tilt_profile) * _L  
    if (which_problem=="twist"):
        tilt_profile = tilt * ((_x-_x[0])/(_x[ngrid+2*ngc-1]-_x[0])) # add linear twist
        _L  = np.copy(_Lz)
        _Lz = np.cos(tilt_profile) * _L
        _Lx = np.sin(tilt_profile) * _L
    if (which_problem=="flat"):
        tilt_profile = tilt * np.ones(ngrid+2*ngc)  
        _L  = np.copy(_Lz)
        _Lz = np.cos(tilt_profile) * _L
        _Lx = np.sin(tilt_profile) * _L
    # get magnitude of angular momentum
    _L = (_Lx**2. + _Ly**2. + _Lz**2.)**(0.5)

 
    ## currently, the following doesnt do anything
    # make bc_type
    if (bc == 'outflow'):
        bc_type = 1
    elif (bc == 'infinite'):
        bc_type = 2 
    elif (bc == 'outflow_alt'):
        bc_type = 3
    elif (bc == 'infinite_alt'):
        bc_type = 4
    else:
        print "Error! bc needs to be set to \"outflow\", \"infinite\", \"outflow_alt\" or \"infinite_alt\"! Exiting"
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
    print "io_freq     = %s \n" % io_freq
    print "io_prefix   = %s\n" % io_prefix
    print "space_order = %d\n" % space_order
    print "time_order  = %d\n" % time_order
    print "####################\n\n"
    print "Beginning simulation...\n\n"
 
    ########
 
    #############
    ## Evolve! ##
    #############

    # avoid divide by zero error
    cdef double small = 1e-10 # just a real small number! for avoiding 1/0 errors. 

    # Initialize Cython stuff for iteration
    cdef double[:] Lx   = _Lx
    cdef double[:] Ly   = _Ly
    cdef double[:] Lz   = _Lz 
    cdef double[:] L    = _L
    cdef double[:] lx   = _Lx/(_L+small)
    cdef double[:] ly   = _Ly/(_L+small)
    cdef double[:] lz   = _Lz/(_L+small)

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

    # Lense-Thirring frequency; assume bh spin is in z direction. 
    cdef double[:] omega_p   = ( 2. * bhspin / _r**(3.) ) 

    # miscellaneous variables
    cdef double tmp_slope # for data reconstruction
    cdef double tmp_flux  # for flux calculation
    cdef double Lx_tmp,Ly_tmp,Lz_tmp # for source terms
    cdef int i
    cdef double[:] Lx_old = np.copy(_Lx)
    cdef double[:] Ly_old = np.copy(_Ly)
    cdef double[:] Lz_old = np.copy(_Lz)

    # io variables
    cdef FILE      *f_out
    cdef char[40]  io_fn
    cdef int       io_cnt = 0, nstep = 0
    cdef int       predictor = 1
    cdef double    t = 0.

    # for grid
    cdef double[:] x = _x
    cdef double[:] r = np.exp(_x)
    cdef double dx = _dx
    cdef double dt = 1000000000000000000.

    # for dt and tmax determination. Calculate zero-warp viscosity coefficient to determine "viscous time" of disk. 
    Q1_tmp = -1.0*10**(interp_1d(s_arr,Q1_arr,0,ng_Q))


    if (distr_type == "flat"):
        
        np.argmax(_L/r)
    cdef double nu = fabs(2.*Q1_tmp)*HoR**2 * np.median(_r)**(-3./2.) 
    cdef double vel = fabs(Q1_tmp)*HoR**2.  * np.median(_r)**(-3./2.)
    # Weird calculation: integral for t_viscous in code units, where t ~ (effective viscosity) / (log r) ^2
    cdef double t_viscous = (1.0/fabs(2.*Q1_tmp*HoR**2.))*(np.exp(1.5*xmin)*(1. - 1.5*xmin) + (1.5*xmax - 1.)*np.exp(1.5*xmax) ) / (1.5)**2.#(xmax-xmin)**2. / nu
    printf("t_viscous = %e\,r_g/c\n",t_viscous)

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
                tmp_slope = mc((Lx[i]-Lx[i-1])/dx,(Lx[i+1]-Lx[i])/dx)
            #Lx_L[i] = 0.5*(Lx[i] + Lx[i-1])
            #Lx_R[i] = 0.5*(Lx[i] + Lx[i+1])
            Lx_L[i] = Lx[i] - 0.5*dx*tmp_slope
            Lx_R[i] = Lx[i] + 0.5*dx*tmp_slope
            # Ly
            if ((space_order==1) or (i==0) or (i==(ngrid+2*ngc-1))): # do piecewise constant reconstruction for outer-most guard cells
                tmp_slope = 0.0
            if (space_order==2):
                tmp_slope = mc((Ly[i]-Ly[i-1])/dx,(Ly[i+1]-Ly[i])/dx)
            #Ly_L[i] = 0.5*(Ly[i] + Ly[i-1])
            #Ly_R[i] = 0.5*(Ly[i] + Ly[i+1])
            Ly_L[i] = Ly[i] - 0.5*dx*tmp_slope
            Ly_R[i] = Ly[i] + 0.5*dx*tmp_slope
            # Lz
            if ((space_order==1) or (i==0) or (i==(ngrid+2*ngc-1))): # do piecewise constant reconstruction for outer-most guard cells
                tmp_slope = 0.0
            if (space_order==2):
                tmp_slope = mc((Lz[i]-Lz[i-1])/dx,(Lz[i+1]-Lz[i])/dx)
            #Lz_L[i] = 0.5*(Lz[i] + Lz[i-1])
            #Lz_R[i] = 0.5*(Lz[i] + Lz[i+1])
            Lz_L[i] = Lz[i] - 0.5*dx*tmp_slope
            Lz_R[i] = Lz[i] + 0.5*dx*tmp_slope

            # Get magnitude of L
            L_L[i]  = (Lx_L[i]**2. + Ly_L[i]**2. + Lz_L[i]**2.)**0.5
            L_R[i]  = (Lx_R[i]**2. + Ly_R[i]**2. + Lz_R[i]**2.)**0.5

            ## For any gradient quantities, evaluate at cell interfaces here. 
            ## Don't calculate for i==0;
            if not(i==0):
                # warp parameter psi
                psi[i] = ((lx[i]-lx[i-1])**2. + (ly[i]-ly[i-1])**2. + (lz[i]-lz[i-1])**2.)**(0.5) / (dx + soft)
                #psi[i] = fmin(psi[i],10.)

                # Given warmp amplitude psi[i], Q coefficients must be interpolated from arrays for psi ("s_arr") and Q ("Q1,2,3_arr"), which are of length "ng_Q"
                #if (psi[i] > 10.0): # testing, should make this psi max
                #    Q1[i] = 0.0#Q1_arr[ng_Q-1]
                #    Q2[i] = 0.0#Q2_arr[ng_Q-1]
                #    Q3[i] = 0.0#Q3_arr[ng_Q-1]
                #else:
                Q1[i]      = (-1.0*10**(interp_1d(s_arr,Q1_arr,psi[i],ng_Q)))
                Q2[i]      = 10**(interp_1d(s_arr,Q2_arr,psi[i],ng_Q))
                Q3[i]      = 10**(interp_1d(s_arr,Q3_arr,psi[i],ng_Q))

        ####################
        ## cfl condition  ##
        ####################

        # Calculate if time_order=1 or if on predictor step, calculate timestep
        if ((time_order==1) or (predictor==1)):
            # reinitialize dt to be very large
            dt = 1000000000000. 
            for i in range(1,ngrid+2*ngc):
                vel = (-1.0)*Q1[i]
                vel += 2. * Q1[i] * ( (L[i] - L[i-1])/dx ) / (0.5 * (L[i] + L[i-1] + small))

                # for dQdx
                tmp_slope = mc((Q1[i]-Q1[i-1])/dx,(Q1[i+1]-Q1[i])/dx)
                vel += 2. * tmp_slope

                # Q2
                vel += -2. * Q2[i] * psi[i]**2.
                vel += -1. * Q2[i] * ( (L[i] - L[i-1])/dx ) / (0.5 * (L[i] + L[i-1] + small))

                # All fluxes are effectively multiplied by ( H/R ) **2. * r**(-3./2.) in my formulation of the twisted disk equations.
                vel = (HoR**2.)*fabs(vel)*r[i]**(-3./2.)
                nu  = (HoR**2.)*2.0*fabs((Q1[i]**2. + Q2[i]**2.)**0.5)*r[i]**(-3./2.)
                dt  = fmin(dt,fabs(cfl*(dx/vel)/(1. + 2*nu/dx/vel)))



        ######################
        ## calculate fluxes ##
        ######################

        for i in range(ngc,ngrid+ngc+1):
            # Q1
            tmp_flux = (-1.0) * Q1[i]
            tmp_flux += 2. * Q1[i] * ( (L[i] - L[i-1])/dx ) / (0.5 * (L[i] + L[i-1] + small)) 

            # dQ1dx
            tmp_slope = mc((Q1[i]-Q1[i-1])/dx,(Q1[i+1]-Q1[i])/dx)
            tmp_flux += 2. * tmp_slope

            # Q2
            tmp_flux += -2. * Q2[i] * psi[i]**2.
            tmp_flux += -1. * Q2[i] * ( (L[i] - L[i-1])/dx ) / (0.5 * (L[i] + L[i-1] + small)) 

            # upwind the fluxes
            if (tmp_flux >= 0.): 
                F_x[i] = tmp_flux*Lx_R[i-1]
                F_y[i] = tmp_flux*Ly_R[i-1]
                F_z[i] = tmp_flux*Lz_R[i-1]

                ## Add the Q3 terms. What's the best way to handle them? They're weird, lets just make them consistent with upwinding.
                F_x[i] += -Q3[i] * (1./dx) *  ( Ly_R[i-1] * (lz[i] - lz[i-1]) - Lz_R[i-1] * (ly[i] - ly[i-1]) )
                F_y[i] += -Q3[i] * (1./dx) *  ( Lz_R[i-1] * (lx[i] - lx[i-1]) - Lx_R[i-1] * (lz[i] - lz[i-1]) )
                F_z[i] += -Q3[i] * (1./dx) *  ( Lx_R[i-1] * (ly[i] - ly[i-1]) - Ly_R[i-1] * (lx[i] - lx[i-1]) )
            else:
                F_x[i] = tmp_flux*Lx_L[i]
                F_y[i] = tmp_flux*Ly_L[i]
                F_z[i] = tmp_flux*Lz_L[i]

                ## Add the Q3 terms. What's the best way to handle them? They're weird, lets just make them consistent with upwinding.
                F_x[i] += -Q3[i] * (1./dx) *  ( Ly_L[i] * (lz[i] - lz[i-1]) - Lz_L[i] * (ly[i] - ly[i-1]) )
                F_y[i] += -Q3[i] * (1./dx) *  ( Lz_L[i] * (lx[i] - lx[i-1]) - Lx_L[i] * (lz[i] - lz[i-1]) )
                F_z[i] += -Q3[i] * (1./dx) *  ( Lx_L[i] * (ly[i] - ly[i-1]) - Ly_L[i] * (lx[i] - lx[i-1]) )

            ## Add diffusive terms that dont need upwinding
            F_x[i] += -Q2[i] * ( (Lx[i] - Lx[i-1])/dx )
            F_y[i] += -Q2[i] * ( (Ly[i] - Ly[i-1])/dx )
            F_z[i] += -Q2[i] * ( (Lz[i] - Lz[i-1])/dx )

            # Apply HoR factor to everything
            F_x[i] *= (HoR)**2. * r[i]**(-3./2.)
            F_y[i] *= (HoR)**2. * r[i]**(-3./2.)
            F_z[i] *= (HoR)**2. * r[i]**(-3./2.)

        '''
        # Introduce checks so material doesn't flow into boundaries
        F_x[ngc] = fmin(F_x[ngc],0.0)
        F_y[ngc] = fmin(F_y[ngc],0.0)
        F_z[ngc] = fmin(F_z[ngc],0.0)
        if ((bc_type==1) or (bc_type==3)): # Only check for outer boundary if using outflow (bc type 1 or 3) conditions
            F_x[ngc+ngrid] = fmax(F_x[ngc+ngrid],0.0)
            F_y[ngc+ngrid] = fmax(F_y[ngc+ngrid],0.0)
            F_z[ngc+ngrid] = fmax(F_z[ngc+ngrid],0.0)
        '''

        ############################################
        ## update solution and apply source terms ##
        ############################################

        for i in range(ngc,ngrid+ngc):
            if (time_order==2):
                if predictor:
                    Lx[i] = Lx_old[i] - 0.5*(dt/dx)*(F_x[i+1] - F_x[i])# * r[i]**(-3./2.)
                    Ly[i] = Ly_old[i] - 0.5*(dt/dx)*(F_y[i+1] - F_y[i])# * r[i]**(-3./2.)
                    Lz[i] = Lz_old[i] - 0.5*(dt/dx)*(F_z[i+1] - F_z[i])# * r[i]**(-3./2.)
                else:
                    Lx[i] = Lx_old[i] - (dt/dx)*(F_x[i+1] - F_x[i])# * r[i]**(-3./2.)
                    Ly[i] = Ly_old[i] - (dt/dx)*(F_y[i+1] - F_y[i])# * r[i]**(-3./2.)
                    Lz[i] = Lz_old[i] - (dt/dx)*(F_z[i+1] - F_z[i])# * r[i]**(-3./2.)

                    ## include source terms here ##
                    Lx_tmp = Lx[i]*cos(omega_p[i] * dt) - Ly[i]*sin(omega_p[i] * dt) 
                    Ly_tmp = Lx[i]*sin(omega_p[i] * dt) + Ly[i]*cos(omega_p[i] * dt) 
                    Lx[i]  = 1.0*Lx_tmp
                    Ly[i]  = 1.0*Ly_tmp
            elif (time_order==1):
                Lx[i] = Lx_old[i] - (dt/dx)*(F_x[i+1] - F_x[i])# * r[i]**(-3./2.)
                Ly[i] = Ly_old[i] - (dt/dx)*(F_y[i+1] - F_y[i])# * r[i]**(-3./2.)
                Lz[i] = Lz_old[i] - (dt/dx)*(F_z[i+1] - F_z[i])# * r[i]**(-3./2.)

                ## include source terms here ##
                Lx_tmp = Lx[i]*cos(omega_p[i] * dt) - Ly[i]*sin(omega_p[i] * dt) 
                Ly_tmp = Lx[i]*sin(omega_p[i] * dt) + Ly[i]*cos(omega_p[i] * dt) 
                Lx[i]  = 1.0*Lx_tmp
                Ly[i]  = 1.0*Ly_tmp


            L[i] = (Lx[i]**2. + Ly[i]**2. + Lz[i]**2.)**0.5
            lx[i] = Lx[i]/(L[i] + small)
            ly[i] = Ly[i]/(L[i] + small)
            lz[i] = Lz[i]/(L[i] + small)


                

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

        elif (bc_type==2): #### Infinite disk
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
        elif (bc_type==3): #### Outflows unit vectors and surface density
            ## lx
            lx[ngrid+2*ngc-1] = lx[ngrid-1+ngc]
            lx[ngrid+2*ngc-2] = lx[ngrid-1+ngc]
            lx[0] = lx[2]
            lx[1] = lx[2]
            ## ly
            ly[ngrid+2*ngc-1] = ly[ngrid-1+ngc]
            ly[ngrid+2*ngc-2] = ly[ngrid-1+ngc]
            ly[0] = ly[2]
            ly[1] = ly[2]
            ## lz
            lz[ngrid+2*ngc-1] = lz[ngrid-1+ngc]
            lz[ngrid+2*ngc-2] = lz[ngrid-1+ngc]
            lz[0] = lz[2]
            lz[1] = lz[2]
            ## Sigma (Sigma ~ L/r)
            L[ngrid+2*ngc-1] = L[ngrid-1+ngc]*(r[ngrid+2*ngc-1]/r[ngrid-1+ngc])
            L[ngrid+2*ngc-2] = L[ngrid-1+ngc]*(r[ngrid+2*ngc-2]/r[ngrid-1+ngc])
            L[0] = L[2]*(r[0]/r[2])
            L[1] = L[2]*(r[1]/r[2])
            ## Get Lx, Ly, Lz from L and lx, ly, lz
            # Lx
            Lx[ngrid+2*ngc-1] = L[ngrid+2*ngc-1]*lx[ngrid+2*ngc-1]
            Lx[ngrid+2*ngc-2] = L[ngrid+2*ngc-2]*lx[ngrid+2*ngc-2]
            Lx[0] = L[0]*lx[0]
            Lx[1] = L[1]*lx[1]
            # Ly
            Ly[ngrid+2*ngc-1] = L[ngrid+2*ngc-1]*ly[ngrid+2*ngc-1]
            Ly[ngrid+2*ngc-2] = L[ngrid+2*ngc-2]*ly[ngrid+2*ngc-2]
            Ly[0] = L[0]*ly[0]
            Ly[1] = L[1]*ly[1]
            # Lz
            Lz[ngrid+2*ngc-1] = L[ngrid+2*ngc-1]*lz[ngrid+2*ngc-1]
            Lz[ngrid+2*ngc-2] = L[ngrid+2*ngc-2]*lz[ngrid+2*ngc-2]
            Lz[0] = L[0]*lz[0]
            Lz[1] = L[1]*lz[1]
        elif (bc_type==4): #### Outflows unit vectors and surface density. Don't change outer boundaries to represent infinite disk..
            ## lx
            lx[0] = lx[2]
            lx[1] = lx[2]
            ## ly
            ly[0] = ly[2]
            ly[1] = ly[2]
            ## lz
            lz[0] = lz[2]
            lz[1] = lz[2]
            ## Sigma (Sigma ~ L/r)
            L[0] = L[2]*(r[0]/r[2])
            L[1] = L[2]*(r[1]/r[2])
            ## Get Lx, Ly, Lz from L and lx, ly, lz
            # Lx
            Lx[0] = L[0]*lx[0]
            Lx[1] = L[1]*lx[1]
            # Ly
            Ly[0] = L[0]*ly[0]
            Ly[1] = L[1]*ly[1]
            # Lz
            Lz[0] = L[0]*lz[0]
            Lz[1] = L[1]*lz[1]

        ## Also, we want magnitude and unit vectors calculated in our BC, but these are independent of bc_type.
        # L
        L[0] = (Lx[0]**2. + Ly[0]**2. + Lz[0]**2.)**0.5
        L[1] = (Lx[1]**2. + Ly[1]**2. + Lz[1]**2.)**0.5
        L[ngrid+2*ngc-1] = (Lx[ngrid+2*ngc-1]**2. + Ly[ngrid+2*ngc-1]**2. + Lz[ngrid+2*ngc-1]**2.)**0.5
        L[ngrid+2*ngc-2] = (Lx[ngrid+2*ngc-2]**2. + Ly[ngrid+2*ngc-2]**2. + Lz[ngrid+2*ngc-2]**2.)**0.5
        # lx
        lx[0] = Lx[0]/(L[0] + small)
        lx[1] = Lx[1]/(L[1] + small)
        lx[ngrid+2*ngc-1] = Lx[ngrid+2*ngc-1]/(L[ngrid+2*ngc-1] + small)
        lx[ngrid+2*ngc-2] = Lx[ngrid+2*ngc-2]/(L[ngrid+2*ngc-2] + small)
        # ly
        ly[0] = Ly[0]/(L[0] + small)
        ly[1] = Ly[1]/(L[1] + small)
        ly[ngrid+2*ngc-1] = Ly[ngrid+2*ngc-1]/(L[ngrid+2*ngc-1] + small)
        ly[ngrid+2*ngc-2] = Ly[ngrid+2*ngc-2]/(L[ngrid+2*ngc-2] + small)
        # lz
        lz[0] = Lz[0]/(L[0] + small)
        lz[1] = Lz[1]/(L[1] + small)
        lz[ngrid+2*ngc-1] = Lz[ngrid+2*ngc-1]/(L[ngrid+2*ngc-1] + small)
        lz[ngrid+2*ngc-2] = Lz[ngrid+2*ngc-2]/(L[ngrid+2*ngc-2] + small)

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
            if ((t%(io_freq*tmax) < dt)):
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
        fprintf(f_out, "%e ", x[i])
        fprintf(f_out, "%e ", Q1[i])
        fprintf(f_out, "%e ", Q2[i])
        fprintf(f_out, "%e ", Q3[i])
        fprintf(f_out, "%e ", t)
        fprintf(f_out, "\n")
    fclose(f_out)
