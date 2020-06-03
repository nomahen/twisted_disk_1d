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
    alpha,gamma,HoR,tilt,bhspin,r0,rw,rmin,rmax,rho_type,tmax,cfl,ngrid,bc,io_freq,io_prefix,Q_dim,smax,rmax_Q,Q1_path,Q2_path,Q3_path = p
    # change tilt to radians
    tilt *= np.pi/180.

    ## Build interpolation functions 

    # Parse data 
    Q1_parsed = load_Q(Q1_path,dim=Q_dim)
    Q2_parsed = load_Q(Q2_path,dim=Q_dim)
    Q3_parsed = load_Q(Q3_path,dim=Q_dim)

    cdef int ng_Q
    # Get length of Q. Assumes square array if 2d
    if (Q_dim == "1d"): ng_Q = len(Q1_parsed)
    elif (Q_dim == "2d"): ng_Q = len(Q1_parsed[0])

    # Psi array that Q1/Q2/Q3 correspond to
    _s_arr = np.logspace(0,np.log10(smax+1),ng_Q) - 1
    # r array that q1/q2/q3 correspond to, if 2d (radially dependent) q tables
    _r_arr = np.logspace(0,np.log10(rmax_Q),ng_Q)
    if (Q_dim == "2d"):
        if (rmax != rmax_Q):
            print "Warning! rmax = ", rmax,", rmax_Q = ", rmax_Q, ". These should be equal!\n"

    # Interpolate from derivative array too
    if (Q_dim == "1d"):
        dQ1_dpsi_parsed = np.zeros(ng_Q)
        dQ1_dpsi_parsed[1:ng_Q-1] = (Q1_parsed[2:] - Q1_parsed[:ng_Q-2])/(_s_arr[2:] - _s_arr[:ng_Q-2])
        dQ1_dpsi_parsed[0] = 0.0
        dQ1_dpsi_parsed[ng_Q-1] = dQ1_dpsi_parsed[ng_Q-2]
    elif (Q_dim == "2d"):
        dQ1_dpsi_parsed = np.zeros((ng_Q,ng_Q))
        for ii in range(ng_Q):
            dQ1_dpsi_parsed[ii][1:ng_Q-1] = (Q1_parsed[ii][2:] - Q1_parsed[ii][:ng_Q-2])/(_s_arr[2:] - _s_arr[:ng_Q-2])
            dQ1_dpsi_parsed[ii][0] = 0.0
            dQ1_dpsi_parsed[ii][ng_Q-1] = dQ1_dpsi_parsed[ii][ng_Q-2] 
        Q1_parsed[Q1_parsed>0.] *= 0.0 # temporary fix!!
    ########
 
    ## Build arrays 
    cdef double nu_ref, t_viscous
 
    ## internal coordinate: r = np
    xmin = np.log(rmin)
    xmax = np.log(rmax)
    _dx  = (xmax-xmin)/ngrid
    _x   = np.arange(xmin-(ngc-0.5)*_dx,xmax+(ngc+0.5)*_dx,_dx)
    _r = np.exp(_x)
    #_r = np.logspace(np.log(rmin),np.log(rmax),ngrid,base=np.exp(1)) # natural base
    #_dx = np.average(np.log(_r)[1:] - np.log(_r)[:-1])
 
    # orbital frequency is Keplerian
    omega = _r**(-3./2.)
 
    # density distribution can be "flat" or "gaussian"
    # note: probably most intelligent to normalize rho such that total disk mass
    # is constant regardless of distribution
    # sigma ~ surface density
    if rho_type == "flat":
        _sigma = np.ones(ngrid+2*ngc)*100.
    elif rho_type == "gauss":
        _sigma = (1.0/(rw*np.sqrt(2.0*np.pi))) * np.exp(-0.5*((_r - r0)/rw)**2.0) + np.ones(ngrid+2*ngc)
    else:
        print "Error! rho_type needs to be set to \"gauss\" or \"flat\"! Exiting"
        exit()

    # build angular momentum quantities
    # Here we are using Lambda = I * Omega * r**2., where I = (h/r)^2 * Sigma * r^2 (assuming gamma = 1)
    lda_mag = (HoR)**2. * _sigma * omega**2. * _r**4.
    prec = 0.0
    lda_unit = np.array([np.array([np.sin(tilt)*np.cos(prec),np.sin(tilt)*np.sin(prec),np.cos(tilt)])]*(ngrid+2*ngc)) # each annulus oriented in the same direction initially
    _lda_vec  = np.copy(lda_unit) # building [Lambda_x,Lambda_y,Lambda_z] for each radial grid element
    for j in range(3): _lda_vec[:,j] *= lda_mag
 
 
    # for Lense-Thirring source term
    _omega_p = np.zeros(3*(ngrid+2*ngc))
    _omega_p = np.reshape(_omega_p, (ngrid+2*ngc,3))
    _omega_p[:,2] = 2.0 * bhspin / _r**(3.0)# in tau coordinates _r**3.0 # x/y components are zero, z component is LT precession frequency
 
    # calculate (approximate) viscous time (t_visc = r0**2./nu1(psi=0))
    if (Q_dim == "1d"):
        nu_ref    = (-2.0/3.0)*(-1.0*10**(interp_1d(_s_arr,np.log10(-Q1_parsed + 1e-30),0,ng_Q)))*((HoR**2.0)*r0**0.5)
    elif (Q_dim == "2d"):
        nu_ref = (-2.0/3.0)*(-1.0*10**(interp_2d(_r_arr,_s_arr,np.log10(-Q1_parsed + 1e-30),rmax_Q,0,ng_Q)))*((HoR**2.0)*r0**0.5)

    # We define the viscous time according to the radius halfway through the disk
    t_viscous = rmax**2/nu_ref # / rmin**(1.5) # tau units
 
    # convert tmax, io_freq from t_viscous units to code units
    tmax    = tmax*t_viscous
 
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
    # make dim_Type
    if (Q_dim == '1d'):
        dim_type = 0
    elif (Q_dim == '2d'):
        dim_type = 1
 
 
    ########
 
    ## Do initial input/output
 
    print "#### Parameters ####\n"
    print "alpha     = %s\n" % alpha
    print "gamma     = %s\n" % gamma
    print "HoR       = %s\n" % HoR
    print "tilt      = %s [deg]\n" % tilt
    print "bhspin    = %s\n" % bhspin
    print "r0        = %s [r_g]\n" % r0
    print "rw        = %s [r_g]\n" % rw
    print "rmin      = %s [r_g]\n" % rmin
    print "rmax      = %s [r_g]\n" % rmax
    print "rho_type  = %s \n" % rho_type
    print "tmax      = %s [t_viscous]\n" % (tmax/t_viscous)
    print "t_viscous = %s [r_g/c]\n" % t_viscous
    print "cfl       = %s\n" % cfl
    print "bc        = %s\n" % bc
    print "io_freq   = %s [t_viscous]\n" % io_freq
    print "io_prefix = %s\n" % io_prefix
    print "Q_dim     = %s\n" % Q_dim
    print "smax      = %s\n" % smax
    print "Q1_path   = %s\n" % Q1_path
    print "Q2_path   = %s\n" % Q2_path
    print "Q3_path   = %s\n" % Q3_path
    print "Q length  = %s\n" % ng_Q
    print "####################\n\n"
    print "Beginning simulation...\n\n"
 
    ########
 
    #############
    ## Evolve! ##
    #############

    # Initialize Cython stuff for iteration
    cdef double[:] r = _r
    cdef double    dx = _dx
    cdef double[:] sigma = _sigma
    cdef double[:,:] omega_p = _omega_p
    cdef double[:] L  = np.sqrt(_lda_vec[:,0]**2.0 + _lda_vec[:,1]**2.0 + _lda_vec[:,2]**2.0)
    cdef double[:] Lx   = _lda_vec[:,0]
    cdef double[:] Ly   = _lda_vec[:,1]
    cdef double[:] Lz   = _lda_vec[:,2]
    cdef double[:] lx = _lda_vec[:,0]/np.sqrt(_lda_vec[:,0]**2.0 + _lda_vec[:,1]**2.0 + _lda_vec[:,2]**2.0)
    cdef double[:] ly = _lda_vec[:,1]/np.sqrt(_lda_vec[:,0]**2.0 + _lda_vec[:,1]**2.0 + _lda_vec[:,2]**2.0)
    cdef double[:] lz = _lda_vec[:,2]/np.sqrt(_lda_vec[:,0]**2.0 + _lda_vec[:,1]**2.0 + _lda_vec[:,2]**2.0)
    cdef double[:] omega_p_z = np.copy(_omega_p[:,2]) #* _r**(1.5) # convert to per tau units

    # Primitive variables
    cdef double[:] Lx_L      = np.zeros(ngrid+2*ngc)
    cdef double[:] Lx_R      = np.zeros(ngrid+2*ngc)
    cdef double[:] Ly_L      = np.zeros(ngrid+2*ngc)
    cdef double[:] Ly_R      = np.zeros(ngrid+2*ngc)
    cdef double[:] Lz_L      = np.zeros(ngrid+2*ngc)
    cdef double[:] Lz_R      = np.zeros(ngrid+2*ngc)
    cdef double[:] L_L       = np.zeros(ngrid+2*ngc)
    cdef double[:] L_R       = np.zeros(ngrid+2*ngc)

    # Cell centered gradients
    cdef double[:] dLx_dx    = np.zeros(ngrid+2*ngc)
    cdef double[:] dLy_dx    = np.zeros(ngrid+2*ngc)
    cdef double[:] dLz_dx    = np.zeros(ngrid+2*ngc)
    cdef double[:] dlx_dx    = np.zeros(ngrid+2*ngc)
    cdef double[:] dly_dx    = np.zeros(ngrid+2*ngc)
    cdef double[:] dlz_dx    = np.zeros(ngrid+2*ngc)
    cdef double[:] psi       = np.zeros(ngrid+2*ngc)
    cdef double[:] Q1        = np.zeros(ngrid+2*ngc)
    cdef double[:] Q2        = np.zeros(ngrid+2*ngc)
    cdef double[:] Q3        = np.zeros(ngrid+2*ngc)
    cdef double[:] dQ1_dpsi  = np.zeros(ngrid+2*ngc)
    cdef double[:] dpsi_dx   = np.zeros(ngrid+2*ngc)

    # Cell interface gradients
    cdef double[:] dLx_dx_L    = np.zeros(ngrid+2*ngc)
    cdef double[:] dLx_dx_R    = np.zeros(ngrid+2*ngc)
    cdef double[:] dLy_dx_L    = np.zeros(ngrid+2*ngc)
    cdef double[:] dLy_dx_R    = np.zeros(ngrid+2*ngc)
    cdef double[:] dLz_dx_L    = np.zeros(ngrid+2*ngc)
    cdef double[:] dLz_dx_R    = np.zeros(ngrid+2*ngc)
    cdef double[:] dlx_dx_L    = np.zeros(ngrid+2*ngc)
    cdef double[:] dlx_dx_R    = np.zeros(ngrid+2*ngc)
    cdef double[:] dly_dx_L    = np.zeros(ngrid+2*ngc)
    cdef double[:] dly_dx_R    = np.zeros(ngrid+2*ngc)
    cdef double[:] dlz_dx_L    = np.zeros(ngrid+2*ngc)
    cdef double[:] dlz_dx_R    = np.zeros(ngrid+2*ngc)
    cdef double[:] psi_L       = np.zeros(ngrid+2*ngc)
    cdef double[:] psi_R       = np.zeros(ngrid+2*ngc)
    cdef double[:] Q1_L        = np.zeros(ngrid+2*ngc)
    cdef double[:] Q1_R        = np.zeros(ngrid+2*ngc)
    cdef double[:] Q2_L        = np.zeros(ngrid+2*ngc)
    cdef double[:] Q2_R        = np.zeros(ngrid+2*ngc)
    cdef double[:] Q3_L        = np.zeros(ngrid+2*ngc)
    cdef double[:] Q3_R        = np.zeros(ngrid+2*ngc)
    cdef double[:] dQ1_dpsi_L  = np.zeros(ngrid+2*ngc)
    cdef double[:] dQ1_dpsi_R  = np.zeros(ngrid+2*ngc)
    cdef double[:] dpsi_dx_L   = np.zeros(ngrid+2*ngc)
    cdef double[:] dpsi_dx_R   = np.zeros(ngrid+2*ngc)
    cdef double[:] dQ1_dx_L  = np.zeros(ngrid+2*ngc)
    cdef double[:] dQ1_dx_R  = np.zeros(ngrid+2*ngc)

    # Fluxes (interior cell interfaces only only)
    cdef double[:] F_x       = np.zeros(ngrid+2*ngc-3)
    cdef double[:] F_x_L     = np.zeros(ngrid+2*ngc-3)
    cdef double[:] F_x_R     = np.zeros(ngrid+2*ngc-3)
    cdef double[:] F_y       = np.zeros(ngrid+2*ngc-3)
    cdef double[:] F_y_L     = np.zeros(ngrid+2*ngc-3)
    cdef double[:] F_y_R     = np.zeros(ngrid+2*ngc-3)
    cdef double[:] F_z       = np.zeros(ngrid+2*ngc-3)
    cdef double[:] F_z_L     = np.zeros(ngrid+2*ngc-3)
    cdef double[:] F_z_R     = np.zeros(ngrid+2*ngc-3)
    cdef double[:,:] F_all   = np.zeros((3,ngrid+2*ngc-3))

    # miscellaneous variables
    cdef double Lx_tmp, Ly_tmp
    cdef double r_shift_half = np.exp(_dx/2.)
    cdef double small = 1e-30
    cdef double vL,vR,sL,sR
    cdef double dt = 1000000000000000000.
    cdef double tmp_slope # for data reconstruction
    cdef double vel,vel_L,vel_R,nu,nu_L,nu_R
    cdef double velmax  = 0.5 # in fractions of the sound speed
    cdef double sigma_floor = 1e-3 * np.min(_sigma)
    cdef int i
    cdef double[:] Lx_inf = _lda_vec[ngrid+2*ngc-2:ngrid+2*ngc,0]
    cdef double[:] Ly_inf = _lda_vec[ngrid+2*ngc-2:ngrid+2*ngc,1]
    cdef double[:] Lz_inf = _lda_vec[ngrid+2*ngc-2:ngrid+2*ngc,2]
    cdef double[:] Lx_old = _lda_vec[:,0] # Lx,Ly,Lz_old are for half saving variables during half time-step evolution
    cdef double[:] Ly_old = _lda_vec[:,1]
    cdef double[:] Lz_old = _lda_vec[:,2]

    # For use if timing a section is desired
    cdef double clock_time = 0.0
    cdef clock_t start, end

    # for Q1, Q2, Q3
    cdef double[:] Q1_1d_arr = np.zeros(ng_Q)
    cdef double[:] Q2_1d_arr = np.zeros(ng_Q)
    cdef double[:] Q3_1d_arr = np.zeros(ng_Q)
    cdef double[:] dQ1_dpsi_1d_arr = np.zeros(ng_Q)
    cdef double[:,:] Q1_2d_arr,Q2_2d_arr,Q3_2d_arr,dQ1_dpsi_2d_arr
    if (Q_dim == "1d"):
        Q1_1d_arr = np.log10(-Q1_parsed + small)
        Q2_1d_arr = np.log10(Q2_parsed + small)
        Q3_1d_arr = Q3_parsed#np.log10(Q3_parsed + small)
        dQ1_dpsi_1d_arr = dQ1_dpsi_parsed#np.log10(dQ1_dpsi_parsed + small)
    elif (Q_dim == "2d"):
        Q1_2d_arr = np.log10(-Q1_parsed + small)
        Q2_2d_arr = np.log10(Q2_parsed + small)
        Q3_2d_arr = Q3_parsed#np.log10(Q3_parsed + small)
        dQ1_dpsi_2d_arr = dQ1_dpsi_parsed#np.log10(dQ1_dpsi_parsed + small)
    cdef double[:] s_arr = _s_arr
    cdef double[:] r_arr = _r_arr


    # io variables
    cdef FILE      *f_out
    cdef char[40]  io_fn
    cdef int       io_cnt = 0, nstep = 0
    cdef int       predictor = 1
    cdef int       do_predictor = 1
    cdef double    t = 0.

    # initialize output files
    '''for i in range(ngrid):
        sprintf(io_fn,"%s%d.csv",io_prefix,i)
        f_out = fopen(io_fn,"a+")
        fprintf(f_out, "%e ", Lx[i])
        fprintf(f_out, "%e ", Ly[i])
        fprintf(f_out, "%e ", Lz[i])
        fprintf(f_out, "%e ", r[i])
        fprintf(f_out, "%e ", Q1[i])
        fprintf(f_out, "%e ", Q2[i])
        fprintf(f_out, "%e ", Q3[i])
        fprintf(f_out, "%e ", t)
        fprintf(f_out, "\n")
        fclose(f_out)'''

    ### Do initial outputs
    printf("Doing initial output...\n")
    sprintf(io_fn,"%s%d.csv",io_prefix,io_cnt)
    f_out = fopen(io_fn,"w")
    for i in range(ngrid+2*ngc):
        fprintf(f_out, "%e ", Lx[i])
        fprintf(f_out, "%e ", Ly[i])
        fprintf(f_out, "%e ", Lz[i])
        fprintf(f_out, "%e ", r[i])
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

        ## During first iteration we have cell-centered, Lx,Ly,Lz,lx,ly,lz. We can use this to calculate most cell-interface values: 
        # Lx, Ly, Lz, dLx, dLy, dLz, dlx, dly, dlz, psi, Q1, Q2, Q3
        for i in range(2,ngrid+2*ngc-2):
            
            ## Linear reconstruction for Lx,Ly,Lz
            # Lx
            tmp_slope = minmod( (Lx[i] - Lx[i-1])/dx, (Lx[i+1] - Lx[i])/dx)
            Lx_L[i] = Lx[i] - tmp_slope*dx/2.
            Lx_R[i] = Lx[i] + tmp_slope*dx/2.
            # Ly
            tmp_slope = minmod( (Ly[i] - Ly[i-1])/dx, (Ly[i+1] - Ly[i])/dx)
            Ly_L[i] = Ly[i] - tmp_slope*dx/2.
            Ly_R[i] = Ly[i] + tmp_slope*dx/2.
            # Lz
            tmp_slope = minmod( (Lz[i] - Lz[i-1])/dx, (Lz[i+1] - Lz[i])/dx)
            Lz_L[i] = Lz[i] - tmp_slope*dx/2.
            Lz_R[i] = Lz[i] + tmp_slope*dx/2.
            # Calcualte full L from these quantities
            L_L[i]  = (Lx_L[i]**2. + Ly_L[i]**2. + Lz_L[i]**2.)**0.5
            L_R[i]  = (Lx_R[i]**2. + Ly_R[i]**2. + Lz_R[i]**2.)**0.5
            '''
            ## Do 2nd order, cell-centered differencing for gradients, evaluated at the cell interfaces
            # Lx
            dLx_dx_L[i] = (Lx[i+1]+Lx[i]-Lx[i-1]-Lx[i-2])/4./dx # i - 1/2
            dLx_dx_R[i] = (Lx[i+2]+Lx[i+1]-Lx[i]-Lx[i-1])/4./dx # i + 1/2
            # Ly
            dLy_dx_L[i] = (Ly[i+1]+Ly[i]-Ly[i-1]-Ly[i-2])/4./dx # i - 1/2
            dLy_dx_R[i] = (Ly[i+2]+Ly[i+1]-Ly[i]-Ly[i-1])/4./dx # i + 1/2
            # Lz
            dLz_dx_L[i] = (Lz[i+1]+Lz[i]-Lz[i-1]-Lz[i-2])/4./dx # i - 1/2
            dLz_dx_R[i] = (Lz[i+2]+Lz[i+1]-Lz[i]-Lz[i-1])/4./dx # i + 1/2
            # lx
            dlx_dx_L[i] = (lx[i+1]+lx[i]-lx[i-1]-lx[i-2])/4./dx # i - 1/2
            dlx_dx_R[i] = (lx[i+2]+lx[i+1]-lx[i]-lx[i-1])/4./dx # i + 1/2
            # ly
            dly_dx_L[i] = (ly[i+1]+ly[i]-ly[i-1]-ly[i-2])/4./dx # i - 1/2
            dly_dx_R[i] = (ly[i+2]+ly[i+1]-ly[i]-ly[i-1])/4./dx # i + 1/2
            # lz
            dlz_dx_L[i] = (lz[i+1]+lz[i]-lz[i-1]-lz[i-2])/4./dx # i - 1/2
            dlz_dx_R[i] = (lz[i+2]+lz[i+1]-lz[i]-lz[i-1])/4./dx # i + 1/2'''
            ## Do 2nd order, cell-centered differencing for gradients, evaluated at the cell interfaces. Try shorter stencil?
            # Lx
            dLx_dx_L[i] = (Lx[i]-Lx[i-1])/2./dx # i - 1/2
            dLx_dx_R[i] = (Lx[i+1]-Lx[i])/2./dx # i + 1/2
            # Ly
            dLy_dx_L[i] = (Ly[i]-Ly[i-1])/2./dx # i - 1/2
            dLy_dx_R[i] = (Ly[i+1]-Ly[i])/2./dx # i + 1/2
            # Lz
            dLz_dx_L[i] = (Lz[i]-Lz[i-1])/2./dx # i - 1/2
            dLz_dx_R[i] = (Lz[i+1]-Lz[i])/2./dx # i + 1/2
            # lx
            dlx_dx_L[i] = (lx[i]-lx[i-1])/2./dx # i - 1/2
            dlx_dx_R[i] = (lx[i+1]-lx[i])/2./dx # i + 1/2
            # ly
            dly_dx_L[i] = (ly[i]-ly[i-1])/2./dx # i - 1/2
            dly_dx_R[i] = (ly[i+1]-ly[i])/2./dx # i + 1/2
            # lz
            dlz_dx_L[i] = (lz[i]-lz[i-1])/2./dx # i - 1/2
            dlz_dx_R[i] = (lz[i+1]-lz[i])/2./dx # i + 1/2

            ## Calculate psi from these gradients
            # psi
            psi_L[i]    = (dlx_dx_L[i]**2. + dly_dx_L[i]**2. + dlz_dx_L[i]**2.)**0.5
            psi_R[i]    = (dlx_dx_R[i]**2. + dly_dx_R[i]**2. + dlz_dx_R[i]**2.)**0.5

            ## Interpolate Q1, Q2, Q3. Interpolation depends on whether we have Q coefficients that depend just on psi ("1D") or psi and radius ("2D"). 
            #if (dim_type == 0):  # 1D Q tables
            #    Q1[i]       = -1.0*(10**(interp_1d(s_arr,Q1_1d_arr,psi[i],ng_Q)))
            #    Q2[i]       = 10**(interp_1d(s_arr,Q2_1d_arr,psi[i],ng_Q))
            #    Q3[i]       = interp_1d(s_arr,Q3_1d_arr,psi[i],ng_Q)#10**(interp_1d(s_arr,Q3_arr,psi[i],ng_Q))
            #    dQ1_dpsi[i] = interp_1d(s_arr,dQ1_dpsi_1d_arr,psi[i],ng_Q)#10**(interp_1d(s_arr,dQ1_dpsi_arr,psi[i],ng_Q))
            #elif (dim_type == 1): # 2D Q_tables
            #    Q1[i]       = -1.0*(10**(interp_2d(r_arr,s_arr,Q1_2d_arr,r[i],psi[i],ng_Q)))
            #    Q2[i]       = 10**(interp_2d(r_arr,s_arr,Q2_2d_arr,r[i],psi[i],ng_Q))
            #    Q3[i]       = interp_2d(r_arr,s_arr,Q3_2d_arr,r[i],psi[i],ng_Q)#10**(interp_1d(s_arr,Q3_arr,psi[i],ng_Q))
            #    dQ1_dpsi[i] = interp_2d(r_arr,s_arr,dQ1_dpsi_2d_arr,r[i],psi[i],ng_Q)#10**(interp_1d(s_arr,dQ1_dpsi_arr,psi[i],ng_Q))
            # assuming 1D Q tables, for now!
            Q1_L[i]       = -1.0*(10**(interp_1d(s_arr,Q1_1d_arr,psi_L[i],ng_Q)))
            Q2_L[i]       = 10**(interp_1d(s_arr,Q2_1d_arr,psi_L[i],ng_Q))
            Q3_L[i]       = interp_1d(s_arr,Q3_1d_arr,psi_L[i],ng_Q)
            dQ1_dpsi_L[i] = interp_1d(s_arr,dQ1_dpsi_1d_arr,psi_L[i],ng_Q)
            Q1_R[i]       = -1.0*(10**(interp_1d(s_arr,Q1_1d_arr,psi_R[i],ng_Q)))
            Q2_R[i]       = 10**(interp_1d(s_arr,Q2_1d_arr,psi_R[i],ng_Q))
            Q3_R[i]       = interp_1d(s_arr,Q3_1d_arr,psi_R[i],ng_Q)
            dQ1_dpsi_R[i] = interp_1d(s_arr,dQ1_dpsi_1d_arr,psi_R[i],ng_Q)

        ## During 2nd iteration, we can do CFL condition and calculate dpsi_dx at cell-interfaces; everything else we got last iteration. 
        # If using 2nd order predictor-correct time evolution, only do cfl on the predictor step. 
        if (predictor or (not do_predictor)): 
            ## cfl condition
            # Here, we do something qualitatively similar to Equation (50) of Diego Munoz 2012
            dt = 1000000000000000000.
            for i in range(2,ngrid+2*ngc-2):

                # Can calculate dpsi_dx at the same time as CFL condition. 
                dpsi_dx_L[i] = (psi_R[i]-psi_L[i-1])/2./dx # i-1/2
                dpsi_dx_R[i] = (psi_R[i+1]-psi_L[i])/2./dx # i+1/2
                dQ1_dx_L[i] = (Q1_R[i]-Q1_L[i-1])/2./dx # i-1/2
                dQ1_dx_R[i] = (Q1_R[i+1]-Q1_L[i])/2./dx # i+1/2

                ## Check velocity and effective diffusion coefficient for each cell interface to decide timestep. 
                # left
                vel_L = (HoR**2.)*fmax(fabs(-Q1_L[i] + 2.*dQ1_dx_L[i] + 2.*Q2_L[i]*psi_L[i]), fabs((1./L_L[i]**2.)*(L_L[i]**2.*(-1.*Q1_L[i] + 2.*dQ1_dx_L[i] + 2.*Q2_L[i]*psi_L[i]**2.) + (Lx_L[i]*dLx_dx_L[i] + Ly_L[i]*dLy_dx_L[i] + Lz_L[i]*dLz_dx_L[i])*(2.*Q1_L[i] + Q2_L[i])))) 
                nu_L  = (HoR**2.)*(Q1_L[i]**2. + Q2_L[i]**2. + Q3_L[i]**2.)**(0.5)
                # right
                vel_R = (HoR**2.)*fmax(fabs(-Q1_R[i] + 2.*dQ1_dx_R[i] + 2.*Q2_R[i]*psi_R[i]), fabs((1./L_R[i]**2.)*(L_R[i]**2.*(-1.*Q1_R[i] + 2.*dQ1_dx_R[i] + 2.*Q2_R[i]*psi_R[i]**2.) + (Lx_R[i]*dLx_dx_R[i] + Ly_R[i]*dLy_dx_R[i] + Lz_R[i]*dLz_dx_R[i])*(2.*Q1_R[i] + Q2_R[i])))) 
                nu_R  = (HoR**2.)*(Q1_R[i]**2. + Q2_R[i]**2. + Q3_R[i]**2.)**(0.5)
                # choose maximum velocity, diffusion to constrain timestep
                vel = fmax(vel_L/r_shift_half**(-1.5),vel_R*r_shift_half**(-1.5)) * r[i]**(-1.5)
                nu = fmax(nu_L/r_shift_half**(-1.5),nu_R*r_shift_half**(-1.5)) * r[i]**(-1.5) ### later put r_shift_half back

                # Check tosee if wave velocities are bounded as they are expected to be
                #if (vel > velmax*(HoR**2.)*r[i]**(-1.5)): printf("Wave speed eclipses half the sound speed at i = %d! vel/velmax = %e\n",i,vel/(velmax*(HoR**2.)*r[i]**(-1.5)))
                
                dt = fmin(dt,fabs(cfl*(dx/vel/(1. + 2.*nu/(vel*dx)))))

            # This step is for predictor half-timestep evolution; will multiply by two afterwards
            if (do_predictor): dt = dt/2. 
        else:
            # If we're using predictor-corrector time evolution, and we're on the "corrector" stage, we only need to calculate dpsi_dx here. 
            for i in range(1,ngrid+2*ngc-1):
                dpsi_dx_L[i] = (psi_R[i]-psi_L[i-1])/2./dx # i-1/2
                dpsi_dx_R[i] = (psi_R[i+1]-psi_L[i])/2./dx # i+1/2
                dQ1_dx_L[i] = (Q1_R[i]-Q1_L[i-1])/2./dx # i-1/2
                dQ1_dx_R[i] = (Q1_R[i+1]-Q1_L[i])/2./dx # i+1/2


        # for PLM, lets fill guard cells
        apply_outflow(Lx_L,ngrid+2*ngc)
        apply_outflow(Lx_R,ngrid+2*ngc)
        apply_outflow(Ly_L,ngrid+2*ngc)
        apply_outflow(Ly_R,ngrid+2*ngc)
        apply_outflow(Lz_L,ngrid+2*ngc)
        apply_outflow(Lz_R,ngrid+2*ngc)
        apply_outflow(L_L,ngrid+2*ngc)
        apply_outflow(L_R,ngrid+2*ngc)
        apply_outflow(Q1_L,ngrid+2*ngc)
        apply_outflow(Q1_R,ngrid+2*ngc)

        # Set gradients to zero in guard cells
        apply_zero(dLx_dx_L,ngrid+2*ngc)
        apply_zero(dLx_dx_R,ngrid+2*ngc)
        apply_zero(dLy_dx_L,ngrid+2*ngc)
        apply_zero(dLy_dx_R,ngrid+2*ngc)
        apply_zero(dLz_dx_L,ngrid+2*ngc)
        apply_zero(dLz_dx_R,ngrid+2*ngc)
        apply_zero(Q2_L,ngrid+2*ngc)
        apply_zero(Q2_R,ngrid+2*ngc)
        apply_zero(Q3_L,ngrid+2*ngc)
        apply_zero(Q3_R,ngrid+2*ngc)
        apply_zero(dQ1_dpsi_L,ngrid+2*ngc)
        apply_zero(dQ1_dpsi_R,ngrid+2*ngc)
        apply_zero(psi_L,ngrid+2*ngc)
        apply_zero(psi_R,ngrid+2*ngc)
        apply_zero(dpsi_dx_L,ngrid+2*ngc)
        apply_zero(dpsi_dx_R,ngrid+2*ngc)
        apply_zero(dQ1_dx_L,ngrid+2*ngc)
        apply_zero(dQ1_dx_R,ngrid+2*ngc)

        ## evolve (get fluxes)
        for i in range(ngrid+2*ngc-3):

            ## Left side of fluxes at cell interfaces

            # (L) Q1 advective term
            F_x_L[i] = (HoR**2.) * (Lx_R[i+1]) * (-1.) * Q1_R[i+1] 
            F_y_L[i] = (HoR**2.) * (Ly_R[i+1]) * (-1.) * Q1_R[i+1]
            F_z_L[i] = (HoR**2.) * (Lz_R[i+1]) * (-1.) * Q1_R[i+1]

            # (L) Q1 diffusive term
            F_x_L[i] += (HoR**2.) * (2. * Q1_R[i+1] * (Lx_R[i+1]/L_R[i+1]**2.))*(Lx_R[i+1]*dLx_dx_R[i+1] + Ly_R[i+1]*dLy_dx_R[i+1] + Lz_R[i+1]*dLz_dx_R[i+1]) 
            F_y_L[i] += (HoR**2.) * (2. * Q1_R[i+1] * (Ly_R[i+1]/L_R[i+1]**2.))*(Lx_R[i+1]*dLx_dx_R[i+1] + Ly_R[i+1]*dLy_dx_R[i+1] + Lz_R[i+1]*dLz_dx_R[i+1]) 
            F_z_L[i] += (HoR**2.) * (2. * Q1_R[i+1] * (Lz_R[i+1]/L_R[i+1]**2.))*(Lx_R[i+1]*dLx_dx_R[i+1] + Ly_R[i+1]*dLy_dx_R[i+1] + Lz_R[i+1]*dLz_dx_R[i+1]) 

            # (L) dQ1_dpsi term
            F_x_L[i] += (HoR**2.) * (2. * dQ1_dx_R[i+1]  * Lx_R[i+1])
            F_y_L[i] += (HoR**2.) * (2. * dQ1_dx_R[i+1]  * Ly_R[i+1])
            F_z_L[i] += (HoR**2.) * (2. * dQ1_dx_R[i+1]  * Lz_R[i+1])

            # (L) Q2 advective term 1 (depends on L)
            F_x_L[i] += (HoR**2.) * (Q2_R[i+1]/L_R[i+1]**2.) * (Lx_R[i+1]*dLx_dx_R[i+1] + Ly_R[i+1]*dLy_dx_R[i+1] + Lz_R[i+1]*dLz_dx_R[i+1]) * Lx_R[i+1]
            F_y_L[i] += (HoR**2.) * (Q2_R[i+1]/L_R[i+1]**2.) * (Lx_R[i+1]*dLx_dx_R[i+1] + Ly_R[i+1]*dLy_dx_R[i+1] + Lz_R[i+1]*dLz_dx_R[i+1]) * Ly_R[i+1]
            F_z_L[i] += (HoR**2.) * (Q2_R[i+1]/L_R[i+1]**2.) * (Lx_R[i+1]*dLx_dx_R[i+1] + Ly_R[i+1]*dLy_dx_R[i+1] + Lz_R[i+1]*dLz_dx_R[i+1]) * Lz_R[i+1]

            # (L) Q2 advective term 2 (depends on psi)
            F_x_L[i] += (HoR**2.) * (-2. * Q2_R[i+1] * psi_R[i+1]**2.) * Lx_R[i+1]
            F_y_L[i] += (HoR**2.) * (-2. * Q2_R[i+1] * psi_R[i+1]**2.) * Ly_R[i+1]
            F_z_L[i] += (HoR**2.) * (-2. * Q2_R[i+1] * psi_R[i+1]**2.) * Lz_R[i+1]

            # (L) Q2 diffusive term
            F_x_L[i] += (HoR**2.) * (-1. * Q2_R[i+1])*dLx_dx_R[i+1]
            F_y_L[i] += (HoR**2.) * (-1. * Q2_R[i+1])*dLy_dx_R[i+1]
            F_z_L[i] += (HoR**2.) * (-1. * Q2_R[i+1])*dLz_dx_R[i+1]

            # (L) Q3 precession term
            F_x_L[i] += (HoR**2.) * (-1. * Q3_R[i+1]) * (Ly_R[i+1]*dLz_dx_R[i+1] - Lz_R[i+1]*dLy_dx_R[i+1]) 
            F_y_L[i] += (HoR**2.) * (-1. * Q3_R[i+1]) * (Lz_R[i+1]*dLx_dx_R[i+1] - Lx_R[i+1]*dLz_dx_R[i+1]) 
            F_z_L[i] += (HoR**2.) * (-1. * Q3_R[i+1]) * (Lx_R[i+1]*dLy_dx_R[i+1] - Ly_R[i+1]*dLx_dx_R[i+1]) 

            ## Right side of fluxes at cell interfaces

            # (R) Q1 advective term
            F_x_R[i] = (HoR**2.) * (Lx_L[i+2]) * (-1.) * Q1_L[i+2]
            F_y_R[i] = (HoR**2.) * (Ly_L[i+2]) * (-1.) * Q1_L[i+2]
            F_z_R[i] = (HoR**2.) * (Lz_L[i+2]) * (-1.) * Q1_L[i+2]
 
            # (R) Q1 diffusive term
            F_x_R[i] += (HoR**2.) * (2. * Q1_L[i+2] * (Lx_L[i+2]/L_L[i+2]**2.))*(Lx_L[i+2]*dLx_dx_L[i+2] + Ly_L[i+2]*dLy_dx_L[i+2] + Lz_L[i+2]*dLz_dx_L[i+2])
            F_y_R[i] += (HoR**2.) * (2. * Q1_L[i+2] * (Ly_L[i+2]/L_L[i+2]**2.))*(Lx_L[i+2]*dLx_dx_L[i+2] + Ly_L[i+2]*dLy_dx_L[i+2] + Lz_L[i+2]*dLz_dx_L[i+2])
            F_z_R[i] += (HoR**2.) * (2. * Q1_L[i+2] * (Lz_L[i+2]/L_L[i+2]**2.))*(Lx_L[i+2]*dLx_dx_L[i+2] + Ly_L[i+2]*dLy_dx_L[i+2] + Lz_L[i+2]*dLz_dx_L[i+2]) 

            # (R) dQ1_dpsi term
            F_x_R[i] += (HoR**2.) * (2. * dQ1_dx_L[i+2]  * Lx_L[i+2])
            F_y_R[i] += (HoR**2.) * (2. * dQ1_dx_L[i+2]  * Ly_L[i+2])
            F_z_R[i] += (HoR**2.) * (2. * dQ1_dx_L[i+2]  * Lz_L[i+2])

            # (R) Q2 advective term 1 (depends on magnitude of L)
            F_x_R[i] += (HoR**2.) * (Q2_L[i+2]/L_L[i+2]**2.) * (Lx_L[i+2]*dLx_dx_L[i+2] + Ly_L[i+2]*dLy_dx_L[i+2] + Lz_L[i+2]*dLz_dx_L[i+2]) * Lx_L[i+2]
            F_y_R[i] += (HoR**2.) * (Q2_L[i+2]/L_L[i+2]**2.) * (Lx_L[i+2]*dLx_dx_L[i+2] + Ly_L[i+2]*dLy_dx_L[i+2] + Lz_L[i+2]*dLz_dx_L[i+2]) * Ly_L[i+2]
            F_z_R[i] += (HoR**2.) * (Q2_L[i+2]/L_L[i+2]**2.) * (Lx_L[i+2]*dLx_dx_L[i+2] + Ly_L[i+2]*dLy_dx_L[i+2] + Lz_L[i+2]*dLz_dx_L[i+2]) * Lz_L[i+2]

            # (R) Q2 advective term 2 (depends on psi)
            F_x_R[i] += (HoR**2.) * (-2. * Q2_L[i+2] * psi_L[i+2]**2.) * Lx_L[i+2]
            F_y_R[i] += (HoR**2.) * (-2. * Q2_L[i+2] * psi_L[i+2]**2.) * Ly_L[i+2]
            F_z_R[i] += (HoR**2.) * (-2. * Q2_L[i+2] * psi_L[i+2]**2.) * Lz_L[i+2]

            # (R) Q2 diffusive term
            F_x_R[i] += (HoR**2.) * (-1. * Q2_L[i+2])*dLx_dx_L[i+2]
            F_y_R[i] += (HoR**2.) * (-1. * Q2_L[i+2])*dLy_dx_L[i+2]
            F_z_R[i] += (HoR**2.) * (-1. * Q2_L[i+2])*dLz_dx_L[i+2]

            # (L) Q3 precession term
            F_x_R[i] += (HoR**2.) * (-1. * Q3_L[i+2]) * (Ly_L[i+2]*dLz_dx_L[i+2] - Lz_L[i+2]*dLy_dx_L[i+2]) 
            F_y_R[i] += (HoR**2.) * (-1. * Q3_L[i+2]) * (Lz_L[i+2]*dLx_dx_L[i+2] - Lx_L[i+2]*dLz_dx_L[i+2]) 
            F_z_R[i] += (HoR**2.) * (-1. * Q3_L[i+2]) * (Lx_L[i+2]*dLy_dx_L[i+2] - Ly_L[i+2]*dLx_dx_L[i+2]) 

            ## Calculate wave velocities

            # Get minimum signal speed by taking minimum eigen values at left and right faces, and then the minimum of those
            vL = fmin(-Q1_R[i+1] + 2.*dQ1_dx_R[i+1] - 2.*Q2_R[i+1]*psi_R[i+1]**2., (1./L_R[i+1]**2.)*(L_R[i+1]**2.*(-1.*Q1_R[i+1] + 2.*dQ1_dx_R[i+1] - 2.*Q2_R[i+1]*psi_R[i+1]**2.) + (Lx_R[i+1]*dLx_dx_R[i+1] + Ly_R[i+1]*dLy_dx_R[i+1] + Lz_R[i+1]*dLz_dx_R[i+1])*(2.*Q1_R[i+1] + Q2_R[i+1]))) 
            vR = fmin(-Q1_L[i+2] + 2.*dQ1_dx_L[i+2] - 2.*Q2_L[i+2]*psi_L[i+2]**2., (1./L_L[i+2]**2.)*(L_L[i+2]**2.*(-1.*Q1_L[i+2] + 2.*dQ1_dx_L[i+2] - 2.*Q2_L[i+2]*psi_L[i+2]**2.) + (Lx_L[i+2]*dLx_dx_L[i+2] + Ly_L[i+2]*dLy_dx_L[i+2] + Lz_L[i+2]*dLz_dx_L[i+2])*(2.*Q1_L[i+2] + Q2_L[i+2]))) 
            sL = (HoR**2.)*fmin(vL,vR)


            # Get maximum signal speed by taking maximum eigen values at left and right faces, and then the maximum of those
            vL = fmax(-Q1_R[i+1] + 2.*dQ1_dx_R[i+1] - 2.*Q2_R[i+1]*psi_R[i+1]**2., (1./L_R[i+1]**2.)*(L_R[i+1]**2.*(-1.*Q1_R[i+1] + 2.*dQ1_dx_R[i+1] - 2.*Q2_R[i+1]*psi_R[i+1]**2.) + (Lx_R[i+1]*dLx_dx_R[i+1] + Ly_R[i+1]*dLy_dx_R[i+1] + Lz_R[i+1]*dLz_dx_R[i+1])*(2.*Q1_R[i+1] + Q2_R[i+1]))) 
            vR = fmax(-Q1_L[i+2] + 2.*dQ1_dx_L[i+2] - 2.*Q2_L[i+2]*psi_L[i+2]**2., (1./L_L[i+2]**2.)*(L_L[i+2]**2.*(-1.*Q1_L[i+2] + 2.*dQ1_dx_L[i+2] - 2.*Q2_L[i+2]*psi_L[i+2]**2.) + (Lx_L[i+2]*dLx_dx_L[i+2] + Ly_L[i+2]*dLy_dx_L[i+2] + Lz_L[i+2]*dLz_dx_L[i+2])*(2.*Q1_L[i+2] + Q2_L[i+2]))) 
            sR = (HoR**2.)*fmax(vL,vR)

            # Limit fluxes!
            #F_x_L[i] = F_x_L[i] * fmin(1.,  fabs(velmax*Lx_L[i]/F_x_L[i]))
            #F_x_R[i] = F_x_R[i] * fmin(1.,  fabs(velmax*Lx_R[i]/F_x_R[i]))
            #F_y_L[i] = F_y_L[i] * fmin(1.,  fabs(velmax*Ly_L[i]/F_y_L[i]))
            #F_y_R[i] = F_y_R[i] * fmin(1.,  fabs(velmax*Ly_R[i]/F_y_R[i]))
            #F_z_L[i] = F_z_L[i] * fmin(1.,  fabs(velmax*Lz_L[i]/F_z_L[i]))
            #F_z_R[i] = F_z_R[i] * fmin(1.,  fabs(velmax*Lz_R[i]/F_z_R[i]))
            #sL = fmax(sL,-velmax)
            #sR = fmin(sR,velmax)
            
            # Equations 9.82-9.91 of dongwook ams notes
            if (sL >= 0.):
                F_x[i] = F_x_L[i]
                F_y[i] = F_y_L[i]
                F_z[i] = F_z_L[i]
            elif ((sL < 0.) and (sR >= 0.)):
                F_x[i] = (sR*F_x_L[i] - sL*F_x_R[i] + sL*sR*(Lx_L[i+2] - Lx_R[i+1]))/(sR - sL + small)
                F_y[i] = (sR*F_y_L[i] - sL*F_y_R[i] + sL*sR*(Ly_L[i+2] - Ly_R[i+1]))/(sR - sL + small)
                F_z[i] = (sR*F_z_L[i] - sL*F_z_R[i] + sL*sR*(Lz_L[i+2] - Lz_R[i+1]))/(sR - sL + small)
            else:
                F_x[i] = F_x_R[i]
                F_y[i] = F_y_R[i]
                F_z[i] = F_z_R[i]

        # Check to make sure flux doesn't re-enter grid after it's copied out
        '''F_x[0] = fmin(0.,F_x[0])
        F_y[0] = fmin(0.,F_y[0])
        F_z[0] = fmin(0.,F_z[0])
        if(bc_type!=3): # If disk is infinite, dont do this step, we want mass to flow in if disk is infinite
            F_x[ngrid+2*ngc-4] = fmax(0.,F_x[ngrid+2*ngc-4])
            F_y[ngrid+2*ngc-4] = fmax(0.,F_y[ngrid+2*ngc-4])
            F_z[ngrid+2*ngc-4] = fmax(0.,F_z[ngrid+2*ngc-4])
        '''

        ## update
        for i in range(2,ngrid+2*ngc-2):
            Lx[i] = Lx_old[i] - (dt/dx)*(F_x[i-1] - F_x[i-2]) * r[i]**(-1.5)
            Ly[i] = Ly_old[i] - (dt/dx)*(F_y[i-1] - F_y[i-2]) * r[i]**(-1.5)
            Lz[i] = Lz_old[i] - (dt/dx)*(F_z[i-1] - F_z[i-2]) * r[i]**(-1.5)

        for i in range(ngrid+2*ngc):
            ## Update external torques
            Lx_tmp = Lx_old[i]*cos(omega_p_z[i]*dt) - Ly_old[i]*sin(omega_p_z[i]*dt)
            Ly_tmp = Lx_old[i]*sin(omega_p_z[i]*dt) + Ly_old[i]*cos(omega_p_z[i]*dt)
            Lx[i] = 1.0*Lx_tmp
            Ly[i] = 1.0*Ly_tmp


            L[i]  = (Lx[i]**2. + Ly[i]**2. + Lz[i]**2.)**0.5
            lx[i] = Lx[i]/L[i]
            ly[i] = Ly[i]/L[i]
            lz[i] = Lz[i]/L[i]

            # apply density floor
            if (HoR**(-2.)*L[i]*r[i] < sigma_floor):
                printf("Sigma floor = %e, cell i = %d has sigma = %e\n",sigma_floor,i,HoR**(-2.)*L[i]*r[i])
                L[i] = fmax(sigma_floor,(HoR**(-2.))*L[i]*r[i])*HoR**(2.)/r[i]
                printf("and now: Sigma floor = %e, cell i = %d has sigma = %e\n",sigma_floor,i,HoR**(-2.)*L[i]*r[i])
            L[i] = fmax(sigma_floor,(HoR**(-2.))*L[i]/r[i])*HoR**(2.)*r[i]
            Lx[i] = L[i] * lx[i]
            Ly[i] = L[i] * ly[i]
            Lz[i] = L[i] * lz[i]

        ## apply boundary conditions
        ## Weight by R so sigma is continuous! L should have profile at boundaries
        ## Two guard cells on each end
        if   (bc_type==0): #### Apply sink boundary conditions
            ## Lx
            Lx[0] = 1e-10 * Lx[2]*(r[2]/r[0])**(-1.)
            Lx[1] = 1e-10 * Lx[2]*(r[2]/r[1])**(-1.)
            Lx[ngrid+2*ngc-1] = 1e-10 * Lx[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-1])**(-1.)
            Lx[ngrid+2*ngc-2] = 1e-10 * Lx[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-2])**(-1.)
            ## Ly
            Ly[0] = 1e-10 * Ly[2]*(r[2]/r[0])**(-1.)
            Ly[1] = 1e-10 * Ly[2]*(r[2]/r[1])**(-1.)
            Ly[ngrid+2*ngc-1] = 1e-10 * Ly[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-1])**(-1.)
            Ly[ngrid+2*ngc-2] = 1e-10 * Ly[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-2])**(-1.)
            ## Lz
            Lz[0] = 1e-10 * Lz[2]*(r[2]/r[0])**(-1.)
            Lz[1] = 1e-10 * Lz[2]*(r[2]/r[1])**(-1.)
            Lz[ngrid+2*ngc-1] = 1e-10 * Lz[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-1])**(-1.)
            Lz[ngrid+2*ngc-2] = 1e-10 * Lz[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-2])**(-1.)
            ## L
            L[0]  = (Lx[0]**2. + Ly[0]**2. + Lz[0]**2.)**0.5
            L[1]  = (Lx[1]**2. + Ly[1]**2. + Lz[1]**2.)**0.5
            L[ngrid+2*ngc-1]  = (Lx[ngrid+2*ngc-1]**2. + Ly[ngrid+2*ngc-1]**2. + Lz[ngrid+2*ngc-1]**2.)**0.5
            L[ngrid+2*ngc-2]  = (Lx[ngrid+2*ngc-2]**2. + Ly[ngrid+2*ngc-2]**2. + Lz[ngrid+2*ngc-2]**2.)**0.5

        elif (bc_type==1): #### Apply outflow boundary conditions
            ## Lx
            Lx[0] = Lx[2]*(r[2]/r[0])**(-1.)
            Lx[1] = Lx[2]*(r[2]/r[1])**(-1.)
            Lx[ngrid+2*ngc-1] = Lx[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-1])**(-1.)
            Lx[ngrid+2*ngc-2] = Lx[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-2])**(-1.)
            ## Ly
            Ly[0] = Ly[2]*(r[2]/r[0])**(-1.)
            Ly[1] = Ly[2]*(r[2]/r[1])**(-1.)
            Ly[ngrid+2*ngc-1] = Ly[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-1])**(-1.)
            Ly[ngrid+2*ngc-2] = Ly[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-2])**(-1.)
            ## Lz
            Lz[0] = Lz[2]*(r[2]/r[0])**(-1.)
            Lz[1] = Lz[2]*(r[2]/r[1])**(-1.)
            Lz[ngrid+2*ngc-1] = Lz[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-1])**(-1.)
            Lz[ngrid+2*ngc-2] = Lz[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-2])**(-1.)
            ## L
            L[0]  = (Lx[0]**2. + Ly[0]**2. + Lz[0]**2.)**0.5
            L[1]  = (Lx[1]**2. + Ly[1]**2. + Lz[1]**2.)**0.5
            L[ngrid+2*ngc-1]  = (Lx[ngrid+2*ngc-1]**2. + Ly[ngrid+2*ngc-1]**2. + Lz[ngrid+2*ngc-1]**2.)**0.5
            L[ngrid+2*ngc-2]  = (Lx[ngrid+2*ngc-2]**2. + Ly[ngrid+2*ngc-2]**2. + Lz[ngrid+2*ngc-2]**2.)**0.5

        elif (bc_type==2): #### Apply outflow outer and sink inner boundary conditions
            ## Lx
            Lx[0] = 1e-10 * Lx[2]*(r[2]/r[0])**(-1.)
            Lx[1] = Lx[2]*(r[2]/r[1])**(-1.)
            Lx[ngrid+2*ngc-1] = Lx[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-1])**(-1.)
            Lx[ngrid+2*ngc-2] = Lx[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-2])**(-1.)
            ## Ly
            Ly[0] = 1e-10 * Ly[2]*(r[2]/r[0])**(-1.)
            Ly[1] = Ly[2]*(r[2]/r[1])**(-1.)
            Ly[ngrid+2*ngc-1] = Ly[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-1])**(-1.)
            Ly[ngrid+2*ngc-2] = Ly[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-2])**(-1.)
            ## Lz
            Lz[0] = 1e-10 * Lz[2]*(r[2]/r[0])**(-1.)
            Lz[1] = Lz[2]*(r[2]/r[1])**(-1.)
            Lz[ngrid+2*ngc-1] = Lz[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-1])**(-1.)
            Lz[ngrid+2*ngc-2] = Lz[ngrid+2*ngc-3]*(r[ngrid+2*ngc-3]/r[ngrid+2*ngc-2])**(-1.)
            ## L
            L[0]  = (Lx[0]**2. + Ly[0]**2. + Lz[0]**2.)**0.5
            L[1]  = (Lx[1]**2. + Ly[1]**2. + Lz[1]**2.)**0.5
            L[ngrid+2*ngc-1]  = (Lx[ngrid+2*ngc-1]**2. + Ly[ngrid+2*ngc-1]**2. + Lz[ngrid+2*ngc-1]**2.)**0.5
            L[ngrid+2*ngc-2]  = (Lx[ngrid+2*ngc-2]**2. + Ly[ngrid+2*ngc-2]**2. + Lz[ngrid+2*ngc-2]**2.)**0.5

        elif (bc_type==3): #### Infinite disk
            ## Lx
            Lx[0] = Lx[2]*(r[2]/r[0])**(-1.)
            Lx[1] = Lx[2]*(r[2]/r[1])**(-1.)
            Lx[ngrid+2*ngc-1] = Lx_inf[1] 
            Lx[ngrid+2*ngc-2] = Lx_inf[0]
            ## Ly
            Ly[0] = Ly[2]*(r[2]/r[0])**(-1.)
            Ly[1] = Ly[2]*(r[2]/r[1])**(-1.)
            Ly[ngrid+2*ngc-1] = Ly_inf[1]
            Ly[ngrid+2*ngc-2] = Ly_inf[0]
            ## Lz
            Lz[0] = Lz[2]*(r[2]/r[0])**(-1.)
            Lz[1] = Lz[2]*(r[2]/r[1])**(-1.)
            Lz[ngrid+2*ngc-1] = Lz_inf[1]
            Lz[ngrid+2*ngc-2] = Lz_inf[0]
            ## L
            L[0]  = (Lx[0]**2. + Ly[0]**2. + Lz[0]**2.)**0.5
            L[1]  = (Lx[1]**2. + Ly[1]**2. + Lz[1]**2.)**0.5
            L[ngrid+2*ngc-1]  = (Lx[ngrid+2*ngc-1]**2. + Ly[ngrid+2*ngc-1]**2. + Lz[ngrid+2*ngc-1]**2.)**0.5
            L[ngrid+2*ngc-2]  = (Lx[ngrid+2*ngc-2]**2. + Ly[ngrid+2*ngc-2]**2. + Lz[ngrid+2*ngc-2]**2.)**0.5


        if (predictor and do_predictor):
            dt = dt*2.    # During the CFL step, we halved the timestep to do the half timestep update; we're now return dt to the size of a full timestep. 
            predictor = 0 # next while loop iteration due to the full update
        else:
            #### Update timestep and do outputs
            t += dt
            nstep += 1

            # Update old quantities; these dont change during predictor stage
            Lx_old = Lx
            Ly_old = Ly
            Lz_old = Lz
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
                    fprintf(f_out, "%e ", r[i])
                    fprintf(f_out, "%e ", Q1[i])
                    fprintf(f_out, "%e ", Q2[i])
                    fprintf(f_out, "%e ", Q3[i])
                    fprintf(f_out, "%e ", t)
                    fprintf(f_out, "\n")
                fclose(f_out)
                io_cnt += 1 

    ### Do initial outputs
    printf("Doing final output...\n")
    sprintf(io_fn,"%s%d.csv",io_prefix,io_cnt)
    f_out = fopen(io_fn,"w")
    for i in range(ngrid+2*ngc):
        fprintf(f_out, "%e ", Lx[i])
        fprintf(f_out, "%e ", Ly[i])
        fprintf(f_out, "%e ", Lz[i])
        fprintf(f_out, "%e ", r[i])
        fprintf(f_out, "%e ", Q1[i])
        fprintf(f_out, "%e ", Q2[i])
        fprintf(f_out, "%e ", Q3[i])
        fprintf(f_out, "%e ", t)
        fprintf(f_out, "\n")
    fclose(f_out)
