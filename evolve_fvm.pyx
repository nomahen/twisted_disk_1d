"""
1D evolution of twisted accretion disks
 
Finite volume method
"""
 
import numpy as np
import matplotlib.pyplot as plt
cimport numpy as np
cimport cython
from libc.stdio cimport printf,fopen,fclose,fprintf,FILE,sprintf
from libc.stdlib cimport malloc
from libc.math cimport fmin, fmax, fabs, isnan, sin, cos

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
cdef double[:,:] flux_calc(double HoR, double velmax, double dt, int ngrid, double[:] Lx_L, double[:] Lx_R, double[:] Ly_L, double[:] Ly_R, double[:] Lz_L, double[:] Lz_R, double[:] L_L, double[:] L_R, double[:] dLx_dx_L, double[:] dLx_dx_R, double[:] dLy_dx_L, double[:] dLy_dx_R, double[:] dLz_dx_L, double[:] dLz_dx_R, double[:] dlx_dx_L, double[:] dlx_dx_R, double[:] dly_dx_L, double[:] dly_dx_R, double[:] dlz_dx_L, double[:] dlz_dx_R, double[:] psi_L, double[:] psi_R, double[:] Q1_L, double[:] Q1_R, double[:] Q2_L, double[:] Q2_R, double[:] Q3_L, double[:] Q3_R, double[:] dQ1_dpsi_L, double[:] dQ1_dpsi_R, double[:] dpsi_dx_L, double[:] dpsi_dx_R):
        cdef double vL,vR,sL,sR
        cdef double small = 1e-30
        cdef int i
        cdef double[:] F_x       = np.zeros(ngrid-3)
        cdef double[:] F_x_L     = np.zeros(ngrid-3)
        cdef double[:] F_x_R     = np.zeros(ngrid-3)
        cdef double[:] F_y       = np.zeros(ngrid-3)
        cdef double[:] F_y_L     = np.zeros(ngrid-3)
        cdef double[:] F_y_R     = np.zeros(ngrid-3)
        cdef double[:] F_z       = np.zeros(ngrid-3)
        cdef double[:] F_z_L     = np.zeros(ngrid-3)
        cdef double[:] F_z_R     = np.zeros(ngrid-3)
        cdef double[:,:] F_all   = np.zeros((3,ngrid-3))

        # fluxcalc
        for i in range(ngrid-3):

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
            F_x_L[i] += (HoR**2.) * (2. * dpsi_dx_R[i+1]*dQ1_dpsi_R[i+1]  * Lx_R[i+1])
            F_y_L[i] += (HoR**2.) * (2. * dpsi_dx_R[i+1]*dQ1_dpsi_R[i+1]  * Ly_R[i+1])
            F_z_L[i] += (HoR**2.) * (2. * dpsi_dx_R[i+1]*dQ1_dpsi_R[i+1]  * Lz_R[i+1])

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
            F_x_R[i] += (HoR**2.) * (2. * dpsi_dx_L[i+2]*dQ1_dpsi_L[i+2]  * Lx_L[i+2])
            F_y_R[i] += (HoR**2.) * (2. * dpsi_dx_L[i+2]*dQ1_dpsi_L[i+2]  * Ly_L[i+2])
            F_z_R[i] += (HoR**2.) * (2. * dpsi_dx_L[i+2]*dQ1_dpsi_L[i+2]  * Lz_L[i+2])

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
            vL = fmin(-Q1_R[i+1] + 2.*dpsi_dx_R[i+1]*dQ1_dpsi_R[i+1] - 2.*Q2_R[i+1]*psi_R[i+1]**2., (1./L_R[i+1]**2.)*(L_R[i+1]**2.*(-1.*Q1_R[i+1] + 2.*dQ1_dpsi_R[i+1]*dpsi_dx_R[i+1] - 2.*Q2_R[i+1]*psi_R[i+1]**2.) + (Lx_R[i+1]*dLx_dx_R[i+1] + Ly_R[i+1]*dLy_dx_R[i+1] + Lz_R[i+1]*dLz_dx_R[i+1])*(2.*Q1_R[i+1] + Q2_R[i+1]))) 
            vR = fmin(-Q1_L[i+2] + 2.*dpsi_dx_L[i+2]*dQ1_dpsi_L[i+2] - 2.*Q2_L[i+2]*psi_L[i+2]**2., (1./L_L[i+2]**2.)*(L_L[i+2]**2.*(-1.*Q1_L[i+2] + 2.*dQ1_dpsi_L[i+2]*dpsi_dx_L[i+2] - 2.*Q2_L[i+2]*psi_L[i+2]**2.) + (Lx_L[i+2]*dLx_dx_L[i+2] + Ly_L[i+2]*dLy_dx_L[i+2] + Lz_L[i+2]*dLz_dx_L[i+2])*(2.*Q1_L[i+2] + Q2_L[i+2]))) 
            sL = (HoR**2.)*fmin(vL,vR)


            # Get maximum signal speed by taking maximum eigen values at left and right faces, and then the maximum of those
            vL = fmax(-Q1_R[i+1] + 2.*dpsi_dx_R[i+1]*dQ1_dpsi_R[i+1] - 2.*Q2_R[i+1]*psi_R[i+1]**2., (1./L_R[i+1]**2.)*(L_R[i+1]**2.*(-1.*Q1_R[i+1] + 2.*dQ1_dpsi_R[i+1]*dpsi_dx_R[i+1] - 2.*Q2_R[i+1]*psi_R[i+1]**2.) + (Lx_R[i+1]*dLx_dx_R[i+1] + Ly_R[i+1]*dLy_dx_R[i+1] + Lz_R[i+1]*dLz_dx_R[i+1])*(2.*Q1_R[i+1] + Q2_R[i+1]))) 
            vR = fmax(-Q1_L[i+2] + 2.*dpsi_dx_L[i+2]*dQ1_dpsi_L[i+2] - 2.*Q2_L[i+2]*psi_L[i+2]**2., (1./L_L[i+2]**2.)*(L_L[i+2]**2.*(-1.*Q1_L[i+2] + 2.*dQ1_dpsi_L[i+2]*dpsi_dx_L[i+2] - 2.*Q2_L[i+2]*psi_L[i+2]**2.) + (Lx_L[i+2]*dLx_dx_L[i+2] + Ly_L[i+2]*dLy_dx_L[i+2] + Lz_L[i+2]*dLz_dx_L[i+2])*(2.*Q1_L[i+2] + Q2_L[i+2]))) 
            sR = (HoR**2.)*fmax(vL,vR)

            # Make corrections for velmax!
            sL = fmax(sL,-velmax)
            sR = fmin(sR,velmax)
            
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
        F_x[ngrid-4] = fmax(0.,F_x[ngrid-4])
        F_y[ngrid-4] = fmax(0.,F_y[ngrid-4])
        F_z[ngrid-4] = fmax(0.,F_z[ngrid-4])

        # package and return
        F_all[0] = F_x
        F_all[1] = F_y
        F_all[2] = F_z
        return F_all

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double minmod(double a, double b):
    if   ( (fabs(a) < fabs(b)) and (a*b > 0.) ):
            return a
    elif ( (fabs(b) < fabs(a)) and (a*b > 0.) ):
            return b
    else: 
            return 0.

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double interp_1d(double[:] x, double[:] y, double new_x, int nx):
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
cdef double interp_2d(double[:] x, double[:] y, double[:,:] z, double new_x, double new_y, int n):
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
    cdef double alpha, gamma, HoR, tilt, bhspin, r0, rw, rmin, rmax, tmax, smax, cfl
    cdef int    ngrid, bc_type, dim_type, io_freq
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
    _r = np.logspace(np.log(rmin),np.log(rmax),ngrid,base=np.exp(1)) # natural base
    _dx = np.average(np.log(_r)[1:] - np.log(_r)[:-1])
 
    # orbital frequency is Keplerian
    omega = _r**(-3./2.)
 
    # density distribution can be "flat" or "gaussian"
    # note: probably most intelligent to normalize rho such that total disk mass
    # is constant regardless of distribution
    # sigma ~ surface density
    if rho_type == "flat":
        _sigma = np.ones(ngrid)*100.
    elif rho_type == "gauss":
        _sigma = (1.0/(rw*np.sqrt(2.0*np.pi))) * np.exp(-0.5*((_r - r0)/rw)**2.0) + np.ones(ngrid)
    else:
        print "Error! rho_type needs to be set to \"gauss\" or \"flat\"! Exiting"
        exit()

    # build angular momentum quantities
    # Here we are using Lambda = I * Omega * r**2., where I = (h/r)^2 * Sigma * r^2 (assuming gamma = 1)
    lda_mag = (HoR)**2. * _sigma * omega**2. * _r**4.
    prec = 0.0
    lda_unit = np.array([np.array([np.sin(tilt)*np.cos(prec),np.sin(tilt)*np.sin(prec),np.cos(tilt)])]*ngrid) # each annulus oriented in the same direction initially
    _lda_vec  = np.copy(lda_unit) # building [Lambda_x,Lambda_y,Lambda_z] for each radial grid element
    for j in range(3): _lda_vec[:,j] *= lda_mag
 
 
    # for Lense-Thirring source term
    _omega_p = np.zeros(3*ngrid)
    _omega_p = np.reshape(_omega_p, (ngrid,3))
    _omega_p[:,2] = 2.0 * bhspin / _r**(1.5)# in tau coordinates _r**3.0 # x/y components are zero, z component is LT precession frequency
 
    # calculate (approximate) viscous time (t_visc = r0**2./nu1(psi=0))
    if (Q_dim == "1d"):
        nu_ref    = (-2.0/3.0)*(-1.0*10**(interp_1d(_s_arr,np.log10(-Q1_parsed + 1e-30),0,ng_Q)))*((HoR**2.0)*r0**0.5)
    elif (Q_dim == "2d"):
        nu_ref = (-2.0/3.0)*(-1.0*10**(interp_2d(_r_arr,_s_arr,np.log10(-Q1_parsed + 1e-30),rmax_Q,0,ng_Q)))*((HoR**2.0)*r0**0.5)
    t_viscous = r0**2/nu_ref / rmin**(1.5) # tau units
 
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
    print "io_freq   = %s\n" % io_freq
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
    cdef double[:] Lx_L      = np.zeros(ngrid)
    cdef double[:] Lx_R      = np.zeros(ngrid)
    cdef double[:] Ly_L      = np.zeros(ngrid)
    cdef double[:] Ly_R      = np.zeros(ngrid)
    cdef double[:] Lz_L      = np.zeros(ngrid)
    cdef double[:] Lz_R      = np.zeros(ngrid)
    cdef double[:] L_L       = np.zeros(ngrid)
    cdef double[:] L_R       = np.zeros(ngrid)

    # Cell centered gradients
    cdef double[:] dLx_dx    = np.zeros(ngrid)
    cdef double[:] dLy_dx    = np.zeros(ngrid)
    cdef double[:] dLz_dx    = np.zeros(ngrid)
    cdef double[:] dlx_dx    = np.zeros(ngrid)
    cdef double[:] dly_dx    = np.zeros(ngrid)
    cdef double[:] dlz_dx    = np.zeros(ngrid)
    cdef double[:] psi       = np.zeros(ngrid)
    cdef double[:] Q1        = np.zeros(ngrid)
    cdef double[:] Q2        = np.zeros(ngrid)
    cdef double[:] Q3        = np.zeros(ngrid)
    cdef double[:] dQ1_dpsi  = np.zeros(ngrid)
    cdef double[:] dpsi_dx   = np.zeros(ngrid)

    # Cell interface gradients
    cdef double[:] dLx_dx_L    = np.zeros(ngrid)
    cdef double[:] dLx_dx_R    = np.zeros(ngrid)
    cdef double[:] dLy_dx_L    = np.zeros(ngrid)
    cdef double[:] dLy_dx_R    = np.zeros(ngrid)
    cdef double[:] dLz_dx_L    = np.zeros(ngrid)
    cdef double[:] dLz_dx_R    = np.zeros(ngrid)
    cdef double[:] dlx_dx_L    = np.zeros(ngrid)
    cdef double[:] dlx_dx_R    = np.zeros(ngrid)
    cdef double[:] dly_dx_L    = np.zeros(ngrid)
    cdef double[:] dly_dx_R    = np.zeros(ngrid)
    cdef double[:] dlz_dx_L    = np.zeros(ngrid)
    cdef double[:] dlz_dx_R    = np.zeros(ngrid)
    cdef double[:] psi_L       = np.zeros(ngrid)
    cdef double[:] psi_R       = np.zeros(ngrid)
    cdef double[:] Q1_L        = np.zeros(ngrid)
    cdef double[:] Q1_R        = np.zeros(ngrid)
    cdef double[:] Q2_L        = np.zeros(ngrid)
    cdef double[:] Q2_R        = np.zeros(ngrid)
    cdef double[:] Q3_L        = np.zeros(ngrid)
    cdef double[:] Q3_R        = np.zeros(ngrid)
    cdef double[:] dQ1_dpsi_L  = np.zeros(ngrid)
    cdef double[:] dQ1_dpsi_R  = np.zeros(ngrid)
    cdef double[:] dpsi_dx_L   = np.zeros(ngrid)
    cdef double[:] dpsi_dx_R   = np.zeros(ngrid)

    # Fluxes (interior cell interfaces only only)
    cdef double[:] F_x       = np.zeros(ngrid-3)
    cdef double[:] F_x_L     = np.zeros(ngrid-3)
    cdef double[:] F_x_R     = np.zeros(ngrid-3)
    cdef double[:] F_y       = np.zeros(ngrid-3)
    cdef double[:] F_y_L     = np.zeros(ngrid-3)
    cdef double[:] F_y_R     = np.zeros(ngrid-3)
    cdef double[:] F_z       = np.zeros(ngrid-3)
    cdef double[:] F_z_L     = np.zeros(ngrid-3)
    cdef double[:] F_z_R     = np.zeros(ngrid-3)
    cdef double[:,:] F_all   = np.zeros((3,ngrid-3))

    # miscellaneous variables
    cdef double Lx_tmp, Ly_tmp
    cdef double small = 1e-30
    cdef double dt = 1000000000000000000.
    cdef double tmp_slope # for data reconstruction
    cdef double vel,nu
    cdef double velmax  = 0.5*HoR # 1/2 sound speed in code units
    cdef int i
    cdef double[:] Lx_inf = _lda_vec[ngrid-2:ngrid,0]
    cdef double[:] Ly_inf = _lda_vec[ngrid-2:ngrid,1]
    cdef double[:] Lz_inf = _lda_vec[ngrid-2:ngrid,2]
    cdef double[:] Lx_old = _lda_vec[:,0] # Lx,Ly,Lz_old are for half saving variables during half time-step evolution
    cdef double[:] Ly_old = _lda_vec[:,1]
    cdef double[:] Lz_old = _lda_vec[:,2]

    # for Q1, Q2, Q3
    cdef double[:] Q1_1d_arr,Q2_1d_arr,Q3_1d_arr,dQ1_dpsi_1d_arr
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
    for i in range(ngrid):
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
        fclose(f_out)
    
    # iterate!
    while (t < tmax):
 
        ### Reconstruct, Evolve, Average algorithm

        ## calculate cell-centered gradients

        for i in range(1,ngrid-1):
            dLx_dx[i]  = (0.5/dx) * (Lx[i+1] - Lx[i-1])
            dLy_dx[i]  = (0.5/dx) * (Ly[i+1] - Ly[i-1])
            dLz_dx[i]  = (0.5/dx) * (Lz[i+1] - Lz[i-1])

            dlx_dx[i]  = (0.5/dx) * (lx[i+1] - lx[i-1])
            dly_dx[i]  = (0.5/dx) * (ly[i+1] - ly[i-1])
            dlz_dx[i]  = (0.5/dx) * (lz[i+1] - lz[i-1])

            psi[i]      = (dlx_dx[i]**2. + dly_dx[i]**2. + dlz_dx[i]**2.)**0.5
            if (dim_type == 0): # 1D Q tables
                Q1[i]       = -1.0*(10**(interp_1d(s_arr,Q1_1d_arr,psi[i],ng_Q)))
                Q2[i]       = 10**(interp_1d(s_arr,Q2_1d_arr,psi[i],ng_Q))
                Q3[i]       = interp_1d(s_arr,Q3_1d_arr,psi[i],ng_Q)#10**(interp_1d(s_arr,Q3_arr,psi[i],ng_Q))
                dQ1_dpsi[i] = interp_1d(s_arr,dQ1_dpsi_1d_arr,psi[i],ng_Q)#10**(interp_1d(s_arr,dQ1_dpsi_arr,psi[i],ng_Q))
            elif (dim_type == 1): # 2D Q_tables
                Q1[i]       = -1.0*(10**(interp_2d(r_arr,s_arr,Q1_2d_arr,r[i],psi[i],ng_Q)))
                Q2[i]       = 10**(interp_2d(r_arr,s_arr,Q2_2d_arr,r[i],psi[i],ng_Q))
                Q3[i]       = interp_2d(r_arr,s_arr,Q3_2d_arr,r[i],psi[i],ng_Q)#10**(interp_1d(s_arr,Q3_arr,psi[i],ng_Q))
                dQ1_dpsi[i] = interp_2d(r_arr,s_arr,dQ1_dpsi_2d_arr,r[i],psi[i],ng_Q)#10**(interp_1d(s_arr,dQ1_dpsi_arr,psi[i],ng_Q))
            dpsi_dx[i] = (0.5/dx) * (psi[i+1] - psi[i-1])

        # only do the cfl on predictor step
        if (predictor or (not do_predictor)): 
            ## cfl condition
            # Here, we do something qualitatively similar to Equation (50) of Diego Munoz 2012
            dt = 1000000000000000000.
            for i in range(1,ngrid-1):
                vel = (HoR**2.)*fmax(fabs(-Q1[i] + 2.*dpsi_dx[i]*dQ1_dpsi[i] + 2.*Q2[i]*psi[i]), fabs((1./L[i]**2.)*(L[i]**2.*(-1.*Q1[i] + 2.*dQ1_dpsi[i]*dpsi_dx[i] + 2.*Q2[i]*psi[i]**2.) + (Lx[i]*dLx_dx[i] + Ly[i]*dLy_dx[i] + Lz[i]*dLz_dx[i])*(2.*Q1[i] + Q2[i])))) 
                nu  = (HoR**2.)*(Q1[i]**2. + Q2[i]**2. + Q3[i]**2.)**(0.5)
                if (vel > velmax): printf("Wave speed eclipses half the sound speed at i = %d!\n",i)
                #vel = fmin(vel,velmax)
                dt = fmin(dt,fabs(cfl*(dx/vel/(1. + 2.*nu/(vel*dx)))))
            if predictor: dt = dt/2. # this is for half-timestep evolution; will multiply by two afterwards

        ## get cell-interface gradients

        for i in range(1,ngrid-1):
            ## Try minmod slope limiter
            # L
            tmp_slope = minmod( (dLx_dx[i] - dLx_dx[i-1])/dx, (dLx_dx[i+1] - dLx_dx[i])/dx)
            dLx_dx_L[i] = dLx_dx[i] - tmp_slope*dx/2.
            dLx_dx_R[i] = dLx_dx[i] + tmp_slope*dx/2.
            tmp_slope = minmod( (dLy_dx[i] - dLy_dx[i-1])/dx, (dLy_dx[i+1] - dLy_dx[i])/dx)
            dLy_dx_L[i] = dLy_dx[i] - tmp_slope*dx/2.
            dLy_dx_R[i] = dLy_dx[i] + tmp_slope*dx/2.
            tmp_slope = minmod( (dLz_dx[i] - dLz_dx[i-1])/dx, (dLz_dx[i+1] - dLz_dx[i])/dx)
            dLz_dx_L[i] = dLz_dx[i] - tmp_slope*dx/2.
            dLz_dx_R[i] = dLz_dx[i] + tmp_slope*dx/2.
            # l
            tmp_slope = minmod( (dlx_dx[i] - dlx_dx[i-1])/dx, (dlx_dx[i+1] - dlx_dx[i])/dx)
            dlx_dx_L[i] = dlx_dx[i] - tmp_slope*dx/2.
            dlx_dx_R[i] = dlx_dx[i] + tmp_slope*dx/2.
            tmp_slope = minmod( (dly_dx[i] - dly_dx[i-1])/dx, (dly_dx[i+1] - dly_dx[i])/dx)
            dly_dx_L[i] = dly_dx[i] - tmp_slope*dx/2.
            dly_dx_R[i] = dly_dx[i] + tmp_slope*dx/2.
            tmp_slope = minmod( (dlz_dx[i] - dlz_dx[i-1])/dx, (dlz_dx[i+1] - dlz_dx[i])/dx)
            dlz_dx_L[i] = dlz_dx[i] - tmp_slope*dx/2.
            dlz_dx_R[i] = dlz_dx[i] + tmp_slope*dx/2.
            # Q
            tmp_slope = minmod( (Q1[i] - Q1[i-1])/dx, (Q1[i+1] - Q1[i])/dx)
            Q1_L[i] = Q1[i] - tmp_slope*dx/2.
            Q1_R[i] = Q1[i] + tmp_slope*dx/2.
            tmp_slope = minmod( (Q2[i] - Q2[i-1])/dx, (Q2[i+1] - Q2[i])/dx)
            Q2_L[i] = Q2[i] - tmp_slope*dx/2.
            Q2_R[i] = Q2[i] + tmp_slope*dx/2.
            tmp_slope = minmod( (Q3[i] - Q3[i-1])/dx, (Q3[i+1] - Q3[i])/dx)
            Q3_L[i] = Q3[i] - tmp_slope*dx/2.
            Q3_R[i] = Q3[i] + tmp_slope*dx/2.
            # psi
            tmp_slope = minmod( (psi[i] - psi[i-1])/dx, (psi[i+1] - psi[i])/dx)
            psi_L[i] = psi[i] - tmp_slope*dx/2.
            psi_R[i] = psi[i] + tmp_slope*dx/2.
            # dpsi_dx
            tmp_slope = minmod( (dpsi_dx[i] - dpsi_dx[i-1])/dx, (dpsi_dx[i+1] - dpsi_dx[i])/dx)
            dpsi_dx_L[i] = dpsi_dx[i] - tmp_slope*dx/2.
            dpsi_dx_R[i] = dpsi_dx[i] + tmp_slope*dx/2.
            # dQ1_dpsi
            tmp_slope = minmod( (dQ1_dpsi[i] - dQ1_dpsi[i-1])/dx, (dQ1_dpsi[i+1] - dQ1_dpsi[i])/dx)
            dQ1_dpsi_L[i] = dQ1_dpsi[i] - tmp_slope*dx/2.
            dQ1_dpsi_R[i] = dQ1_dpsi[i] + tmp_slope*dx/2.

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

        # for PLM, lets fill guard cells
        Lx_L[0] = Lx_L[1]
        Lx_R[0] = Lx_R[1]
        Lx_L[ngrid] = Lx_L[ngrid-1]
        Lx_R[ngrid] = Lx_R[ngrid-1]
        Ly_L[0] = Ly_L[1]
        Ly_R[0] = Ly_R[1]
        Ly_L[ngrid] = Ly_L[ngrid-1]
        Ly_R[ngrid] = Ly_R[ngrid-1]
        Lz_L[0] = Lz_L[1]
        Lz_R[0] = Lz_R[1]
        Lz_L[ngrid] = Lz_L[ngrid-1]
        Lz_R[ngrid] = Lz_R[ngrid-1]
        L_L[0]  = (Lx_L[1]**2. + Ly_L[1]**2. + Lz_L[1]**2.)**0.5
        L_R[0]  = (Lx_R[1]**2. + Ly_R[1]**2. + Lz_R[1]**2.)**0.5
        L_L[ngrid]  = (Lx_L[ngrid-1]**2. + Ly_L[ngrid-1]**2. + Lz_L[ngrid-1]**2.)**0.5
        L_R[ngrid]  = (Lx_R[ngrid-1]**2. + Ly_R[ngrid-1]**2. + Lz_R[ngrid-1]**2.)**0.5

        ## evolve (get fluxes)
        F_all = flux_calc(HoR, velmax, dt, ngrid,  Lx_L,  Lx_R,  Ly_L,  Ly_R,  Lz_L,  Lz_R,  L_L,  L_R,  dLx_dx_L,  dLx_dx_R,  dLy_dx_L,  dLy_dx_R,  dLz_dx_L,  dLz_dx_R,  dlx_dx_L,  dlx_dx_R,  dly_dx_L,  dly_dx_R,  dlz_dx_L,  dlz_dx_R,  psi_L,  psi_R,  Q1_L,  Q1_R,  Q2_L,  Q2_R,  Q3_L,  Q3_R,  dQ1_dpsi_L,  dQ1_dpsi_R,  dpsi_dx_L,  dpsi_dx_R)
        F_x = F_all[0]
        F_y = F_all[1]
        F_z = F_all[2]

        ## update
        for i in range(2,ngrid-2):
            Lx[i] = Lx_old[i] - (dt/dx)*(F_x[i-1] - F_x[i-2])
            Ly[i] = Ly_old[i] - (dt/dx)*(F_y[i-1] - F_y[i-2])
            Lz[i] = Lz_old[i] - (dt/dx)*(F_z[i-1] - F_z[i-2])

        for i in range(ngrid):
            ## Update external torques
            Lx_tmp = Lx_old[i]*cos(omega_p_z[i]*dt) - Ly_old[i]*sin(omega_p_z[i]*dt)
            Ly_tmp = Lx_old[i]*sin(omega_p_z[i]*dt) + Ly_old[i]*cos(omega_p_z[i]*dt)
            Lx[i] = 1.0*Lx_tmp
            Ly[i] = 1.0*Ly_tmp


            L[i]  = (Lx[i]**2. + Ly[i]**2. + Lz[i]**2.)**0.5
            lx[i] = Lx[i]/L[i]
            ly[i] = Ly[i]/L[i]
            lz[i] = Lz[i]/L[i]

        ## apply boundary conditions
        ## Weight by R so sigma is continuous! L should have profile at boundaries
        ## Two guard cells on each end
        if   (bc_type==0): #### Apply sink boundary conditions
            ## Lx
            Lx[0] = 1e-10 * Lx[2]*(r[2]/r[0])**(-1.)
            Lx[1] = 1e-10 * Lx[2]*(r[2]/r[1])**(-1.)
            Lx[ngrid-1] = 1e-10 * Lx[ngrid-3]*(r[ngrid-3]/r[ngrid-1])**(-1.)
            Lx[ngrid-2] = 1e-10 * Lx[ngrid-3]*(r[ngrid-3]/r[ngrid-2])**(-1.)
            ## Ly
            Ly[0] = 1e-10 * Ly[2]*(r[2]/r[0])**(-1.)
            Ly[1] = 1e-10 * Ly[2]*(r[2]/r[1])**(-1.)
            Ly[ngrid-1] = 1e-10 * Ly[ngrid-3]*(r[ngrid-3]/r[ngrid-1])**(-1.)
            Ly[ngrid-2] = 1e-10 * Ly[ngrid-3]*(r[ngrid-3]/r[ngrid-2])**(-1.)
            ## Lz
            Lz[0] = 1e-10 * Lz[2]*(r[2]/r[0])**(-1.)
            Lz[1] = 1e-10 * Lz[2]*(r[2]/r[1])**(-1.)
            Lz[ngrid-1] = 1e-10 * Lz[ngrid-3]*(r[ngrid-3]/r[ngrid-1])**(-1.)
            Lz[ngrid-2] = 1e-10 * Lz[ngrid-3]*(r[ngrid-3]/r[ngrid-2])**(-1.)

        elif (bc_type==1): #### Apply outflow boundary conditions
            ## Lx
            Lx[0] = Lx[2]*(r[2]/r[0])**(-1.)
            Lx[1] = Lx[2]*(r[2]/r[1])**(-1.)
            Lx[ngrid-1] = Lx[ngrid-3]*(r[ngrid-3]/r[ngrid-1])**(-1.)
            Lx[ngrid-2] = Lx[ngrid-3]*(r[ngrid-3]/r[ngrid-2])**(-1.)
            ## Ly
            Ly[0] = Ly[2]*(r[2]/r[0])**(-1.)
            Ly[1] = Ly[2]*(r[2]/r[1])**(-1.)
            Ly[ngrid-1] = Ly[ngrid-3]*(r[ngrid-3]/r[ngrid-1])**(-1.)
            Ly[ngrid-2] = Ly[ngrid-3]*(r[ngrid-3]/r[ngrid-2])**(-1.)
            ## Lz
            Lz[0] = Lz[2]*(r[2]/r[0])**(-1.)
            Lz[1] = Lz[2]*(r[2]/r[1])**(-1.)
            Lz[ngrid-1] = Lz[ngrid-3]*(r[ngrid-3]/r[ngrid-1])**(-1.)
            Lz[ngrid-2] = Lz[ngrid-3]*(r[ngrid-3]/r[ngrid-2])**(-1.)

        elif (bc_type==2): #### Apply outflow outer and sink inner boundary conditions
            ## Lx
            Lx[0] = 1e-10 * Lx[2]*(r[2]/r[0])**(-1.)
            Lx[1] = Lx[2]*(r[2]/r[1])**(-1.)
            Lx[ngrid-1] = Lx[ngrid-3]*(r[ngrid-3]/r[ngrid-1])**(-1.)
            Lx[ngrid-2] = Lx[ngrid-3]*(r[ngrid-3]/r[ngrid-2])**(-1.)
            ## Ly
            Ly[0] = 1e-10 * Ly[2]*(r[2]/r[0])**(-1.)
            Ly[1] = Ly[2]*(r[2]/r[1])**(-1.)
            Ly[ngrid-1] = Ly[ngrid-3]*(r[ngrid-3]/r[ngrid-1])**(-1.)
            Ly[ngrid-2] = Ly[ngrid-3]*(r[ngrid-3]/r[ngrid-2])**(-1.)
            ## Lz
            Lz[0] = 1e-10 * Lz[2]*(r[2]/r[0])**(-1.)
            Lz[1] = Lz[2]*(r[2]/r[1])**(-1.)
            Lz[ngrid-1] = Lz[ngrid-3]*(r[ngrid-3]/r[ngrid-1])**(-1.)
            Lz[ngrid-2] = Lz[ngrid-3]*(r[ngrid-3]/r[ngrid-2])**(-1.)

        elif (bc_type==3): #### Infinite disk
            ## Lx
            Lx[0] = Lx[2]*(r[2]/r[0])**(-1.)
            Lx[1] = Lx[2]*(r[2]/r[1])**(-1.)
            Lx[ngrid-1] = Lx_inf[1] 
            Lx[ngrid-2] = Lx_inf[0]
            ## Ly
            Ly[0] = Ly[2]*(r[2]/r[0])**(-1.)
            Ly[1] = Ly[2]*(r[2]/r[1])**(-1.)
            Ly[ngrid-1] = Ly_inf[1]
            Ly[ngrid-2] = Ly_inf[0]
            ## Lz
            Lz[0] = Lz[2]*(r[2]/r[0])**(-1.)
            Lz[1] = Lz[2]*(r[2]/r[1])**(-1.)
            Lz[ngrid-1] = Lz_inf[1]
            Lz[ngrid-2] = Lz_inf[0]


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

            if (( (t/tmax)%(1e-5) < dt)):
                printf("t/tmax = %e, dt/tmax = %e\n",t/tmax,dt/tmax)
                

            if (nstep%io_freq == 0):
                for i in range(ngrid):
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
                    fclose(f_out)

    # do final output
    printf("Final output...\n")
    for i in range(ngrid):
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
        fclose(f_out)
