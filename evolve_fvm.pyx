"""
1D evolution of twisted accretion disks
 
Finite volume method
"""
 
import numpy as np
cimport numpy as np
cimport cython
from libc.stdio cimport printf,fopen,fclose,fprintf,FILE,sprintf
from libc.stdlib cimport malloc
from libc.math cimport fmin, fmax, fabs, isnan

 
cdef double mycreal(double complex dc):
    cdef double complex* dcptr = &dc
    return (<double *>dcptr)[0]
 
cdef double mycimag(double complex dc):
    cdef double complex* dcptr = &dc
    return (<double *>dcptr)[1]
 
## Helper functions ##
 
def load_Q(path):
    # Builds Q1,Q2 or Q3 from path. Currently assumes 1D file. 
    data = open(path,"r")
    parsed = []
    for line in data:
        parsed.append(np.array(line.split()).astype(float))
    return np.array(parsed)[:,0]

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
def evolve(*p):
    ## Get params 
    cdef double alpha, gamma, HoR, tilt, bhspin, r0, rw, rmin, rmax, tmax, io_freq, smax, cfl
    cdef int    ngrid,bc_type
    cdef bint   dolog
    cdef char*  io_prefix
    alpha,gamma,HoR,tilt,bhspin,r0,rw,rmin,rmax,rho_type,tmax,cfl,ngrid,bc,io_freq,io_prefix,Q_dim,smax,Q1_path,Q2_path,Q3_path = p
    # change tilt to radians
    tilt *= np.pi/180.

    ## Build interpolation functions 

    # Parse data 
    Q1_parsed = load_Q(Q1_path)
    Q2_parsed = load_Q(Q2_path)
    Q3_parsed = load_Q(Q3_path)
    cdef int ng_Q = len(Q1_parsed) # Get length of Q

    # Psi array that Q1/Q2/Q3 correspond to
    _s_arr = np.logspace(0,np.log10(smax+1),ng_Q) - 1
 
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
        _sigma = np.ones(ngrid)
    elif rho_type == "gauss":
        _sigma = (1.0/(rw*np.sqrt(2.0*np.pi))) * np.exp(-0.5*((_r - r0)/rw)**2.0)
        _sigma /= np.average(_sigma)
    else:
        print "Error! rho_type needs to be set to \"gauss\" or \"flat\"! Exiting"
        exit()

    # build angular momentum quantities
    # Here we are using Lambda = I * Omega * r**2., where I = (h/r)^2 * Sigma * r^2 (assuming gamma = 1)
    lda_mag = (HoR)**2. * _sigma * omega * _r**4.
    lda_unit = np.array([np.array([np.sin(tilt),0.0,np.cos(tilt)])]*ngrid) # each annulus oriented in the same direction initially
    _lda_vec  = np.copy(lda_unit) # building [Lambda_x,Lambda_y,Lambda_z] for each radial grid element
    for j in range(3): _lda_vec[:,j] *= lda_mag
 
 
    # for Lense-Thirring source term
    _omega_p = np.zeros(3*ngrid)
    _omega_p = np.reshape(_omega_p, (ngrid,3))
    _omega_p[:,2] = 2.0 * bhspin / _r**3.0 # x/y components are zero, z component is LT precession frequency
 
    # calculate (approximate) viscous time (t_visc = r0**2./nu1(psi=0))
    nu_ref    = (-2.0/3.0)*(-1.0*10**(interp_1d(_s_arr,np.log10(-Q1_parsed + 1e-30),0,ng_Q)))*((HoR**2.0)*r0**0.5)
    t_viscous = r0*r0/nu_ref
 
    # convert tmax, io_freq from t_viscous units to code units
    tmax    = tmax*t_viscous
    io_freq = io_freq*t_viscous
 
    # make bc_type
    if (bc == 'sink'):
        bc_type = 0
    elif (bc == 'outflow'):
        bc_type = 1
    else:
        print "Error! bc needs to be set to \"sink\" or \"outflow\"! Exiting"
        exit()
 
 
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
    print "cfl       = %s\n" % cfl
    print "bc        = %s\n" % bc
    print "io_freq   = %s [t_viscous]\n" % io_freq
    print "io_prefix = %s\n" % io_prefix
    print "Q_dim     = %s\n" % Q_dim
    print "smax      = %s\n" % smax
    print "Q1_path   = %s\n" % Q1_path
    print "Q2_path   = %s\n" % Q2_path
    print "Q3_path   = %s\n" % Q3_path
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
    cdef double[:] omega_p_z = np.copy(_omega_p[:,2])

    cdef double[:] psi       = np.zeros(ngrid)
    cdef double[:] Q1        = np.zeros(ngrid)
    cdef double[:] Q2        = np.zeros(ngrid)
    cdef double[:] Q3        = np.zeros(ngrid)
    cdef double[:] dQLrdx    = np.zeros(ngrid)
    cdef double[:] dldx_x    = np.zeros(ngrid)
    cdef double[:] dldx_y    = np.zeros(ngrid)
    cdef double[:] dldx_z    = np.zeros(ngrid)
    cdef double[:] Lxdldx_x  = np.zeros(ngrid)
    cdef double[:] Lxdldx_y  = np.zeros(ngrid)
    cdef double[:] Lxdldx_z  = np.zeros(ngrid)

    cdef double[:] Lx_L      = np.zeros(ngrid)
    cdef double[:] Lx_R      = np.zeros(ngrid)
    cdef double[:] Ly_L      = np.zeros(ngrid)
    cdef double[:] Ly_R      = np.zeros(ngrid)
    cdef double[:] Lz_L      = np.zeros(ngrid)
    cdef double[:] Lz_R      = np.zeros(ngrid)
    cdef double[:] L_L       = np.zeros(ngrid)
    cdef double[:] L_R       = np.zeros(ngrid)

    # Fluxes
    cdef double[:] F_x       = np.zeros(ngrid)
    cdef double[:] F_x_L     = np.zeros(ngrid)
    cdef double[:] F_x_R     = np.zeros(ngrid)
    cdef double[:] F_y       = np.zeros(ngrid)
    cdef double[:] F_y_L     = np.zeros(ngrid)
    cdef double[:] F_y_R     = np.zeros(ngrid)
    cdef double[:] F_z       = np.zeros(ngrid)
    cdef double[:] F_z_L     = np.zeros(ngrid)
    cdef double[:] F_z_R     = np.zeros(ngrid)
    cdef double[:] fAdv_x_L  = np.zeros(ngrid)
    cdef double[:] fAdv_x_R  = np.zeros(ngrid)
    cdef double[:] fAdv_y_L  = np.zeros(ngrid)
    cdef double[:] fAdv_y_R  = np.zeros(ngrid)
    cdef double[:] fAdv_z_L  = np.zeros(ngrid)
    cdef double[:] fAdv_z_R  = np.zeros(ngrid)
    cdef double[:] fDif_x_L  = np.zeros(ngrid)
    cdef double[:] fDif_x_R  = np.zeros(ngrid)
    cdef double[:] fDif_y_L  = np.zeros(ngrid)
    cdef double[:] fDif_y_R  = np.zeros(ngrid)
    cdef double[:] fDif_z_L  = np.zeros(ngrid)
    cdef double[:] fDif_z_R  = np.zeros(ngrid)

    # miscellaneous variables
    cdef double small = 1e-30
    cdef double[:] Q1_arr = np.log10(-Q1_parsed + small)
    cdef double[:] Q2_arr = np.log10(Q2_parsed + small)
    cdef double[:] Q3_arr = np.log10(Q3_parsed + small)
    cdef double[:] s_arr = _s_arr
    cdef double dt = 100000.
    cdef double vL,vR,sL,sR,vel,nu
    cdef int i

    # io variables
    cdef FILE      *f_out
    cdef char[40]  io_fn
    cdef int       io_cnt = 0

    # iterate!
    t = 0.
    while (t < tmax):
 
        #### Save before updates
        if ((t%io_freq < dt)):
            printf("t/tmax = %e, dt/tmax = %e\n",t/tmax,dt/tmax)
            sprintf(io_fn,"%s%d.csv",io_prefix,io_cnt)

            f_out = fopen(io_fn,"w")
            for i in range(ngrid):
                fprintf(f_out, "%e ", Lx[i])
                fprintf(f_out, "%e ", Ly[i])
                fprintf(f_out, "%e ", Lz[i])
                fprintf(f_out, "%e ", r[i])
                fprintf(f_out, "%e ", Q1[i])
                fprintf(f_out, "%e ", Q2[i])
                fprintf(f_out, "%e ", Q3[i])
                fprintf(f_out, "\n")
            fclose(f_out)
            io_cnt += 1


        ### Reconstruct, Evolve, Average algorithm

        ## calculate cell-centered arguments in flux terms

        for i in range(1,ngrid-1):
            L[i]   = (Lx[i]**2. + Ly[i]**2. + Lz[i])**0.5
            psi[i] = (0.5/dx)*( (lx[i+1] - lx[i-1])**2. + (ly[i+1] - ly[i-1])**2. + (lz[i+1] - lz[i-1])**2.)**0.5
            Q1[i]  = (-1.0*10**(interp_1d(s_arr,Q1_arr,psi[i],ng_Q)))
            Q2[i]  = 10**(interp_1d(s_arr,Q2_arr,psi[i],ng_Q))
            Q3[i]  = 10**(interp_1d(s_arr,Q3_arr,psi[i],ng_Q))

            # center differencing for diffusive terms
            dQLrdx[i]   = (0.5/dx)*(Q1[i+1]*L[i+1]*r[i+1]**(-1.5) - Q1[i-1]*L[i-1]*r[i-1]**(-1.5))
            dldx_x[i]   = (0.5/dx)*(lx[i+1] - lx[i-1])
            dldx_y[i]   = (0.5/dx)*(ly[i+1] - ly[i-1])
            dldx_z[i]   = (0.5/dx)*(lz[i+1] - lz[i-1])
            Lxdldx_x[i] = Ly[i]*dldx_z[i] - Lz[i]*dldx_y[i]
            Lxdldx_y[i] = Lz[i]*dldx_x[i] - Lx[i]*dldx_z[i]
            Lxdldx_z[i] = Lx[i]*dldx_y[i] - Ly[i]*dldx_x[i]


        ## need to fill guard cell values for these quantities as well
        L[0]              = (Lx[0]**2. + Ly[0]**2. + Lz[0])**0.5
        L[ngrid-1]        = (Lx[ngrid-1]**2. + Ly[ngrid-1]**2. + Lz[ngrid-1])**0.5
        psi[0]            = psi[1]
        psi[ngrid-1]      = psi[ngrid-2]
        Q1[0]             = Q1[1]
        Q1[ngrid-1]       = Q1[ngrid-2]
        Q2[0]             = Q2[1]
        Q2[ngrid-1]       = Q2[ngrid-2]
        Q3[0]             = Q3[1]
        Q3[ngrid-1]       = Q3[ngrid-2]
        dQLrdx[0]         = dQLrdx[1]
        dQLrdx[ngrid-1]   = dQLrdx[ngrid-2]
        dldx_x[0]         = dldx_x[1]
        dldx_x[ngrid-1]   = dldx_x[ngrid-2]
        dldx_y[0]         = dldx_y[1]
        dldx_y[ngrid-1]   = dldx_y[ngrid-2]
        dldx_z[0]         = dldx_z[1]
        dldx_z[ngrid-1]   = dldx_z[ngrid-2]
        Lxdldx_x[0]       = Lxdldx_x[1]
        Lxdldx_x[ngrid-1] = Lxdldx_x[ngrid-2]
        Lxdldx_y[0]       = Lxdldx_y[1]
        Lxdldx_y[ngrid-1] = Lxdldx_y[ngrid-2]
        Lxdldx_z[0]       = Lxdldx_z[1]
        Lxdldx_z[ngrid-1] = Lxdldx_z[ngrid-2]

        ## reconstruct

        for i in range(ngrid):
            # first order godunov
            Lx_L[i] = Lx[i]
            Lx_R[i] = Lx[i]
            Ly_L[i] = Ly[i]
            Ly_R[i] = Ly[i]
            Lz_L[i] = Lz[i]
            Lz_R[i] = Lz[i]
            L_L[i]  = (Lx_L[i]**2. + Ly_L[i]**2. + Lz_L[i]**2.)**0.5
            L_R[i]  = (Lx_R[i]**2. + Ly_R[i]**2. + Lz_R[i]**2.)**0.5


            ### do left flux first from (1:ngrid-1) inclusive
            if (i > 0):
                ## advective fluxes at cell interfaces
                fAdv_x_L[i] = -0.5*( (r[i]**(-3./2.))*(Q1[i]+2.*Q2[i]*psi[i]**2.) + (r[i-1]**(-3./2.))*(Q1[i-1]+2.*Q2[i-1]*psi[i-1]**2.) ) * Lx_L[i]
                fAdv_y_L[i] = -0.5*( (r[i]**(-3./2.))*(Q1[i]+2.*Q2[i]*psi[i]**2.) + (r[i-1]**(-3./2.))*(Q1[i-1]+2.*Q2[i-1]*psi[i-1]**2.) ) * Ly_L[i]
                fAdv_z_L[i] = -0.5*( (r[i]**(-3./2.))*(Q1[i]+2.*Q2[i]*psi[i]**2.) + (r[i-1]**(-3./2.))*(Q1[i-1]+2.*Q2[i-1]*psi[i-1]**2.) ) * Lz_L[i]
                ## diffusive fluxes at cell interfaces
                # 1st term
                fDif_x_L[i] = -2.*(Lx_L[i]/(L_L[i]+small)) * 0.5 * (dQLrdx[i] + dQLrdx[i-1])
                fDif_y_L[i] = -2.*(Ly_L[i]/(L_L[i]+small)) * 0.5 * (dQLrdx[i] + dQLrdx[i-1])
                fDif_z_L[i] = -2.*(Lz_L[i]/(L_L[i]+small)) * 0.5 * (dQLrdx[i] + dQLrdx[i-1])
                # 2nd term
                fDif_x_L[i] +=  L_L[i]*0.5*(Q2[i]*r[i]**(-1.5)*dldx_x[i] + Q2[i-1]*r[i-1]**(-1.5)*dldx_x[i-1])
                fDif_y_L[i] +=  L_L[i]*0.5*(Q2[i]*r[i]**(-1.5)*dldx_y[i] + Q2[i-1]*r[i-1]**(-1.5)*dldx_y[i-1])
                fDif_z_L[i] +=  L_L[i]*0.5*(Q2[i]*r[i]**(-1.5)*dldx_z[i] + Q2[i-1]*r[i-1]**(-1.5)*dldx_z[i-1])
                # 3rd term
                fDif_x_L[i] += -0.5*(Q3[i]*r[i]**(-1.5)*Lxdldx_x[i] + Q3[i-1]*r[i-1]**(-1.5)*Lxdldx_x[i-1])
                fDif_y_L[i] += -0.5*(Q3[i]*r[i]**(-1.5)*Lxdldx_y[i] + Q3[i-1]*r[i-1]**(-1.5)*Lxdldx_y[i-1])
                fDif_z_L[i] += -0.5*(Q3[i]*r[i]**(-1.5)*Lxdldx_z[i] + Q3[i-1]*r[i-1]**(-1.5)*Lxdldx_z[i-1])

                ## Combine
                F_x_L[i] = (HoR**2.)*(fDif_x_L[i] + fAdv_x_L[i])
                F_y_L[i] = (HoR**2.)*(fDif_y_L[i] + fAdv_y_L[i])
                F_z_L[i] = (HoR**2.)*(fDif_z_L[i] + fAdv_z_L[i])
            else:
                F_x_L[i] = 0.
                F_y_L[i] = 0.
                F_z_L[i] = 0.
 
            ### do right flux from (0:ngrid-2) inclusive
            if (i < ngrid-1):
                ## advective fluxes at cell interfaces
                fAdv_x_R[i] = -0.5*( (r[i]**(-3./2.))*(Q1[i]+2.*Q2[i]*psi[i]**2.) + (r[i+1]**(-3./2.))*(Q1[i+1]+2.*Q2[i+1]*psi[i+1]**2.) ) * Lx_R[i]
                fAdv_y_R[i] = -0.5*( (r[i]**(-3./2.))*(Q1[i]+2.*Q2[i]*psi[i]**2.) + (r[i+1]**(-3./2.))*(Q1[i+1]+2.*Q2[i+1]*psi[i+1]**2.) ) * Ly_R[i]
                fAdv_z_R[i] = -0.5*( (r[i]**(-3./2.))*(Q1[i]+2.*Q2[i]*psi[i]**2.) + (r[i+1]**(-3./2.))*(Q1[i+1]+2.*Q2[i+1]*psi[i+1]**2.) ) * Lz_R[i]
                ## diffusive fluxes at cell interfaces
                # 1st term
                fDif_x_R[i] = -2.*(Lx_R[i]/(L_R[i]+small)) * 0.5 * (dQLrdx[i] + dQLrdx[i+1])
                fDif_y_R[i] = -2.*(Ly_R[i]/(L_R[i]+small)) * 0.5 * (dQLrdx[i] + dQLrdx[i+1])
                fDif_z_R[i] = -2.*(Lz_R[i]/(L_R[i]+small)) * 0.5 * (dQLrdx[i] + dQLrdx[i+1])
                # 2nd term
                fDif_x_R[i] +=  L_L[i]*0.5*(Q2[i]*r[i]**(-1.5)*dldx_x[i] + Q2[i+1]*r[i+1]**(-1.5)*dldx_x[i+1])
                fDif_y_R[i] +=  L_L[i]*0.5*(Q2[i]*r[i]**(-1.5)*dldx_y[i] + Q2[i+1]*r[i+1]**(-1.5)*dldx_y[i+1])
                fDif_z_R[i] +=  L_L[i]*0.5*(Q2[i]*r[i]**(-1.5)*dldx_z[i] + Q2[i+1]*r[i+1]**(-1.5)*dldx_z[i+1])
                # 3rd term
                fDif_x_R[i] += -0.5*(Q3[i]*r[i]**(-1.5)*Lxdldx_x[i] + Q3[i+1]*r[i+1]**(-1.5)*Lxdldx_x[i+1])
                fDif_y_R[i] += -0.5*(Q3[i]*r[i]**(-1.5)*Lxdldx_y[i] + Q3[i+1]*r[i+1]**(-1.5)*Lxdldx_y[i+1])
                fDif_z_R[i] += -0.5*(Q3[i]*r[i]**(-1.5)*Lxdldx_z[i] + Q3[i+1]*r[i+1]**(-1.5)*Lxdldx_z[i+1])

                ## Combine
                F_x_R[i] = (HoR**2.)*(fDif_x_R[i] + fAdv_x_R[i])
                F_y_R[i] = (HoR**2.)*(fDif_y_R[i] + fAdv_y_R[i])
                F_z_R[i] = (HoR**2.)*(fDif_z_R[i] + fAdv_z_R[i])
            else:
                F_x_R[i] = 0.
                F_y_R[i] = 0.
                F_z_R[i] = 0.

        ## evolve (get fluxes)

        # hll
        for i in range(1,ngrid-1):
            vL = F_x_L[i]/(Lx_L[i]+small) + F_y_L[i]/(Ly_L[i]+small) + F_z_L[i]/(Lz_L[i]+small)
            vR = F_x_R[i]/(Lx_R[i]+small) + F_y_R[i]/(Ly_R[i]+small) + F_z_R[i]/(Lz_R[i]+small)
            sL = fmin(vL,vR)
            sR = fmax(vL,vR)

            # Equations 9.82-9.91 of dongwook ams notes
            if (sL >= 0.):
                F_x[i] = F_x_L[i]
                F_y[i] = F_y_L[i]
                F_z[i] = F_z_L[i]
            elif ((sL < 0.) and (sR >= 0.)):
                F_x[i] = F_x_L[i] + sL*((sR*Lx_R[i] - sL*Lx_L[i] + F_x_L[i] - F_x_R[i])/(sR-sL+small) - Lx_L[i])
                F_y[i] = F_y_L[i] + sL*((sR*Ly_R[i] - sL*Ly_L[i] + F_y_L[i] - F_y_R[i])/(sR-sL+small) - Ly_L[i])
                F_z[i] = F_z_L[i] + sL*((sR*Lz_R[i] - sL*Lz_L[i] + F_z_L[i] - F_z_R[i])/(sR-sL+small) - Lz_L[i])
            else:
                F_x[i] = F_x_R[i]
                F_y[i] = F_y_R[i]
                F_z[i] = F_z_R[i]

        ## cfl condition
        # Here, we do something qualitatively similar to Equation (50) of Diego Munoz 2012
        for i in range(1,ngrid-1):
            #vel = fabs((HoR**2.)*r[i]*( (2/L[i])*dQLrdx[i] - 2.*Q2[i]*(r[i]**(-1.5))*psi[i]**2.))
            vel = (HoR**2.)*fabs(r[i]**(-1.5) * (Q1[i] + 2.*Q2[i]*psi[i]**2.))
            nu  = (HoR**2.)*(r[i]**0.5)*((1./9.)*Q1[i]**2. + 4.*Q2[i]**2. + Q3[i]**2.)**(0.5)
            dt = fmin(dt,cfl*(dx/vel))#/(1. + 2.*nu/(vel*dx)))

        ## update
        for i in range(1,ngrid-1):
            Lx[i] = Lx[i] - (dt/dx)*(F_x[i+1] - F_x[i])
            Ly[i] = Ly[i] - (dt/dx)*(F_y[i+1] - F_y[i])
            Lz[i] = Lz[i] - (dt/dx)*(F_z[i+1] - F_z[i])
            Lx[i] = fmax(Lx[i],1e-1)
            Ly[i] = fmax(Ly[i],1e-1)
            Lz[i] = fmax(Lz[i],1e-1)

        if(isnan(Lx[i]) or isnan(Ly[i]) or isnan(Lz[i])):
            printf("nans ):\n")
            printf("Lx[i] = %16.8f, Ly[i] = %16.8f, Lz[i] = %16.8f\n",Lx[i],Ly[i],Lz[i])
            printf("F_x[i] = %16.8f, F_y[i] = %16.8f, F_z[i] = %16.8f\n",F_x[i],F_y[i],F_z[i])

        ## apply boundary conditions
        if   (bc_type==0): #### Apply sink boundary conditions
            Lx[0] = 1e-10 * Lx[1]
            Lx[ngrid-1] = 1e-10 * Lx[ngrid-2]
            Ly[0] = 1e-10 * Ly[1]
            Ly[ngrid-1] = 1e-10 * Ly[ngrid-2]
            Lz[0] = 1e-10 * Lz[1]
            Lz[ngrid-1] = 1e-10 * Lz[ngrid-2]
        elif (bc_type==1): #### Apply outflow boundary conditions
            Lx[0] = Lx[1]
            Lx[ngrid-1] = Lx[ngrid-2]
            Ly[0] = Ly[1]
            Ly[ngrid-1] = Ly[ngrid-2]
            Lz[0] = Lz[1]
            Lz[ngrid-1] = Lz[ngrid-2]

        #### Update timestep
        t += dt

