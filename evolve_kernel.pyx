"""
1D evolution of twisted accretion disks
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.stdio cimport printf,fopen,fclose,fprintf,FILE,sprintf
from libc.stdlib cimport malloc
from libc.math cimport fmin, fmax, fabs

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
            

#########


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def evolve(*p):
    ## Get params 
    cdef double alpha, gamma, HoR, tilt, bhspin, r0, rw, rmin, rmax, tmax, io_freq, smax, cfl
    cdef int    ngrid,bc_type
    cdef bint   dolog
    cdef char*  io_prefix
    alpha,gamma,HoR,tilt,bhspin,r0,rw,rmin,rmax,rho_type,tmax,cfl,ngrid,dolog,bc,io_freq,io_prefix,Q_dim,smax,Q1_path,Q2_path,Q3_path = p  

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

    # r + dr arrays can be linear or log10 spaced
    _dr = np.zeros(ngrid)
    if dolog:
        _r = np.logspace(np.log10(rmin),np.log10(rmax),ngrid)
        _dr[1:ngrid-1] = _r[1:ngrid-1]*np.log(10)*(np.log10(_r)[2:ngrid] - np.log10(_r)[:ngrid-2])
        _dr[0]  = _r[0]*np.log(10)*(np.log10(_r)[1] - np.log10(_r)[0])
        _dr[ngrid-1] = _r[ngrid-1]*np.log(10)*(np.log10(_r)[ngrid-1] - np.log10(_r)[ngrid-2])
    else:
        _r = np.linspace(rmin,rmax,ngrid)
        _dr[1:ngrid-1] = _r[2:ngrid] - _r[:ngrid-2]
        _dr[0]  = _dr[1]
        _dr[ngrid-1] = _dr[ngrid-2]

    # orbital frequency is Keplerian
    omega = _r**(-3./2.)

    # change tilt to radians
    tilt *= np.pi/180.

    # density distribution can be "flat" or "gaussian"
    # note: probably most intelligent to normalize rho such that total disk mass
    # is constant regardless of distribution
    if rho_type == "flat":
        _density = np.ones(ngrid)
    elif rho_type == "gauss":
        _density = (1.0/(rw*np.sqrt(2.0*np.pi))) * np.exp(-0.5*((_r - r0)/rw)**2.0)
        _density /= np.average(_density)
    else:
        print "Error! rho_type needs to be set to \"gauss\" or \"flat\"! Exiting"
        exit()

    # build angular momentum quantities
    amom_mag     = _density * omega * _r * _r # angular momentum magnitude
    amom_unit    = np.array([np.sin(tilt),0.0,np.cos(tilt)]) # single amom unit vector
    amom_uvector = np.array([amom_unit]*ngrid) # amom unit vector extended to radial grid
    #prec_c = 180.0*np.pi/180.
    #rc = 30. 
    #amom_unit_inner   = np.array([np.sin(tilt),0.0,np.cos(tilt)]) # amom unit vector, r < rc
    #amom_unit_outer   = np.array([np.sin(tilt)*np.cos(prec_c),np.sin(tilt)*np.sin(prec_c),np.cos(tilt)]) # amom unit vector, r > rc
    #amom_uvector = np.ones((ngrid,3))       # build array for anuglar momentum unit vectors
    #amom_uvector[_r<=rc] *= amom_unit_inner # set array at r <= rc
    #amom_uvector[_r>rc]  *= amom_unit_outer # set array at r >  rc

    _amom_vector  = np.copy(amom_uvector) # building [Lx,Ly,Lz] for each radial grid element
    for j in range(3): _amom_vector[:,j] *= amom_mag

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
    print "dolog     = %s\n" % dolog
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

    ## Prepare initial conditions first

    # Initialize Cython stuff for iteration
    cdef double[:] r = _r
    cdef double[:] dr = _dr 
    cdef double[:] density = _density
    cdef double[:,:] omega_p = _omega_p
    cdef double[:] Lmag = np.sqrt(_amom_vector[:,0]**2.0 + _amom_vector[:,1]**2.0 + _amom_vector[:,2]**2.0)
    cdef double[:] Lx = _amom_vector[:,0]
    cdef double[:] Ly = _amom_vector[:,1]
    cdef double[:] Lz = _amom_vector[:,2]
    cdef double[:] lx = _amom_vector[:,0]/np.sqrt(_amom_vector[:,0]**2.0 + _amom_vector[:,1]**2.0 + _amom_vector[:,2]**2.0)
    cdef double[:] ly = _amom_vector[:,1]/np.sqrt(_amom_vector[:,0]**2.0 + _amom_vector[:,1]**2.0 + _amom_vector[:,2]**2.0)
    cdef double[:] lz = _amom_vector[:,2]/np.sqrt(_amom_vector[:,0]**2.0 + _amom_vector[:,1]**2.0 + _amom_vector[:,2]**2.0)
    cdef double[:] omega_p_z = np.copy(_omega_p[:,2])

    cdef double[:] psi   = np.zeros(ngrid)
    cdef double[:] Q1    = np.zeros(ngrid)
    cdef double[:] Q2    = np.zeros(ngrid)
    cdef double[:] Q3    = np.zeros(ngrid)
    cdef double[:] nu1   = np.zeros(ngrid)
    cdef double[:] nu2   = np.zeros(ngrid)
    cdef double[:] nu3   = np.zeros(ngrid)
    cdef double[:] dLxdt = np.zeros(ngrid)
    cdef double[:] dLydt = np.zeros(ngrid)
    cdef double[:] dLzdt = np.zeros(ngrid)
    cdef double[:] dSdt  = np.zeros(ngrid)

    cdef double psi_x,psi_y,psi_z
    cdef double f1_x,f1_y,f1_z,f2_x,f2_y,f2_z,f3_x,f3_y,f3_z,f4_x,f4_y,f4_z,f5_x,f5_y,f5_z
    cdef double g1,g2 # for mass equation
    cdef double small = 1e-30
    cdef int i, io_cnt

    # Initialize time and output counter
    t = 0.0
    io_cnt = 0

    cdef double[:] Q1_arr = np.log10(-Q1_parsed + small)
    cdef double[:] Q2_arr = np.log10(Q2_parsed + small)
    cdef double[:] Q3_arr = np.log10(Q3_parsed + small)
    cdef double[:] s_arr = _s_arr
    cdef FILE *f_out
    cdef char[40]  io_fn
    cdef double dt 

    # trying upstream diff
    cdef int upstream = 1
    cdef double v_adv
    cdef double Lmag_tmp

    ## initialize psi, nu
    for i in range(1,ngrid-1):
        # calculate warp parameter
        psi_x = (0.5*r[i]/dr[i])*(lx[i+1]-lx[i-1])
        psi_y = (0.5*r[i]/dr[i])*(ly[i+1]-ly[i-1])
        psi_z = (0.5*r[i]/dr[i])*(lz[i+1]-lz[i-1])
        psi[i] = (psi_x*psi_x + psi_y*psi_y + psi_z*psi_z)**0.5

        # calculate nu1,nu2,nu3
        nu1[i] = (-2.0/3.0)*(-1.0*10**(interp_1d(s_arr,Q1_arr,psi[i],ng_Q)))*((HoR**2.0)*r[i]**0.5)
        nu2[i] = 2.0*10**(interp_1d(s_arr,Q2_arr,psi[i],ng_Q))*((HoR**2.0)*r[i]**0.5)
        nu3[i] = 10**(interp_1d(s_arr,Q3_arr,psi[i],ng_Q))*((HoR**2.0)*r[i]**0.5)

    # iterate!
    while (t < tmax):

        ### cfl condition
        dt = 100000000
        for i in range(1,ngrid-1):
            dt = fmin(dt,cfl*0.5*dr[i]/((nu1[i]*3./dr[i]) + (nu2[i]*0.5/dr[i]) + (nu3[i]/dr[i]) + fabs(nu2[i]*psi[i]*psi[i]/r[i] - 1.5*nu1[i]/r[i])))

        #### Lets begin constructing the terms to evolve Lx, Ly, and Lz
        for i in range(1,ngrid-1):

            ## f1
            f1_x = (3.0/(4.0*r[i]))*(1.0/dr[i]**2.0)*((r[i+1]+r[i])*(lx[i+1]+lx[i])*(nu1[i+1]*Lmag[i+1]-nu1[i]*Lmag[i]) - (r[i]+r[i-1])*(lx[i]+lx[i-1])*(nu1[i]*Lmag[i]-nu1[i-1]*Lmag[i-1]))
            f1_y = (3.0/(4.0*r[i]))*(1.0/dr[i]**2.0)*((r[i+1]+r[i])*(ly[i+1]+ly[i])*(nu1[i+1]*Lmag[i+1]-nu1[i]*Lmag[i]) - (r[i]+r[i-1])*(ly[i]+ly[i-1])*(nu1[i]*Lmag[i]-nu1[i-1]*Lmag[i-1]))
            f1_z = (3.0/(4.0*r[i]))*(1.0/dr[i]**2.0)*((r[i+1]+r[i])*(lz[i+1]+lz[i])*(nu1[i+1]*Lmag[i+1]-nu1[i]*Lmag[i]) - (r[i]+r[i-1])*(lz[i]+lz[i-1])*(nu1[i]*Lmag[i]-nu1[i-1]*Lmag[i-1]))

            ## f2
            f2_x = (1.0/(16.0*r[i]))*(1.0/dr[i]**2.0)*((nu2[i+1]+nu2[i])*(r[i+1]+r[i])*(Lmag[i+1]+Lmag[i])*(lx[i+1]-lx[i]) - (nu2[i]+nu2[i-1])*(r[i]+r[i-1])*(Lmag[i]+Lmag[i-1])*(lx[i]-lx[i-1]))
            f2_y = (1.0/(16.0*r[i]))*(1.0/dr[i]**2.0)*((nu2[i+1]+nu2[i])*(r[i+1]+r[i])*(Lmag[i+1]+Lmag[i])*(ly[i+1]-ly[i]) - (nu2[i]+nu2[i-1])*(r[i]+r[i-1])*(Lmag[i]+Lmag[i-1])*(ly[i]-ly[i-1]))
            f2_z = (1.0/(16.0*r[i]))*(1.0/dr[i]**2.0)*((nu2[i+1]+nu2[i])*(r[i+1]+r[i])*(Lmag[i+1]+Lmag[i])*(lz[i+1]-lz[i]) - (nu2[i]+nu2[i-1])*(r[i]+r[i-1])*(Lmag[i]+Lmag[i-1])*(lz[i]-lz[i-1]))

            ## f3
            if (upstream):
                v_adv = nu2[i]*psi[i]*psi[i]/r[i] - 1.5*nu1[i]/r[i]
                if (v_adv < 0.):
                    f3_x = (1.0/r[i]/dr[i])*((nu2[i+1]*psi[i+1]**2. - 1.5*nu1[i+1])*Lx[i+1] - (nu2[i]*psi[i]**2. - 1.5*nu1[i])*Lx[i])
                    f3_y = (1.0/r[i]/dr[i])*((nu2[i+1]*psi[i+1]**2. - 1.5*nu1[i+1])*Ly[i+1] - (nu2[i]*psi[i]**2. - 1.5*nu1[i])*Ly[i])
                    f3_z = (1.0/r[i]/dr[i])*((nu2[i+1]*psi[i+1]**2. - 1.5*nu1[i+1])*Lz[i+1] - (nu2[i]*psi[i]**2. - 1.5*nu1[i])*Lz[i])
                    g2   = (1.0/r[i]/dr[i])*(density[i+1]*nu2[i+1]*psi[i+1]**2. - density[i]*nu2[i]*psi[i]**2.)
                elif (v_adv > 0.):
                    f3_x = (1.0/r[i]/dr[i])*((nu2[i]*psi[i]**2. - 1.5*nu1[i])*Lx[i] - (nu2[i-1]*psi[i-1]**2. - 1.5*nu1[i-1])*Lx[i-1])
                    f3_y = (1.0/r[i]/dr[i])*((nu2[i]*psi[i]**2. - 1.5*nu1[i])*Ly[i] - (nu2[i-1]*psi[i-1]**2. - 1.5*nu1[i-1])*Ly[i-1])
                    f3_z = (1.0/r[i]/dr[i])*((nu2[i]*psi[i]**2. - 1.5*nu1[i])*Lz[i] - (nu2[i-1]*psi[i-1]**2. - 1.5*nu1[i-1])*Lz[i-1])
                    g2   = (1.0/r[i]/dr[i])*(density[i]*nu2[i]*psi[i]**2. - density[i-1]*nu2[i-1]*psi[i-1]**2.)
                else:
                    f3_x = 0.
                    f3_y = 0.
                    f3_z = 0.
            else:
                f3_x = (1.0/(8.0*r[i]))*(1.0/dr[i]**3.0)*((0.5*(nu2[i+1]+nu2[i])*((r[i+1]+r[i])**2.0)*((lx[i+1]-lx[i])**2.0 + (ly[i+1]-ly[i])**2.0 + (lz[i+1]-lz[i])**2.0) - 3.0*(nu1[i+1]+nu1[i]))*(Lx[i+1] + Lx[i]) - (0.5*(nu2[i]+nu2[i-1])*((r[i]+r[i-1])**2.0)*((lx[i]-lx[i-1])**2.0 + (ly[i]-ly[i-1])**2.0 + (lz[i]-lz[i-1])**2.0) - 3.0*(nu1[i]+nu1[i-1]))*(Lx[i] + Lx[i-1]))
                f3_y = (1.0/(8.0*r[i]))*(1.0/dr[i]**3.0)*((0.5*(nu2[i+1]+nu2[i])*((r[i+1]+r[i])**2.0)*((lx[i+1]-lx[i])**2.0 + (ly[i+1]-ly[i])**2.0 + (lz[i+1]-lz[i])**2.0) - 3.0*(nu1[i+1]+nu1[i]))*(Ly[i+1] + Ly[i]) - (0.5*(nu2[i]+nu2[i-1])*((r[i]+r[i-1])**2.0)*((lx[i]-lx[i-1])**2.0 + (ly[i]-ly[i-1])**2.0 + (lz[i]-lz[i-1])**2.0) - 3.0*(nu1[i]+nu1[i-1]))*(Ly[i] + Ly[i-1]))
                f3_z = (1.0/(8.0*r[i]))*(1.0/dr[i]**3.0)*((0.5*(nu2[i+1]+nu2[i])*((r[i+1]+r[i])**2.0)*((lx[i+1]-lx[i])**2.0 + (ly[i+1]-ly[i])**2.0 + (lz[i+1]-lz[i])**2.0) - 3.0*(nu1[i+1]+nu1[i]))*(Lz[i+1] + Lz[i]) - (0.5*(nu2[i]+nu2[i-1])*((r[i]+r[i-1])**2.0)*((lx[i]-lx[i-1])**2.0 + (ly[i]-ly[i-1])**2.0 + (lz[i]-lz[i-1])**2.0) - 3.0*(nu1[i]+nu1[i-1]))*(Lz[i] + Lz[i-1]))

            ## f4
            f4_x = (1.0/(8.0*r[i]))*(1.0/dr[i]**2.0)*((nu3[i+1]+nu3[i])*(r[i+1]+r[i])*((Ly[i+1]+Ly[i])*(lz[i+1]-lz[i]) - (Lz[i+1]+Lz[i])*(ly[i+1]-ly[i])) - (nu3[i]+nu3[i-1])*(r[i]+r[i-1])*((Ly[i]+Ly[i-1])*(lz[i]-lz[i-1]) - (Lz[i]+Lz[i-1])*(ly[i]-ly[i-1])))
            f4_y = (1.0/(8.0*r[i]))*(1.0/dr[i]**2.0)*((nu3[i+1]+nu3[i])*(r[i+1]+r[i])*((Lz[i+1]+Lz[i])*(lx[i+1]-lx[i]) - (Lx[i+1]+Lx[i])*(lz[i+1]-lz[i])) - (nu3[i]+nu3[i-1])*(r[i]+r[i-1])*((Lz[i]+Lz[i-1])*(lx[i]-lx[i-1]) - (Lx[i]+Lx[i-1])*(lz[i]-lz[i-1])))
            f4_z = (1.0/(8.0*r[i]))*(1.0/dr[i]**2.0)*((nu3[i+1]+nu3[i])*(r[i+1]+r[i])*((Lx[i+1]+Lx[i])*(ly[i+1]-ly[i]) - (Ly[i+1]+Ly[i])*(lx[i+1]-lx[i])) - (nu3[i]+nu3[i-1])*(r[i]+r[i-1])*((Lx[i]+Lx[i-1])*(ly[i]-ly[i-1]) - (Ly[i]+Ly[i-1])*(lx[i]-lx[i-1])))

            ## f5
            f5_x = omega_p_z[i]*(-1.0)*Ly[i]
            f5_y = omega_p_z[i]*Lx[i]
            
            ## g1 
            g1 = (1.0/r[i]/dr[i])*(((r[i+1]**0.5 + r[i]**0.5)/(dr[i+1] + dr[i]))*(nu1[i+1]*density[i+1]*r[i+1]**0.5 - nu1[i]*density[i]*r[i]**0.5) - ((r[i]**0.5 + r[i-1]**0.5)/(dr[i] + dr[i-1]))*(nu1[i]*density[i]*r[i]**0.5 - nu1[i-1]*density[i-1]*r[i-1]**0.5))

            #### Fill derivative arrays
            dLxdt[i] = f1_x + f2_x + f3_x + f4_x + f5_x
            dLydt[i] = f1_y + f2_y + f3_y + f4_y + f5_y
            dLzdt[i] = f1_z + f2_z + f3_z + f4_z
            dSdt[i]  = g1 + g2

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
                fprintf(f_out, "%e ", nu1[i])
                fprintf(f_out, "%e ", nu2[i])
                fprintf(f_out, "%e ", nu3[i])
                fprintf(f_out, "%e ", density[i])
                fprintf(f_out, "\n")
            fclose(f_out)
            io_cnt += 1

        ## Apply updates!
        for i in range(1,ngrid-1):
            Lx[i] = Lx[i] + dt*dLxdt[i]
            Ly[i] = Ly[i] + dt*dLydt[i]
            Lz[i] = Lz[i] + dt*dLzdt[i]
            density[i] = density[i] + dt*dSdt[i]
            Lmag[i] = (Lx[i]**2.0 + Ly[i]**2.0 + Lz[i]**2.0)**0.5

            lx[i] = Lx[i]/Lmag[i]
            ly[i] = Ly[i]/Lmag[i]
            lz[i] = Lz[i]/Lmag[i]
        for i in range(1,ngrid-1):
            # calculate warp parameter
            psi_x = (0.5*r[i]/dr[i])*(lx[i+1]-lx[i-1])
            psi_y = (0.5*r[i]/dr[i])*(ly[i+1]-ly[i-1])
            psi_z = (0.5*r[i]/dr[i])*(lz[i+1]-lz[i-1])
            psi[i] = (psi_x*psi_x + psi_y*psi_y + psi_z*psi_z)**0.5
            # calculate nu1,nu2,nu3
            nu1[i] = (-2.0/3.0)*(-1.0*10**(interp_1d(s_arr,Q1_arr,psi[i],ng_Q)))*((HoR**2.0)*r[i]**0.5)
            nu2[i] = 2.0*10**(interp_1d(s_arr,Q2_arr,psi[i],ng_Q))*((HoR**2.0)*r[i]**0.5)
            nu3[i] = 10**(interp_1d(s_arr,Q3_arr,psi[i],ng_Q))*((HoR**2.0)*r[i]**0.5)


        # Apply BCs
        if   (bc_type==0): #### Apply sink boundary conditions
            Lx[0] = 1e-10 * Lx[1]
            Lx[ngrid-1] = 1e-10 * Lx[ngrid-2]
            Ly[0] = 1e-10 * Ly[1]
            Ly[ngrid-1] = 1e-10 * Ly[ngrid-2]
            Lz[0] = 1e-10 * Lz[1]
            Lz[ngrid-1] = 1e-10 * Lz[ngrid-2]
            density[0] = 1e-10 * density[1]
            density[ngrid-1] = 1e-10 * density[ngrid-2]
            psi[0] = 1e-10 * psi[1]
            psi[ngrid-1] = 1e-10 * psi[ngrid-2]
            nu1[0] = 1e-10 * nu1[1]
            nu1[ngrid-1] = 1e-10 * nu1[ngrid-2]
            nu2[0] = 1e-10 * nu2[1]
            nu2[ngrid-1] = 1e-10 * nu2[ngrid-2]
            nu3[0] = 1e-10 * nu3[1]
            nu3[ngrid-1] = 1e-10 * nu3[ngrid-2]
        elif (bc_type==1): #### Apply outflow boundary conditions
            Lx[0] = Lx[1];
            Lx[ngrid-1] = Lx[ngrid-2];
            Ly[0] = Ly[1];
            Ly[ngrid-1] = Ly[ngrid-2];
            Lz[0] = Lz[1];
            Lz[ngrid-1] = Lz[ngrid-2];
            density[0] = density[1];
            density[ngrid-1] = density[ngrid-2];
            psi[0] = psi[1]
            psi[ngrid-1] = psi[ngrid-2]
            nu1[0] = nu1[1]
            nu1[ngrid-1] = nu1[ngrid-2]
            nu2[0] = nu2[1]
            nu2[ngrid-1] = nu2[ngrid-2]
            nu3[0] = nu3[1]
            nu3[ngrid-1] = nu3[ngrid-2]

        #### Update timestep
        t += dt

