import numpy as np
from astropy.table import QTable
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy.interpolate import griddata

def build_table(filename):
    data_array = np.genfromtxt(filename)

    ## Read from columns of data arrays
    Lx = data_array[:,0]
    Ly = data_array[:,1]
    Lz = data_array[:,2]
    r  = data_array[:,3]
    Q1 = data_array[:,4]
    Q2 = data_array[:,5]
    Q3 = data_array[:,6]

    ## Get more quantities
    L = np.sqrt(Lx*Lx + Ly*Ly + Lz*Lz)
    lx = Lx/L
    ly = Ly/L
    lz = Lz/L
    tilt = np.arcsin(np.sqrt(lx*lx + ly*ly))*180.0/np.pi
    prec = np.arctan2(ly,lx)*180.0/np.pi

    ## Compile tables
    table_list   = [Lx,Ly,Lz,r,L,lx,ly,lz,tilt,prec]
    table_titles = ["Lx","Ly","Lz","r","L","lx","ly","lz","tilt","prec"]
    return QTable(data=table_list,names=table_titles)

def build_data(prefix,ngrid,tgrid,data_ngrid,order=3,convert=True,HoR=1e-3):
    '''
    ## Read from columns of data arrays
    Lx = data_array[:,0]
    Ly = data_array[:,1]
    Lz = data_array[:,2]
    r  = data_array[:,3]
    Q1 = data_array[:,4]
    Q2 = data_array[:,5]
    Q3 = data_array[:,6]
    t  = data_array[:,7]
    '''

    master_data = []
    for i in range(data_ngrid):
        fn = prefix + str(i) + ".csv"
        data_array = np.genfromtxt(fn)
        master_data.append(data_array)
    master_data = np.array(master_data)

    if convert:
        #tgrid = np.size(master_data[:,:,3])/ngrid
        rmin = np.min(master_data[:,:,3])
        rmax = np.max(master_data[:,:,3])
        tmin = 0.
        tmax = np.max(master_data[:,:,7])*rmin**(1.5)
        rspace = np.linspace(rmin,rmax,ngrid)
        tspace = np.linspace(tmin,tmax,tgrid)
        print "shape of tspace: ", np.shape(tspace)
        data_tau = np.copy(master_data[0,:,7])
        data_r   = np.copy(master_data[:,0,3])
        R, T = np.meshgrid(rspace,tspace,indexing='ij')


        new_tau = T / R**(1.5)
        print master_data[:,0,3], len(master_data[:,0,3])
        print master_data[0,:,7], master_data[:,0,7]

        new_it = griddata(data_tau, np.arange(np.size(data_tau)), new_tau.ravel(),method='linear')
        new_ir = griddata(data_r, np.arange(np.size(data_r)), R.ravel(),method='linear')

        Lx = map_coordinates(master_data[:,:,0], np.array([new_ir, new_it]),
                                    order=order).reshape(new_tau.shape)
        Ly = map_coordinates(master_data[:,:,1], np.array([new_ir, new_it]),
                                    order=order).reshape(new_tau.shape)
        Lz = map_coordinates(master_data[:,:,2], np.array([new_ir, new_it]),
                                    order=order).reshape(new_tau.shape)
        r  = map_coordinates(master_data[:,:,3], np.array([new_ir, new_it]),
                                    order=order).reshape(new_tau.shape)
        Q1 = map_coordinates(master_data[:,:,4], np.array([new_ir, new_it]),
                                    order=order).reshape(new_tau.shape)
        Q2 = map_coordinates(master_data[:,:,5], np.array([new_ir, new_it]),
                                    order=order).reshape(new_tau.shape)
        Q3 = map_coordinates(master_data[:,:,6], np.array([new_ir, new_it]),
                                    order=order).reshape(new_tau.shape)
        t  = map_coordinates(master_data[:,:,7], np.array([new_ir, new_it]),
                                    order=order).reshape(new_tau.shape)*r**1.5



    else: # Do code units
        Lx = master_data[:,:,0]
        Ly = master_data[:,:,1]
        Lz = master_data[:,:,2]
        r  = master_data[:,:,3]
        Q1 = master_data[:,:,4]
        Q2 = master_data[:,:,5]
        Q3 = master_data[:,:,6]
        t  = master_data[:,:,7] 

        # just so the rest of the code will work
        R = np.copy(r)
        T = np.copy(t)

    # extra quantities that might be useful
    L = (Lx*Lx + Ly*Ly + Lz*Lz)**0.5
    lx = Lx/L
    ly = Ly/L
    lz = Lz/L

    # tilt angle
    tilt = np.arcsin(np.sqrt(lx*lx + ly*ly))*180.0/np.pi

    # precession angle
    prec = np.arctan2(ly,lx)*180.0/np.pi

    # surface density
    sigma = L/r/HoR**2.

    # dt
    dt = np.zeros(np.shape(r))
    dt[:,1:] = t[:,1:] - t[:,:-1]
    dt[:,0] = dt[:,1]

    # dr
    dr = np.zeros(np.shape(r))
    dr[1:,:] = r[1:,:] - r[:-1,:]
    dr[0,:] = r[1,:]

    # use dsigma_dt o calculate mdot and infer vr
    dsigma_dt = np.zeros(np.shape(r))
    dsigma_dt[:,1:] = (sigma[:,1:] - sigma[:,:-1])/dt[:,1:]
    dsigma_dt[:,0] = dsigma_dt[:,0]
    mdot = 2.*np.pi*r*dsigma_dt*dr
    vr = -1. * dsigma_dt * dr / sigma

    # precession rate
    dpdt = np.zeros(np.shape(r))
    dpdt[:,1:] = (prec[:,1:] - prec[:,:-1])
    dpdt[:,0] = dpdt[:,1]
    dpdt = dpdt/dt

    # warp amplitude
    psi = np.zeros(np.shape(r))
    dlx = lx[1:,:] - lx[:-1,:]
    dly = ly[1:,:] - ly[:-1,:]
    dlz = lz[1:,:] - lz[:-1,:]
    psi[1:,:] = (dlx*dlx + dly*dly + dlz*dlz)**(0.5)
    psi[0,:]  = psi[1,:]
    psi = r*psi/dr


    ## Calculation of Dogan instability criteria! See Dogan et al 2018 for details.
    # for Keplerian disk, the parameter a = 2.
    a = 2.
    # dQ1dr
    dQ1dr = np.zeros(np.shape(r))
    dQ1dr[1:,:] = Q1[1:,:] - Q1[:-1,:]
    dQ1dr[0,:] = dQ1dr[1,:]
    dQ1dr = dQ1dr/dr
    # dQ2dr
    dQ2dr = np.zeros(np.shape(r))
    dQ2dr[1:,:] = Q2[1:,:] - Q2[:-1,:]
    dQ2dr[0,:] = dQ2dr[1,:]
    dQ2dr = dQ2dr/dr
    # dQ1dpsi
    dQ1dpsi = np.zeros(np.shape(r))
    dQ1dpsi[1:,:] = Q1[1:,:] - Q1[:-1,:]
    dQ1dpsi[0,:] = dQ1dpsi[1,:]
    dQ1dpsi = dQ1dpsi/psi
    # dQ1dpsi
    dQ2dpsi = np.zeros(np.shape(r))
    dQ2dpsi[1:,:] = Q2[1:,:] - Q2[:-1,:]
    dQ2dpsi[0,:] = dQ2dpsi[1,:]
    dQ2dpsi = dQ2dpsi/psi
    # dogan1 is Equation (40) of Dogan 2018
    dogan1 = a*dQ1dpsi - dQ2dpsi 
    # dogan2 is the second: expression in Equation (41) of Dogan 2018
    dogan2 = 4*a*(Q1*Q2 + (Q1*dQ2dr - dQ1dr*Q2)*psi)
    # The disk is locally unstable if: dogan1 > 0 or if dogan1 < 0 and dogan2 > 0
    ##


    ## Compile tables
    table_list   = [r,t,Lx,Ly,Lz,Q1,Q2,Q3,L,lx,ly,lz,tilt,prec,sigma,R,T,dt,dr,mdot,vr,dpdt,psi,dogan1,dogan2]
    table_titles = ["r","t","Lx","Ly","Lz","Q1","Q2","Q3","L","lx","ly","lz","tilt","prec","sigma","R","T","dt","dr","mdot","vr","dpdt","psi","dogan1","dogan2"]
    return QTable(data=table_list,names=table_titles)

def get_fn_list(path_to_outputs,first,last):
    fn_prefix = path_to_outputs
    fn_suffix = ".csv"
    fileno = np.linspace(first,last,last-first+1).astype(int).astype(str)
    fns = []
    for fno in fileno:
        fns.append(fn_prefix + fno + fn_suffix)
    return np.array(fns)
