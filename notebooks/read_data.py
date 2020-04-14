import numpy as np
from astropy.table import QTable
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

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

def build_data(prefix, ngrid,order=3,convert=True):
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
    for i in range(ngrid):
        fn = prefix + str(i) + ".csv"
        data_array = np.genfromtxt(fn)
        master_data.append(data_array)
    master_data = np.array(master_data)

    if convert:
        tgrid = np.size(master_data[:,:,3])/ngrid
        rmin = np.min(master_data[:,:,3])
        rmax = np.max(master_data[:,:,3])
        tmin = 0.
        tmax = np.max(master_data[:,:,7])*rmax**(1.5)
        rspace = np.linspace(rmin,rmax,ngrid)
        tspace = np.linspace(tmin,tmax,tgrid)
        tau = master_data[0,:,7]

        R, T = np.meshgrid(rspace,tspace,indexing='ij')


        new_tau = T / R**(1.5)

        it = interp1d(tau, np.arange(len(tau)), bounds_error=False)
        ir = interp1d(rspace, np.arange(ngrid), bounds_error=False)

        new_it = it(new_tau.ravel())
        new_ir = ir(R.ravel())

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
                                    order=order).reshape(new_tau.shape)


    else: # Do code units
        Lx = master_data[:,:,0]
        Ly = master_data[:,:,1]
        Lz = master_data[:,:,2]
        r  = master_data[:,:,3]
        Q1 = master_data[:,:,4]
        Q2 = master_data[:,:,5]
        Q3 = master_data[:,:,6]
        t  = master_data[:,:,7] 

    L = (Lx*Lx + Ly*Ly + Lz*Lz)**0.5
    lx = Lx/L
    ly = Ly/L
    lz = Lz/L
    tilt = np.arcsin(np.sqrt(lx*lx + ly*ly))*180.0/np.pi
    prec = np.arctan2(ly,lx)*180.0/np.pi


    ## Compile tables
    table_list   = [r,t,Lx,Ly,Lz,Q1,Q2,Q3,L,lx,ly,lz,tilt,prec]
    table_titles = ["r","t","Lx","Ly","Lz","Q1","Q2","Q3","L","lx","ly","lz","tilt","prec"]
    return QTable(data=table_list,names=table_titles)

def get_fn_list(path_to_outputs,first,last):
    fn_prefix = path_to_outputs
    fn_suffix = ".csv"
    fileno = np.linspace(first,last,last-first+1).astype(int).astype(str)
    fns = []
    for fno in fileno:
        fns.append(fn_prefix + fno + fn_suffix)
    return np.array(fns)
