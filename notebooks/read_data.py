import numpy as np
from astropy.table import QTable

def build_table(filename):
    data_array = np.genfromtxt(filename)

    ## Read from columns of data arrays
    Lx = data_array[:,0]
    Ly = data_array[:,1]
    Lz = data_array[:,2]
    r  = data_array[:,3]

    ## Get more quantities
    L = np.sqrt(Lx*Lx + Ly*Ly + Lz*Lz)
    lx = Lx/L
    ly = Ly/L
    lz = Lz/L
    rho = L*r**(-1./2.)
    tilt = np.arcsin(np.sqrt(lx*lx + ly*ly))*180.0/np.pi
    prec = np.arctan2(ly,lx)*180.0/np.pi

    ## Derivative quantitiess need to know how grid is spaced (linear or log)
    if ((r[1] - r[0]) - (r[2] - r[1]) < 1e-14):
        dr = r[2:] - r[:-2]
        print "Assuming linearly spaced grid..."
    elif (((np.log10(r)[1] - np.log10(r)[0]) - (np.log10(r)[2] - np.log10(r)[1]))  < 1e-14):
        dr = r[1:-1]*np.log(10)*(np.log10(r)[2:] - np.log10(r)[:-2])
        print "Assuming log10 spaced grid..."
    else:
        print "Error: Something is wrong with the log check! Exiting!"
        return

    ## Need to fill guard cells for derivative quantities
    # Build warp amplitude Psi
    psi = np.zeros(len(L))
    psi_x = (0.5*r[1:-1]/dr)*(lx[2:]-lx[:-2])
    psi_y = (0.5*r[1:-1]/dr)*(ly[2:]-ly[:-2])
    psi_z = (0.5*r[1:-1]/dr)*(lz[2:]-lz[:-2])
    psi[1:-1] = np.sqrt(psi_x**2.0 + psi_y**2.0 + psi_z**2.0)
    psi[0] = psi[1]
    psi[-1] = psi[-2]

    ## Compile tables
    table_list   = [Lx,Ly,Lz,r,L,lx,ly,lz,rho,tilt,prec,psi]
    table_titles = ["Lx","Ly","Lz","r","L","lx","ly","lz","rho","tilt","prec","psi"]
    return QTable(data=table_list,names=table_titles)

def get_fn_list(path_to_outputs,first,last):
    fn_prefix = path_to_outputs + "evolve_"
    fn_suffix = ".csv"
    fileno = np.linspace(first,last,last-first+1).astype(int).astype(str)
    fns = []
    for fno in fileno:
        fns.append(fn_prefix + fno + fn_suffix)
    return np.array(fns)
