import numpy as np
from astropy.table import QTable

def build_table(filename):
    data_array = np.genfromtxt(filename)

    ## Read from columns of data arrays
    Lx = data_array[:,0]
    Ly = data_array[:,1]
    Lz = data_array[:,2]
    r  = data_array[:,3]
    nu1 = data_array[:,4]
    nu2 = data_array[:,5]
    nu3 = data_array[:,6]

    ## Get more quantities
    L = np.sqrt(Lx*Lx + Ly*Ly + Lz*Lz)
    lx = Lx/L
    ly = Ly/L
    lz = Lz/L
    rho = L*r**(-1./2.)
    tilt = np.arcsin(np.sqrt(lx*lx + ly*ly))*180.0/np.pi
    prec = np.arctan2(ly,lx)*180.0/np.pi

    ## Derivative quantitiess need to know how grid is spaced (linear or log)
    dr_chk_lin = np.abs((r[1] - r[0]) - (r[2] - r[1]))/r[1]
    dr_chk_log = np.abs((np.log10(r)[1] - np.log10(r)[0]) - (np.log10(r)[2] - np.log10(r)[1]))/np.log10(r)[1]
    
    dr = np.zeros(len(L))
    if (dr_chk_lin < dr_chk_log):
        dr[1:-1] = r[2:] - r[:-2]
        dr[0]  = r[0]*np.log(10)*(np.log10(r)[1] - np.log10(r)[0])
        dr[-1] = r[-1]*np.log(10)*(np.log10(r)[-1] - np.log10(r)[-2])
        print "Assuming linearly spaced grid..."
    else:
        dr[1:-1] = r[1:-1]*np.log(10)*(np.log10(r)[2:] - np.log10(r)[:-2])
        dr[0]  = dr[1]
        dr[-1] = dr[-2]
        print "Assuming log10 spaced grid..."

    ## Need to fill guard cells for derivative quantities
    # Build warp amplitude Psi
    psi = np.zeros(len(L))
    psi_x = (0.5*r[1:-1]/dr[1:-1])*(lx[2:]-lx[:-2])
    psi_y = (0.5*r[1:-1]/dr[1:-1])*(ly[2:]-ly[:-2])
    psi_z = (0.5*r[1:-1]/dr[1:-1])*(lz[2:]-lz[:-2])
    psi[1:-1] = np.sqrt(psi_x**2.0 + psi_y**2.0 + psi_z**2.0)
    psi[0] = psi[1]
    psi[-1] = psi[-2]
    
    vadv = 1.5*nu1/dr - nu2*psi/r
    mdot = rho*2*np.pi*r*vadv

    ## Compile tables
    table_list   = [Lx,Ly,Lz,r,L,lx,ly,lz,rho,tilt,prec,psi,mdot,dr]
    table_titles = ["Lx","Ly","Lz","r","L","lx","ly","lz","rho","tilt","prec","psi","mdot","dr"]
    return QTable(data=table_list,names=table_titles)

def get_fn_list(path_to_outputs,first,last):
    fn_prefix = path_to_outputs
    fn_suffix = ".csv"
    fileno = np.linspace(first,last,last-first+1).astype(int).astype(str)
    fns = []
    for fno in fileno:
        fns.append(fn_prefix + fno + fn_suffix)
    return np.array(fns)
