import numpy as np
from astropy.table import QTable

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

def get_fn_list(path_to_outputs,first,last):
    fn_prefix = path_to_outputs
    fn_suffix = ".csv"
    fileno = np.linspace(first,last,last-first+1).astype(int).astype(str)
    fns = []
    for fno in fileno:
        fns.append(fn_prefix + fno + fn_suffix)
    return np.array(fns)
