import numpy as np
import matplotlib.pyplot as plt
from read_data import *
import argparse
import glob
from scipy.interpolate import interp1d
from scipy.integrate import quad

p = argparse.ArgumentParser()
p.add_argument('-g',"--grid_list",dest="grid_list",type=str,default='50,100,200', help="Comma separated string of grids.")
p.add_argument('-p',"--path",dest="path",type=str,default="./outputs", help="Path where ./n50, ./n100, etc... are located")
p.add_argument('-f',"--file",dest="file",type=str,default="/output_", help="Prefix of each .csv file.")
p.add_argument('-H',"--hor",dest="hor",type=float,default=0.001,help="Disk aspect ratio H/R")

args = p.parse_args()
grid_list  = args.grid_list
path       = args.path
fileprefix = args.file
HoR        = args.hor

## Parse command line into array of grid sizes and simulation labels for each grid size
data_ngrid = np.array(grid_list.split(',')).astype(int)
prefixes = []
for n in data_ngrid:
    prefixes.append("n" + str(n))

## Load data into data dictionary for each simulation in "prefixes"
data_dict = {}
for pref in prefixes:
    path_to_data = path + "/" + pref + fileprefix
    print "Unpacking simulation %s...\n" % path_to_data

    ## This block of code is to get the total number of output files in the target directory
    tmp_fileno_list = []
    for f in glob.glob(path_to_data+"*"): tmp_fileno_list.append(f.replace(path_to_data,"").replace(".csv",""))
    tmp_fileno_list = np.array(tmp_fileno_list).astype(int)
    if np.size(tmp_fileno_list)==0: 
        print "Error: No outputs detected with prefix \"%s\"! Exiting!" % path_to_data
        exit()
    first = np.min(tmp_fileno_list)
    last  = np.max(tmp_fileno_list)
    ##

    filenames = get_fn_list(path_to_data,first,last)
    tmp_data = []
    for i,fn in enumerate(filenames):
        print "Loading file %d..." % i
        tmp_data.append(build_table(fn,HoR))
    data_dict[pref] = tmp_data
    
# get final snapshot
time_index = -1

## Code to visualize data
fig,ax = plt.subplots(1,2,figsize=(10,4))

for i,f in enumerate(prefixes):
    plot_r  = data_dict[f][time_index]["r"]
    plot_Lz = data_dict[f][time_index]["Lz"]

    time   = data_dict[f][time_index]["t"][0]
    rmin = np.min(plot_r)
    rmax = np.max(plot_r)
    print "simulaton %s at time %e" % (f,time)

    ax[0].plot(plot_r,plot_Lz,label=f,ls='--')
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$\Lambda_z$')
    ax[0].set_xlim(rmin,rmax)

def map_cell_volume(x,field,xmin,xmax):
    dx_base  = xmax - xmin
    dx_high  = np.abs(x[1]-x[0])
    lower_index = np.argmin(np.abs(x-xmin))
    upper_index = np.argmin(np.abs(x-xmax))
    num_cells = upper_index - lower_index + 1
    if (num_cells==1):
        cell_volume = field[lower_index]*dx_high
    if (num_cells>=2):
        x_lower_right = x[lower_index] + dx_high/2.
        W_lower = (x_lower_right - xmin)/dx_base
        x_upper_left  = x[upper_index] - dx_high/2.
        W_upper = (dx_high - (xmax - x_upper_left))/dx_base
        cell_volume = field[lower_index]*W_lower*dx_base + field[upper_index]*W_upper*dx_base
        for i in range(lower_index+1,upper_index): # does nothing if num_cells == 2
            cell_volume += field[i]*dx_high
    return cell_volume


xs = [] # For each simulation, this is the log(r) array
fs = [] # For each simulation, this is the field array
ns = [] # For each simulation, this is the number of cells
dxs = [] # For each simulation, this is the cell width
for i,f in enumerate(prefixes):
    print "\nUnpacking simulation %s at time %e r_g/c\n" % (f,data_dict[f][time_index]["t"][0])
    xs.append(data_dict[f][time_index]["r"]) # Take log to convert to internal coordinates
    fs.append(data_dict[f][time_index]["Lz"])        # We can test convergence for Lx, Ly or Lz
    ns.append(len(data_dict[f][time_index]["Lx"]))
    dx = np.average(xs[i][1:]-xs[i][:-1])
    dxs.append(dx)
xs = np.array(xs)
fs = np.array(fs)
ns = np.array(ns)
dxs = np.array(dxs)

ref_x  = np.copy(xs[0])
ref_dx = np.average(ref_x[1:]-ref_x[:-1])
ref_f  = np.copy(fs[0]) 

# Lets integrate the total code volume within each cell of the lowest resolution grid
num_pts_ref = len(ref_x)
sols = []
for run in range(len(prefixes)):
    sol = np.zeros(num_pts_ref-4) # dont include guard cells (4 = 2 on each side)
    # function for piecewise function data
    data_pc = interp1d(xs[run],fs[run], kind="nearest")
    for i in range(2,num_pts_ref-2):
        cell_xmin = ref_x[i] - ref_dx/2.
        cell_xmax = ref_x[i] + ref_dx/2.
        #new_cell_volume = map_cell_volume(xs[run],fs[run],cell_xmin,cell_xmax)
        sol[i-2] = quad(data_pc,cell_xmin,cell_xmax)[0]#new_cell_volume
    sols.append(sol)


# S will contain our errors
S = np.zeros(len(prefixes))
for run in range(len(prefixes[:-1])):
    norm = 0.
    for i in range(num_pts_ref-4):
        S[run] += np.abs(sols[run][i] - sols[-1][i])

print "\n## Calculated error ##"
print "log S        = ", np.log(S)
print "S            = ", S
print "sim names    = ", prefixes
print "number cells = ", ns
print "####\n"

## Lets start making our fit

# We fit in logspace to get a power law
fit_x = np.log(ns[:-1])
fit_y = np.log(S[:-1])
# This assumes a relation "fit_y = m*f_x"; i.e. y-intercept is zero. 
#m = np.linalg.lstsq(fit_x.reshape(-1,1), fit_y)[0][0]
p = np.polyfit(fit_x,fit_y,deg=1)
m = p[0]
b = p[1]

print "\nOur fit is %e-th order accurate with error constant %e\n\n" % (m, np.exp(b))

## Lets plot

# Lambda function for our fit so we can plot
fit = lambda x: m*x + b

plot_x = np.log(np.linspace(0,10000,10000))
plot_y = fit(plot_x)

# Switch from logspace to log2 space so simulations will be evenly spaced
plot_x = np.log2(np.exp(plot_x))
fit_x  = np.log2(np.exp(fit_x))

# Now plot!
ax[1].plot(plot_x,plot_y,color='peru')
ax[1].scatter(fit_x,fit_y,color='black',s=20,zorder=3)
ax[1].set_xlim(fit_x[0] - 1, fit_x[-1] + 1)
ax[1].set_xlabel(r'${\rm log}_2n_{\rm grid}$')
ax[1].set_ylabel(r'${\rm log}_2S$')

fig.suptitle(r"$L_1 = %5.4fn_{\rm grid}^{%3.2f}$" % (np.exp(b),m))

# Save figure
if 1:
    plt.savefig(path + "/convergence_autofit_integrated.pdf")


##

