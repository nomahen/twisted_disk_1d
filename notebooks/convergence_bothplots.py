import numpy as np
import matplotlib.pyplot as plt
from read_data import *
import argparse
import glob
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
    

## Code to visualize data
def plot_interface_multi(table,prefixes,time_ind):
    fig, ax = plt.subplots(3,2,figsize=(20,24))

    for i,f in enumerate(prefixes):
        plot_r  = table[f][time_ind]["r"]
        plot_d  = table[f][time_ind]["sigma"]
        plot_Lx = table[f][time_ind]["Lx"]
        plot_Ly = table[f][time_ind]["Ly"]
        plot_Lz = table[f][time_ind]["Lz"]
        plot_t  = table[f][time_ind]["tilt"]
        plot_p  = table[f][time_ind]["prec"]

        time   = table[f][time_ind]["t"][0]
        rmin = np.min(plot_r)
        rmax = np.max(plot_r)
        print "simulaton %s at time %e r_g/c" % (f,time)

        ax[0][0].plot(plot_r,plot_t,label=f)
        ax[0][0].set_xlabel(r'$r\,[r_{\rm g}]$')
        ax[0][0].set_ylabel(r'$T\,[{\rm deg}]$')
        #ax[0][0].set_ylim(0,2)
        ax[0][0].set_xlim(rmin,rmax)
        ax[0][0].set_xscale('log')
        ax[0][0].legend(frameon=False,ncol=len(prefixes))#,fontsize='x-small')

        ax[1][0].plot(plot_r,plot_p,label=f)
        ax[1][0].set_xlabel(r'$r\,[r_{\rm g}]$')
        ax[1][0].set_ylabel(r'$P\,[{\rm deg}]$')
        #ax[1][0].set_ylim(-1e-4,1e-4)
        ax[1][0].set_xlim(rmin,rmax)
        ax[1][0].set_xscale('log')
        #ax[1][0].legend(frameon=False)

        ax[2][0].plot(plot_r,plot_d,label=f)
        ax[2][0].set_xlabel(r'$r\,[r_{\rm g}]$')
        ax[2][0].set_ylabel(r'$\Sigma$')
        #ax[2][0].set_ylim(-1e-4,1e-4)
        ax[2][0].set_xlim(rmin,rmax)
        ax[2][0].set_xscale('log')
        ax[2][0].set_yscale('log')
        #ax[2][0].legend(frameon=False)

        ax[0][1].plot(plot_r,plot_Lx,label=f)
        ax[0][1].set_xlabel(r'$r\,[r_{\rm g}]$')
        ax[0][1].set_ylabel(r'$\Lambda_x$')
        #ax[0][1].set_ylim(0,50000)
        ax[0][1].set_xlim(rmin,rmax)
        ax[0][1].set_xscale('log')
        #ax[0][1].legend(frameon=False)
        #ax[0][1].set_yscale('log')

        ax[1][1].plot(plot_r,plot_Ly,label=f)
        ax[1][1].set_xlabel(r'$r\,[r_{\rm g}]$')
        ax[1][1].set_ylabel(r'$\Lambda_y$')
        #ax[1][1].set_ylim(-50,1)
        ax[1][1].set_xlim(rmin,rmax)
        ax[1][1].set_xscale('log')
        #ax[1][1].legend(frameon=False)

        ax[2][1].plot(plot_r,plot_Lz,label=f)
        ax[2][1].set_xlabel(r'$r\,[r_{\rm g}]$')
        ax[2][1].set_ylabel(r'$\Lambda_z$')
        #ax[2][1].set_ylim(-1e-3,1e-3)
        ax[2][1].set_xlim(rmin,rmax)
        ax[2][1].set_xscale('log')
        #ax[2][1].legend(frameon=False)

        
    #plt.tight_layout(pad=10.0,w_pad=8.0,h_pad=20.0)
    return fig
fig = plot_interface_multi(data_dict,prefixes,-1)

if 1:
    plt.savefig(path+'/convergence_autofig.pdf')
plt.clf()
##

## Code to build convergence plots
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

time_index = -1

xs = [] # For each simulation, this is the log(r) array
fs = [] # For each simulation, this is the field array
ns = [] # For each simulation, this is the number of cells
dxs = [] # For each simulation, this is the cell width
for i,f in enumerate(prefixes):
    print "\nUnpacking simulation %s at time %e r_g/c\n" % (f,data_dict[f][time_index]["t"][0])
    xs.append(np.log(data_dict[f][time_index]["r"])) # Take log to convert to internal coordinates
    fs.append(data_dict[f][time_index]["Lz"])        # We can test convergence for Lx, Ly or Lz
    ns.append(len(data_dict[f][time_index]["Lx"]))
    dx = np.average(xs[i][1:]-xs[i][:-1])
    #print "dx = ", dx, "example dx = ", xs[i][1]-xs[i][0], " (should be equal!) "
    dxs.append(dx)
xs = np.array(xs)
fs = np.array(fs)
ns = np.array(ns)
dxs = np.array(dxs)

sol_x = np.copy(xs[-1])
sol_dx = np.average(sol_x[1:]-sol_x[:-1])
sol_f = np.copy(fs[-1])

# S will contain our errors
S = np.zeros(len(prefixes))
for run in range(len(prefixes[:-1])):
    norm = 0.
    for i in range(2,ns[run]-2):
        sol_cell_index = np.argmin(np.abs(sol_x-xs[run][i]))
        #print "x at cell = ", xs[run][i], "x at master = ", sol_x[sol_cell_index]
        norm += dxs[run]*np.abs(fs[run][i]-sol_f[sol_cell_index])
    S[run] = np.abs(np.copy(norm))

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
print 2**fit_x, fit_x

# Now plot!
plt.plot(plot_x,plot_y,color='peru')
plt.scatter(fit_x,fit_y,color='black',s=20,zorder=3)
plt.xlim(fit_x[0] - 1, fit_x[-1] + 1)
plt.xlabel(r'${\rm log}_2n_{\rm grid}$')
plt.ylabel(r'${\rm log}_2S$')
plt.title(r"$L_1 = %5.4fn_{\rm grid}^{%3.2f}$" % (np.exp(b),m))
#plt.tight_layout()

# Save figure
if 1:
    plt.savefig(path + "/convergence_autofit.pdf")


##

