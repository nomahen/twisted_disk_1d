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
    if pref == prefixes[-1]: time_length = np.size(tmp_fileno_list) # number of outputs for reference simulation
    ##

    filenames = get_fn_list(path_to_data,first,last)
    tmp_data = []
    for i,fn in enumerate(filenames):
        print "Loading file %d..." % i
        tmp_data.append(build_table(fn,HoR))
    data_dict[pref] = tmp_data

reference_time_array = []
for i in range(time_length):
    reference_time_array.append(data_dict[prefixes[-1]][i]["t"][0])
reference_time_array = np.array(reference_time_array)

time_index = -1

xs = [] # For each simulation, this is the log(r) array
fs = [] # For each simulation, this is the field array
ns = [] # For each simulation, this is the number of cells
ts = [] # For each simulation, this is the time at which nmax steps was reached
dxs = [] # For each simulation, this is the cell width
for i,f in enumerate(prefixes):
    print "\nUnpacking simulation %s at time %e r_g/c\n" % (f,data_dict[f][time_index]["t"][0])
    ts.append(data_dict[f][time_index]["t"][0])
    xs.append(np.log(data_dict[f][time_index]["r"])) # Take log to convert to internal coordinates
    fs.append(data_dict[f][time_index]["Lz"])        # We can test convergence for Lx, Ly or Lz
    ns.append(len(data_dict[f][time_index]["Lx"]))
    dx = np.average(xs[i][1:]-xs[i][:-1])
    #print "dx = ", dx, "example dx = ", xs[i][1]-xs[i][0], " (should be equal!) "
    dxs.append(dx)
ts = np.array(ts)
xs = np.array(xs)
fs = np.array(fs)
ns = np.array(ns)
dxs = np.array(dxs)

sol_x = np.copy(xs[-1])
sol_dx = np.average(sol_x[1:]-sol_x[:-1])
#sol_f = np.copy(fs[-1])

# S will contain our errors
S = np.zeros(len(prefixes))
for run in range(len(prefixes[:-1])):
    current_time_index = np.argmin(np.abs(ts[run]-reference_time_array))
    sol_f = np.copy(data_dict[prefixes[-1]][current_time_index]["Lz"])
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

