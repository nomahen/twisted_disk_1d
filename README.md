Evolving the partial differential equations governing the nonlinear evolution of twisted accretion disks, as provided in Ogilvie 1999, see also Nixon+King 2012.

Brief tutorial:

1st: it's necessary to build tables of Q1,Q2,Q3 (see Ogilvie 1999), which dictate viscosity in the nonlinear (large warp) regime. In the folder "nonlinear coefficients" is the file "Q_tables_1d_serial.py" which does this. It takes several arguments, but the default options are the same as are used in Nixon+King 2012. To build:
"
cd ./nonlinear_coefficients
python Q_tables_1d_serial -py
"
This will build Q_tables_1d_serial in the ./tables directory.

2nd: now we can run the script to evolve the PDEs, "evolve.py". There is also "evolve_interactive.ipynb" in ./notebooks, which can be convenient if you want to test things, and also has the discretization scheme described in markdown cells. "evolve.py" reads parameters from "params.py", which you can modify; however, the default values are the same as are used in Nixon+King 2012. 

If you run:
"
python ./evolve.py
"
It will create a series of outputs "evolve_1.csv","evolve_2.csv",... in ./outputs, which you can then visualize using the jupyter notebook in ./notebooks, "plot_outputs.ipynb", and make a movie of the tilt evolution following the steps there.  
