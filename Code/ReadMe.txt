This folder contains all the codes used to produce the data in Figures_and_Data folder



-The code is run by (for example) :
$ python Meta_curves.py

-Which communicates with (for example) script_det_numba.py
-With all parameters specified within Meta_curves
-Data directory, where all data will be saved to, is specified in Meta_curves.py
-Parameters used for each figure are given in the ParametersListed.txt file

#----------- CODE VERSION 1 -------------------#

The main (and simplest) version of the code are this pair:
Meta_curves.py
script_det_numba.py

The following figures use data made with this pair:
Fig_Arrows
Fig_Tests (Lattice & Roff version)
Fig_Indentation (Every line apart from "no type 1")
Fig_DoSPlot
Fig_Pitch (This was slightly older code and all parameters + data processing scripts are in the folder /PythoCode/Fig_Pitch/

For Fig_DoSPlot and Tests, where it is necessary to output more than just the probability over time, & parameter sweeps are involved (one can find an example of a parameter sweep at the bottom of meta_curves.py) another code is used to condense the data anysis. These are Data_Proc.py and Meta_Data.py and output characteristics of probability curves which can then be used to calulate 'DoS. This is run after you already have run Meta_curves.py and 


#----------- CODE VERSION 2 -------------------#

Another pair explores the effect of Ron Vs Roff energy dependency in rates:

Meta_curves_ron.py
script_det_numba_ron.py

Which creates data for the figure:
Fig_Tests (Ron version)

#----------- CODE VERSION 3 -------------------#

The set of codes for creating the energy barrier plots ****

The energy has to be sampled from adding monomers of a certain type to the lattice. This is done in /Code/PythonCode/Fig_EnergyBarrier/Metrop2Bindings3subs.py We choose a number of the 3rd subunit (subunit 0) to be on the lattice already (some average of the total subunits in the normal run) then we sample the lattice as we add subunits (1 & 2). The output is the total energy 


#----------- CODE VERSION 4 -------------------#

Another pair is for exploring role of VPS4:

This has a different set of arguments so that an energy packet is given to subunits which are more frustrated. The extra arguments are 'Threshold' and 'energy packet', what is the threshold above which they will be given the packet and how much energy are they given. 

#----------- CODE VERSION 5 -------------------#




Another pair is for when there is no type 1:

### where?

Another pair is for exploring role of Torsion:

Another pair is for exploring role of Bindings:








