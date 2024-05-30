This folder contains all the codes used to produce the data in Figures_and_Data folder



The code is run by (for example) :
$ python Meta_curves.py

Which communicates with (for example) script_det_numba.py
With all parameters specified within Meta_curves
Data directory, where all data will be saved to, is specified in Meta_curves.py

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

Another pair explores the effect of Ron Vs Roff energy dependency in rates:

Meta_curves_ron.py
script_det_numba_ron.py

Which creates data for the figure:
Fig_Tests (Ron version)

Another pair is for when there is no type 1:

### where?

Another pair is for exploring role of VPS4:

Another pair is for exploring role of VPS4:





Parameters used for each figure are given in the ParametersListed.txt file


