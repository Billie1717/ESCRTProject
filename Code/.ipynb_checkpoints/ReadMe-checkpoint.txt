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
Fig_Indentation
Fig_DoSPlot
Fig_Pitch

Another pair explores the effect of Ron Vs Roff energy dependency in rates:

Meta_curves_ron.py
script_det_numba_ron.py

Which creates data for the figure:
Fig_Tests (Ron version)




Parameters used for each figure are given in the ParametersListed.txt file


