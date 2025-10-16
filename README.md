CLASS++ Implementation of decaying neutrinos to dark radiation and lighter neutrinos
==============================================

Authors: Guillermo Franco Abellán and Nicola Terzaghi.

This code allows to compute the effects of neutrinos decaying non-relativistically into dark radiation (DR) and lighter neutrinos in a manner 
consistent with the mass splittings from oscillation data, as detailed in the preprint 2510.XXXXX. It is based on a modification of 
CLASS++, a version of CLASS (Julien Lesgourgues, Thomas Tram) written in C++ that was developed by Emil Brinch Holm and Thomas Tram in 
2205.13628 to model the effects of warm dark matter (WDM) decaying non-relativistically into DR.

For information on how to run the code, see the input files:

- `LCDMnu_dec_toNCDM.ini`. This file contains information of the relevant parameters for various scenarios of neutrino decays into DR and 
lighter neutrinos. These scenarios (labeled A1-A3, B1-B2) are classified according to the decay mass gap (solar or atmospheric), the mass ordering 
(normal or inverted), and the number of decay channels (single-decay channel or two-decay channel with the degenerate $\nu_{1,2}$ approximation).

- `LCDMnu_dec_toDR.ini`. This file contains information of the relevant parameters for neutrino decays entirely into DR, assuming three 
degenerate neutrino states. 

- `LCDM.ini` and `LCDMnu.ini`. These files contain information of the relevant parameters for standard LCDM in the presence of 
massless and massive neutrinos, respectively. The second file allows to compute the stable limit of scenarios A1-A3, B1-B2 in 
`LCDMnu_dec_toNCDM.ini`, i.e. those with the same neutrino mass spectrum but in the absence of decays.

The folder `CNB_calculations` contains Jupyter notebooks and auxiliary files that allow to reproduce the plots in the preprint 2510.XXXXX. In 
addition, the notebook `plot_PSD_evolution.ipynb` allows to generate a gif showing the evolution for the phase-space distribution of the 
parent and daughter neutrinos. Finally, the file `LCDMnu_negative.ini` contains information on how to specify negative neutrino masses (only 
relevant for stable neutrinos if one wants to quantify the tension between cosmological and oscillation data).

Any questions or comments may be directed to g.francoabellan@gmail.com or nicola.terzaghi@gmail.com.

------- THE REST IS THE ORIGINAL CLASS README -------

CLASS: Cosmic Linear Anisotropy Solving System 
==============================================

Authors: Julien Lesgourgues and Thomas Tram

with several major inputs from other people, especially Benjamin
Audren, Simon Prunet, Jesus Torrado, Miguel Zumalacarregui, Francesco
Montanari, etc.

For download and information, see http://class-code.net


Compiling CLASS and getting started
-----------------------------------

(the information below can also be found on the webpage, just below
the download button)

Download the code from the webpage and unpack the archive (tar -zxvf
class_vx.y.z.tar.gz), or clone it from
https://github.com/lesgourg/class_public. Go to the class directory
(cd class/ or class_public/ or class_vx.y.z/) and compile (make clean;
make class). You can usually speed up compilation with the option -j:
make -j class. If the first compilation attempt fails, you may need to
open the Makefile and adapt the name of the compiler (default: gcc),
of the optimization flag (default: -O4 -ffast-math) and of the OpenMP
flag (default: -fopenmp; this flag is facultative, you are free to
compile without OpenMP if you don't want parallel execution; note that
you need the version 4.2 or higher of gcc to be able to compile with
-fopenmp). Many more details on the CLASS compilation are given on the
wiki page

https://github.com/lesgourg/class_public/wiki/Installation

(in particular, for compiling on Mac >= 10.9 despite of the clang
incompatibility with OpenMP).

To check that the code runs, type:

    ./class explanatory.ini

The explanatory.ini file is THE reference input file, containing and
explaining the use of all possible input parameters. We recommend to
read it, to keep it unchanged (for future reference), and to create
for your own purposes some shorter input files, containing only the
input lines which are useful for you. Input files must have a *.ini
extension.

If you want to play with the precision/speed of the code, you can use
one of the provided precision files (e.g. cl_permille.pre) or modify
one of them, and run with two input files, for instance:

    ./class test.ini cl_permille.pre

The files *.pre are suppposed to specify the precision parameters for
which you don't want to keep default values. If you find it more
convenient, you can pass these precision parameter values in your *.ini
file instead of an additional *.pre file.

The automatically-generated documentation is located in

    doc/manual/html/index.html
    doc/manual/CLASS_manual.pdf

On top of that, if you wish to modify the code, you will find lots of
comments directly in the files.

Python
------

To use CLASS from python, or ipython notebooks, or from the Monte
Python parameter extraction code, you need to compile not only the
code, but also its python wrapper. This can be done by typing just
'make' instead of 'make class' (or for speeding up: 'make -j'). More
details on the wrapper and its compilation are found on the wiki page

https://github.com/lesgourg/class_public/wiki

Plotting utility
----------------

Since version 2.3, the package includes an improved plotting script
called CPU.py (Class Plotting Utility), written by Benjamin Audren and
Jesus Torrado. It can plot the Cl's, the P(k) or any other CLASS
output, for one or several models, as well as their ratio or percentage
difference. The syntax and list of available options is obtained by
typing 'pyhton CPU.py -h'. There is a similar script for MATLAB,
written by Thomas Tram. To use it, once in MATLAB, type 'help
plot_CLASS_output.m'

Developing the code
--------------------

If you want to develop the code, we suggest that you download it from
the github webpage

https://github.com/lesgourg/class_public

rather than from class-code.net. Then you will enjoy all the feature
of git repositories. You can even develop your own branch and get it
merged to the public distribution. For related instructions, check

https://github.com/lesgourg/class_public/wiki/Public-Contributing

Using the code
--------------

You can use CLASS freely, provided that in your publications, you cite
at least the paper `CLASS II: Approximation schemes <http://arxiv.org/abs/1104.2933>`. Feel free to cite more CLASS papers!

Support
-------

To get support, please open a new issue on the

https://github.com/lesgourg/class_public

webpage!
