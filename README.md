# MultiplexMS - Companion tool
This repository contains the source code of the
MultiplexMS - Companion tool. <br> 
The compiled versions (for Windows and MacOS) of the tool can be found in the 
[Releases](https://github.com/liningtonlab/MultiplexMS/releases) section on 
the right-hand side. 

## How to use the tool

After downloading the right version for your OS, just open the standalone tool.
It might take a moment to start (~10-15 s).

A demo dataset can be downloaded [here](https://github.com/liningtonlab/MultiplexMS/releases).
Around 5-10 min are needed to go through the demo dataset.

A detailed instruction on how to perform the MultiplexMS workflow can be found 
[here](https://liningtonlab.github.io/MultiplexMS_documentation/).

## The MultiplexMS study
All code and data that was used to produce plots in the MultiplexMS study are made available on
Zenodo.
<br>[Link to the Zenodo repository](https://zenodo.org/record/7460400#.Y6IJwnbMI2w) (doi.org/10.5281/zenodo.7460400)
<br>Mass spectrometric data produced for the study was published on MassIVE.
<br>[Link to MS dataset](https://doi.org/10.25345/C5SF2MH02) (dataset MSV000090912)  

## How to compile
If you want to modify the script and desire to recompile the Companion tool, 
please follow the instructions below.<br>
Prerequisites: Python 3.8 and [PyInstaller](https://pyinstaller.org/en/stable/), 
[miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://docs.anaconda.com/anaconda/install/index.html).<br>

Download / Clone the repository and extract files into a folder ('MultiplexMS')

Make changes to the _multiplex_ms_gui.py_, _multiplex_ms_utils.py_ or the _multiplex_ms_gui.spec_<br> 
The _multiplex_ms_gui.spec_ file dictates Pyinstaller how to compile the .py scripts. 

Compile the changed scripts, using the Terminal:

```shell
# Create a fresh conda environment 
conda create -n multiplexms python=3.8
```
```shell
# Activate the environment and install necessary packages
conda activate multiplexms
pip install pyinstaller pandas gooey
```
```shell
# Direct the terminal to the extracted MultiplexMS folder and compile the tool
cd path/to/MultiplexMS
python -m PyInstaller multiplex_ms_gui.spec
```

This should result in two new folders **dist** and **build**, whereby the
executable can be found in **dist**.

## Versions
1.0 - Presubmission

## Feedback
For feedback, questions or comments, please contact either of the following
- [Michael J. J. Recchia](mailto:michael_recchia@sfu.ca)
- [Tim U. H. Baumeister](mailto:tim.baumeister@gmx.de)
- [Roger G. Linington](mailto:rliningt@sfu.ca) (corresponding author)
