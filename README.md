# MultiplexMS - Companion tool
This repository contains the source code of the
MultiplexMS - Companion tool. <br> 
The compiled versions (for Win and MacOS) of the tool can be found in the 
[Releases](https://github.com/liningtonlab/MultiplexMS/releases) section on 
the right-hand side. 

## How to use the tool

A detailed instruction on how to perform the MultiplexMS workflow can be found 
[here](https://liningtonlab.github.io/MultiplexMS_documentation/).

## How to compile
In case, you modify the script and desire to recompile the Companion tool, 
please follow the instructions
Prerequisites: Python 3.8 and [PyInstaller](https://pyinstaller.org/en/stable/)
[miniconda](https://docs.conda.io/en/latest/miniconda.html) / [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) (optional). <br>

Download / Clone the repository and extract files into a folder ('MultiplexMS')

Make changes to the _multiplex_ms_gui.py_ or _multiplex_ms_utils.py_ 

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
For feedback, questions or comments, please contact either 
- [Michael J. J. Recchia](mailto:michael_recchia@sfu.ca)
- [Tim U. H. Baumeister](mailto:tim.baumeister@gmx.de)
- [Roger G. Linington](mailto:rliningt@sfu.ca)