# Code and notebooks for the computational imaging session at ADSI Summer School 2019

This repository documents the code included for the lab section for the computational imaging session
at the ADSI Summer School on the Foundations of Data Science in August 2019, held at the
University of Washington.

If you just want to follow along in the lab, the ipynb files are viewable but not runnable on Github.

## Running this in a virtual environment
Run the following code in the command line, which depends only on you having a system Python and 
Anaconda installed somewhere. Tested on Mac OS and Ubuntu. No promises for Windows, sorry.

pyvenv ./ifds_env_test
source ifds_env_test/bin/activate

pip install pywavelets scikit-image pillow imageio matplotlib scipy pybm3d torch==1.2
ipython kernel install --user --name=ifds_env_test

Then open a jupyter notebook with the command:
jupyter notebook

And open an ipynb file and enjoy!

## Dependencies
To run this code yourself, you'll need the following packages:
pywavelets
scikit-image
pillow
imageio
matplotlib
scipy
pybm3d
torch==1.2

## Caveats and sources
The code here is written for illustration and instruction purpose, not speed.
The code for the DnCNN example is adapted from https://github.com/cszn/DnCNN/
If you use this code yourself for demo purposes, please contact the author Davis Gilton, 
whose email is his last name at wisc.edu.
