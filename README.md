# Calibration of a Stereo Camera System

This project consists of 2 python files. The first python file is associated with question 1, and the second with question 2. 

## File Path
You will notice that these files read from the resources/bust and resources/fountain directories. If you have ensured the correct file path here, there will not be problems while reading in the files. You will also notice that the programs write to an output/ directory. Please add this directory inline with the resources directory

## Question 1
Open the project in the IDE of choice. I suggest VSCODE. You can simply run the program, all files will be written to the output/ directory, where you can view them. The RANSAC threshold and iteration count is a tunable field, at the top of the python file, directly below the imports.

## Question 2
Similar to question 1, you can simply run the program. Unfortunately, you need to run question 1 before question 2, because question 2 reads the P1 and P2 matrices obtained by question 1, from a text file in the output/ directory, written by question 1. You need to also ensure that you write the projection matrices for the bust image, before running the bust code in question 2. Do the same for the fountain. You can choose which dataset to run, via a tunable boolean in an if-else field at the top of both questions, directly below the imports

