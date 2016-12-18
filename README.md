# Affinity-Propagation-GPU
A PyCUDA implementation of Affinity Propagation by Frey and Dueck for creating clusters from LiDAR 3D point cloud data.
Harrison Liew (hl2670) and Joshua Nuez (jn2548)
EECS E4750 Hybrid Computing in Signal & Data Processing Final Project

Instructions for running the code:

1. The 3D LiDAR point cloud data is stored in /data/data.xyz
2. The /src directory contains the source code:
    - kernel.py contains the baseline implementation
    - kernel_opt.py contains the optimized implementation
    - Running main.py will run the baseline kernel
    - Running main_opt.py will run the optimized kernel
    - ap_timed.py has the Python AP implementation
    - utility.py contains necessary helper functions for parsing the data
    - The *.m are the regular & sparse Matlab AP source code
3. To run the programs:
    - Edit the source file N parameter (size of the desired data). This must be a multiple of 1024.
    - Do ./run.sh main<_opt>.py in the root directory for the PyCUDA programs
    - The program will print the statistics for the clustering, and plot a 3D graph of the clustering results.
    - Close to clustering plot to finish (and release the slurm lock on the GPU)
    - For the Python implementation, run python ap_timed.py (keep N less than 4096!)
    - For the Matlab implementation, run apclusterSparse_test.m (also keep N less than 4096!)
4. The /output folder contains the statistics generated in our tests and used for the final report.
