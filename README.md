## GPU101 PiA Course Project

### Scope of the project

The objective of this project is to develop a GPU-accelerated implementation of the Smith-Waterman algorithm, an optimal method for locally aligning pairs of sequences.
The implementation focuses on processing 1000 sequence pairs, where the sequences have a length of 512, and conducting the necessary backtracing operations.

The [report](/docs/report.pdf) contains additional information about the Smith-Waterman algorithm, a comprehensive explanation of my implementation, as well as performance and profiling data.

### Content of the repo

The repository contains:
- in the **sw-original** folder, the original implementation in c: **sw.c**
- in the **sw-accelerated** folder, the final implementation in CUDA: **sw-cuda.cu**

Each folder also contain a specific Makefile.

### Usage

To compile the C program, you only need GCC (GNU Compiler Collection) installed on your machine.
To compile and run CUDA files, you need to have CUDA installed.
Alternatively, if you don’t have access to a GPU, you can use Google Colaboratory.

To compile the C program simply type
```
make
```
Within the scope of the **sw-original** folder.

To compile the CUDA programs type
```
make
```
Within the scope of the **sw-accelerated** folder.

If you are using colab upload the **sw-cuda.cu** file and run the following code block to compile and run the program
```
!nvcc -O3 sw-cuda.cu -o sw-cuda
!./sw-cuda
```
