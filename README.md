## GPU101 PiA Course Project

### Scope of the project

The purpose of the project is to provide a GPU accelerated implementation of a given application.

In this case, the application contains the implementations of the Smith-Waterman algorithm, an optimal algorithm for the local alignment of a pair of sequences.  
It consists of computing an alignment matrix H with the following constraints (weights are set before the computation):

<p align="center">
  <img src="https://user-images.githubusercontent.com/100696829/235691393-dc5e1ecf-594c-4642-bf0f-a68250cc3768.png" height="150"/>
</p>

And tracing back the result of the alignment.

<p align="center">
  <img src="https://user-images.githubusercontent.com/100696829/235692632-be9149ab-dedc-4183-b36d-88ac140f6896.png" height="300"/>
</p>

### Content of the repo

The repository contains:
- a naive implementation with a single thread per block: **sw-cuda-single-thread.cu**
- an alternative implementation with shared memory that works only for strings with a shorter length wrt the given one (<= 128): **sw-cuda-shared-shorter-len.cu**
- the final implementation: **sw-cuda.cu**

### Usage

You only need GCC to compile the C program, while you must have CUDA installed on your machine to compile and run the CUDA files.  
If you don’t have access to a GPU, you can use Google Colaboratory (see this guide: https://github.com/albertozeni/gpu_course_colab).

To compile the C program simply type
```
make
```
Within the scope of this folder.
To compile the CUDA programs type
```
nvcc -O3 <name-of-file>
```
