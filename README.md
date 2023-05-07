# n-body simulation using CUDA C++


## how to run

first, compile the version you want run

`cd scripts` because i dont feel like changing the makefile now

`make cuda` for the fully optimized CUDA accelerated version

`make basic` for the basic, unoptimzed CUDA version

`make omp` for the OpenMP accelerated version on the CPU

`make default` for the basic CPU verson

`make check` compares results from the OpenMP version and the fully optimized CUDA version

then to execute the code, run the following command:

`./bin/nbody <n-bodies> <time iterations>`



<hr>

## reporting times

time_nbody.py will run the four versions listed above at varying numbers of bodies and time iterations