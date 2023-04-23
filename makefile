omp:
	g++ -O3 -fopenmp nbody_cpu.cpp -o bin/nbody

default:
	g++ -O3 nbody_cpu.cpp -o bin/nbody

check:
	nvcc -Xcompiler=-fopenmp -ftz=true -DCHECK nbody.cu -o bin/nbody

cuda:
	nvcc -Xcompiler=-fopenmp -ftz=true nbody.cu -o bin/nbody

clean:
	rm bin/*