omp:
	g++ -O3 -fopenmp ../src/nbody_cpu.cpp -o ../bin/nbody

default:
	g++ -O3 ../src/nbody_cpu.cpp -o ../bin/nbody

check:
	nvcc -Xcompiler=-fopenmp -ftz=true -DCHECK -prec-sqrt=true ../src/check.cu -o ../bin/nbody

cuda:
	nvcc -Xcompiler=-fopenmp -ftz=true ../src/nbody.cu -o ../bin/nbody

basic:
	nvcc -Xcompiler=-fopenmp -ftz=true -prec-sqrt=true ../src/basic.cu -o ../bin/nbody

no_ftz:
	nvcc -Xcompiler=-fopenmp ../src/nbody.cu -o ../bin/nbody

clean:
	rm ../bin/*