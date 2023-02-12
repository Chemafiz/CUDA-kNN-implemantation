#CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
NVCC_OPTS=-O3 -arch=sm_75 -Xcompiler -Wall -Xcompiler -Wextra -m64

main: main.cu Makefile
	nvcc -o main main.cu $(NVCC_OPTS)

#reduce.o: reduce.cu
#	nvcc -c reduce.cu $(NVCC_OPTS)

clean:
	rm -f *.o main
	
