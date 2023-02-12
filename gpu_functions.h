#pragma once
#include <iostream>
#include <ctime>
#include <cmath>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <map>
#include <chrono>
#include <thread>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;  


__global__ void knn_algorithm_GPU(double* predicted_data, double **data, double* distances, int rows, int cols){
  int idx = threadIdx.x+ blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  double distance;
  while(idx < rows){

    distance = 0;
    for(int dimension = 0; dimension<cols; dimension +=1){
      distance += pow(predicted_data[dimension] -  data[idx][dimension], 2);
    }
    distances[idx] =  pow(distance, 0.5);
    idx += stride;
  }

}



void gpu_error(cudaError_t err){
    if (err != cudaSuccess){
        std::cout << "CUDA Runtime Error at: " << std::endl;
        std::cout << cudaGetErrorString(err) << std::endl;
  }
}