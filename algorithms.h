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



void knn_algorithm_CPU(double* predicted_data, double **data, double* distances, int rows, int cols){
  double distance;
  for(int row = 0; row<rows; row +=1){
    distance = 0;
    for(int dimension = 0; dimension<cols; dimension +=1){
      distance += pow(predicted_data[dimension] -  data[row][dimension], 2);
    }
    distances[row] = pow(distance, 0.5);
  }
}



int predict(double* distances, double** target, int k_neighbors, int rows){
  std::vector<std::pair<double, double>> neighbors;
  for (int i = 0; i < rows; i++){
    neighbors.push_back(std::make_pair(distances[i], target[i][0]));
  }
  std::sort(neighbors.begin(), neighbors.end());
  std::vector<std::pair<double, double>>::const_iterator it = neighbors.begin();
  std::vector<int> targets;
  for (int i = 0; i < k_neighbors; i++, it++){
    // cout << it->second << " ";
    targets.push_back(it->second);
  }

  std::map<double, double> frequency;
  for (int x : targets) {
    frequency[x]++;
  }
  
  int max_frequency = 0;
  int mode = -1;
  for (const auto& p : frequency) {
    if (p.second > max_frequency) {
      max_frequency = p.second;
      mode = p.first;
    }
  }
  return mode;
}


