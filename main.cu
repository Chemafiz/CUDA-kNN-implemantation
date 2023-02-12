#include "algorithms.h"
#include "csv_reader.h"
#include "gpu_functions.h"

using namespace std;



void iris_dataset(){
  int rows, cols, rows_target, cols_target;                                       //reading data for IRIS dataset
  int K = 7;
  double** matrix = readCSV("data/iris_data.csv", rows, cols);  
  double** target = readCSV("data/iris_target.csv", rows_target, cols_target);  
  double* predicted_data;
  int *outputs;
  int mode;
  cudaMallocManaged(&predicted_data, cols * sizeof(double));
  cudaMallocManaged(&outputs, K * sizeof(int));
  double *distances;
  cudaMallocManaged(&distances, rows * sizeof(double)); 

  for(int i=0; i<K; i++){
    outputs[i] = 0;
  }


  predicted_data[0] = 5;                                                          //some data to predict                  
  predicted_data[1] = 0.75;
  predicted_data[2] = 5;
  predicted_data[3] = 0.75; 
  size_t threads_per_block = 256;                                                   //execution configuration
  size_t number_of_blocks = (rows + threads_per_block - 1) / threads_per_block;
  
                                             
  auto start = std::chrono::steady_clock::now();                                        //GPU knn algorithm
  knn_algorithm_GPU<<<number_of_blocks, threads_per_block>>>(predicted_data, matrix, distances, rows, cols);
  cudaDeviceSynchronize();
  mode =  predict(distances, target, 7, rows);
  std::cout << "Elapsed for GPU(microsec)=" << since<std::chrono::microseconds>(start).count() << std::endl;
  cout << "The predicted target for GPU is: " << mode << endl;


  start = std::chrono::steady_clock::now();                                                       //CPU knn algorithm
  knn_algorithm_CPU(predicted_data, matrix, distances, rows, cols);
  mode = predict(distances, target, 7, rows);
  std::cout << "Elapsed for CPU(microsec)=" << since<std::chrono::microseconds>(start).count() << std::endl;
  cout << "The predicted target for CPU is: " << mode << endl;


  for (int i = 0; i < rows; i++) {                            //free memory
    cudaFree(&matrix[i]);  
  }
  cudaFree(&matrix);  
  cudaFree(&distances); 
  cudaFree(&predicted_data);  
}




void mnist_dataset(){
  int rows_x_train, rows_y_train, rows_x_test, rows_y_test, cols, cols_target;                //reading MNIST dataset
  double** x_train = readCSV("data/x_train.csv", rows_x_train, cols);  
  double** y_train = readCSV("data/y_train.csv", rows_y_train, cols_target);  
  double* distances;
  double** x_test = readCSV("data/x_train.csv", rows_x_test, cols);  
  double** y_test = readCSV("data/y_train.csv", rows_y_test, cols_target); 
  double* predictable_sample;
  int correct_prediction = 0;
  cudaMallocManaged(&distances, rows_x_train * sizeof(double));    
  cudaMallocManaged(&predictable_sample, cols * sizeof(double));

  size_t threads_per_block = 1024;                                                 
  size_t number_of_blocks = (rows_x_train + threads_per_block - 1) / threads_per_block;


   //GPU knn algorithm
  int samples_number = 500;
  auto start = std::chrono::steady_clock::now();                
  for(int i= 0; i<samples_number; i++){
    predictable_sample = x_test[i];
    knn_algorithm_GPU<<<number_of_blocks, threads_per_block>>>(predictable_sample, x_train, distances, rows_x_train, cols);
    cudaDeviceSynchronize();

    if(predict(distances, y_train, 7, rows_x_train) == y_test[i][0]){
      correct_prediction++;
    }
    
  }
  std::cout << "Elapsed for GPU =  " << since<std::chrono::microseconds>(start).count() * 1e-6 << "  s" <<std::endl;
  std::cout << "Accuracy =  " << (double)correct_prediction/samples_number * 100 << "%" << std::endl;
  

 //CPU knn algorithm
  correct_prediction = 0;
  start = std::chrono::steady_clock::now();
  for(int i= 0; i<samples_number; i++){
    predictable_sample = x_test[i];
    knn_algorithm_CPU(predictable_sample, x_train, distances, rows_x_train, cols);
      if(predict(distances, y_train, 7, rows_x_train) == y_test[i][0]){
        correct_prediction++;
      }
    
  }
  std::cout << "Elapsed for CPU =  " << since<std::chrono::microseconds>(start).count() * 1e-6 << "  s" <<std::endl;
  std::cout << "Accuracy =  " << (double)correct_prediction / samples_number * 100 << "%" << std::endl;
 


}

int main(){
  //two datasets to test

  iris_dataset();           
  // mnist_dataset();

}
