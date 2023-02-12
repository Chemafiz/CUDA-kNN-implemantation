#pragma once
#include "gpu_functions.h"


double** readCSV(string filename, int& rows, int& cols) {
  ifstream file(filename);  // Open the file
  string line;
  rows = 0;  // Initialize the number of rows and columns
  cols = 0;
  while (getline(file, line)) {  // Read each line
    stringstream ss(line);  // Convert the line to a string stream
    string field;
    int c = 0;  // Count the number of fields (columns) in the line
    while (getline(ss, field, ',')) {  // Split the line by comma
      c++;
    }
    if (c > cols) {  // Update the number of columns if necessary
      cols = c;
    }
    rows++;  // Increment the number of rows
  }
  file.close();  // Close the file
  double** matrix;
  // double** matrix = new double*[rows];  // Allocate the matrix
  cudaMallocManaged(&matrix, rows * sizeof(double*));
  for (int i = 0; i < rows; i++) {
    // matrix[i] = new double[cols];
      cudaMallocManaged(&matrix[i], cols * sizeof(double));

  }

  file.open(filename);  // Reopen the file
  int r = 0;
  while (getline(file, line)) {  // Read each line again
    stringstream ss(line);  // Convert the line to a string stream
    string field;
    int c = 0;
    while (getline(ss, field, ',')) {  // Split the line by comma
      matrix[r][c] = stod(field);  // Convert the field to a double and store it in the matrix
      c++;
    }
    r++;
  }
  file.close();  // Close the file

  return matrix;
}



template <
    class result_t   = std::chrono::milliseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}