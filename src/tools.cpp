#include "tools.h"
#include <iostream>
#include <cmath>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  //  Check the estimation vector size should not be zero
  if (estimations.size() == 0) {
    std::cout << "ERROR: estimations vector has size 0 in CalculateRMSE function" << std::endl;
    return rmse;
  }
  // Check the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size()) {
    std::cout << "ERROR: estimations vector and ground_truth vector have different sizes in CalculateRMSE function" << std::endl;
    return rmse;
  }
  // Accumulate squared residuals
  for (int i=0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  rmse /= estimations.size();
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  Hj << 0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;

  // Recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // Prevent division by zero
  if (px == 0 && py == 0) {
      std::cout << "Error: both px and py are 0" << std::endl;
      return Hj;
  }
  
  // Compute the Jacobian matrix
  float sqrt_sum = sqrt(sum);
  float qx = px / sqrt_sum;
  float qy = py / sqrt_sum;
  Hj <<  qx, qy, 0, 0,
        -(py / sum), (px / sum), 0, 0,
         (py * (vx * py - vy * px)) / pow(sum, 3 / 2), (px * (vy * px - vx * py)) / pow(sum, 3 / 2), qx, qy;

  return Hj;
}
