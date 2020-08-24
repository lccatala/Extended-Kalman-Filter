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
  float c1 = px*px + py*py;
  float c2 = sqrt(c1);
  float c3 = c2*c1;

  if (fabs(c1) < 0.0001) {
    px = py = 0.0001;
    c1 = px*px + py*py;
    c2 = sqrt(c1);
    c3 = c2*c1;
  }
  
  Hj <<  px/c2,                 py/c2,               0,     0,
        -py/c1,                 px/c1,               0,     0, 
         py*(vx*py - vy*px)/c3, px*(vy*px-vx*py)/c3, px/c2, py/c2;

  return Hj;
}
