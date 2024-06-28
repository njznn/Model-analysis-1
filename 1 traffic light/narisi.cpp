
#include <iostream>
#include <Eigen/Dense>
#include <map>
#include <vector>
#include <cmath>

#include "gnuplot-iostream.h"
using namespace std;
using namespace Eigen;
# define PI           3.14159265358979323846
#include <fstream>
using std::ofstream;




int main(int argc, char const *argv[]) {
  int st_tock=500;
  VectorXd v1(10);
  v1<<0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8;
  VectorXd t = VectorXd::LinSpaced(500,0,1);
  MatrixXd v_prost1 = VectorXd::Zero(500,11);
  v_prost1(all,0) = t;
  for (size_t j = 1; j < 11; j++) {
    for (size_t i = 0; i < 500; i++) {
      v_prost1(i,j) = -(3/2)*(1-v1[j-1])*pow(t[i],2) + 3*(1-v1[j-1])*t[i] + v1[j-1];
    }
  }
    std::cout << v_prost1 << '\n';
  return 0;
}
