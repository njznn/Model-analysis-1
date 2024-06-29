#include <iostream>
#include <Eigen/Dense>
#include <map>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <random>
#include "/home/ziga/Desktop/FMF/magisterij/modelska_1/8_metropolisov_algoritem/matplotlibcpp.h"

using namespace Eigen;
namespace plt = matplotlibcpp;


double magnetiz( const MatrixXi  &M){
  int N = M.rows();
  int mag = 0;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {

      mag += M(i,j) ;

    }
    }

  return(mag);
}


double energija(const MatrixXi & M, int J, double H){
  double E=0;
  int dimv = M.rows();
  int dims = M.cols();
  for (size_t i = 0; i < dimv; i++) {
    for (size_t j = 0; j < dims; j++) {
      if (i==0 && j ==0){
              E += -J*M(i,j)*(M(1, j) + M(dimv-1,0) + M(i, dims-1) + M(0,1)) - H*M(i,j);
        }
      else if (i==0 && j==(dims-1)){
                E += -J*M(i,j)*(M(i, j-1) + M(i,0) + M(dimv-1, j) + M(1,j))-H*M(i,j);}
      else if (i==(dimv-1) && j==0){
                E += -J*M(i,j)*(M(i, j+1) + M(i,dims-1) + M(0, j) + M(i-1,j))-H*M(i,j);}
      else if (i==(dimv-1) && j==(dims-1)){
                E += -J*M(i,j)*(M(i, 0) + M(i,j-1) + M(0, j) + M(i-1,j))- H*M(i,j);}
      else if (i==(dimv-1)){
                E += -J*M(i,j)*(M(0, j) + M(i-1,j) + M(i, j+1) + M(i,j-1))- H*M(i,j);}
      else if (i==0){
                E += -J*M(i,j)*(M(i+1, j) + M(dimv-1,j) + M(i, j+1) + M(i,j-1))- H*M(i,j);}
      else if (j== (dims-1)){
                E += -J*M(i,j)*(M(i+1, j) + M(i-1,j) + M(i, 0) + M(i,j-1))- H*M(i,j);}
      else if (j== 0){
                E += -J*M(i,j)*(M(i+1, j) + M(i-1,j) + M(i, j+1) + M(i,dimv-1))- H*M(i,j);}
      else{
              E += -J*M(i,j)*(M(i+1, j) + M(i-1,j) + M(i, j+1) + M(i,j-1))-H*M(i,j);
            }

    }
  }
  return(E*0.5);
}

void lilmc(MatrixXi &M, int J, double H, double T){
  int dimv = M.rows();
  int dims = M.cols();
  double ksi;
  double dE=0;
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution1(0,dimv-1);
  std::uniform_int_distribution<int> distribution2(0,dims-1);
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  double E_kon =energija(M, J, H);


  for (size_t k = 0; k < 100; k++) {
    int i = distribution1(generator);
    int j = distribution2(generator);


    if (i==0 && j ==0){
            dE = 2*J*M(i,j)*(M(1, j) + M(dimv-1,0) + M(i, dims-1) + M(0,1)) +2*H*M(i,j);
      }
    else if (i==0 && j==(dims-1)){
              dE = 2*J*M(i,j)*(M(i, j-1) + M(i,0) + M(dimv-1, j) + M(1,j))+ 2*H*M(i,j);}
    else if (i==(dimv-1) && j==0){
              dE = 2*J*M(i,j)*(M(i, j+1) + M(i,dims-1) + M(0, j) + M(i-1,j))+2*H*M(i,j);}
    else if (i==(dimv-1) && j==(dims-1)){
              dE = 2*J*M(i,j)*(M(i, 0) + M(i,j-1) + M(0, j) + M(i-1,j))+ 2*H*M(i,j);}
    else if (i==(dimv-1)){
              dE = 2*J*M(i,j)*(M(0, j) + M(i-1,j) + M(i, j+1) + M(i,j-1))+ 2*H*M(i,j);}
    else if (i==0){
              dE = 2*J*M(i,j)*(M(i+1, j) + M(dimv-1,j) + M(i, j+1) + M(i,j-1))+ 2*H*M(i,j);}
    else if (j== (dims-1)){
              dE = 2*J*M(i,j)*(M(i+1, j) + M(i-1,j) + M(i, 0) + M(i,j-1))+ 2*H*M(i,j);}
    else if (j== 0){
              dE = 2*J*M(i,j)*(M(i+1, j) + M(i-1,j) + M(i, j+1) + M(i,dimv-1))+ 2*H*M(i,j);}
    else{
            dE = 2*J*M(i,j)*(M(i+1, j) + M(i-1,j) + M(i, j+1) + M(i,j-1))+ 2*H*M(i,j);
          }
          if (dE<0){
            M(i,j) = (-1)*M(i,j);
            E_kon+= dE;
          }

          else if (distribution(generator)< exp(-dE/T)){
             M(i,j) = (-1)*M(i,j);
             E_kon+= dE;
            }

  }


}



double resitev(MatrixXi M, int Nobr, int J, double H, double T, int st_tock){

  int dimv = M.rows();
  int dims = M.cols();
  int n = 0;

  double dE=0;

  double Eav = 0;


  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution1(0,dimv-1);
  std::uniform_int_distribution<int> distribution2(0,dims-1);
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  double E_kon =energija(M, J, H);
  double Ekvad = 0;
  double Mkvad = 0;

  double ksi;
  double mag = 0 ;
  for (size_t k = 0; k < Nobr; k++) {
    int i = distribution1(generator);
    int j = distribution2(generator);



    if (i==0 && j ==0){
            dE = +2*J*M(i,j)*(M(1, j) + M(dimv-1,0) + M(i, dims-1) + M(0,1)) + 2*H*M(i,j);
      }
    else if (i==0 && j==(dims-1)){
              dE = +2*J*M(i,j)*(M(i, j-1) + M(i,0) + M(dimv-1, j) + M(1,j))+ 2*H*M(i,j);}
    else if (i==(dimv-1) && j==0){
              dE = +2*J*M(i,j)*(M(i, j+1) + M(i,dims-1) + M(0, j) + M(i-1,j))+2*H*M(i,j);}
    else if (i==(dimv-1) && j==(dims-1)){
              dE = +2*J*M(i,j)*(M(i, 0) + M(i,j-1) + M(0, j) + M(i-1,j))+ 2*H*M(i,j);}
    else if (i==(dimv-1)){
              dE = +2*J*M(i,j)*(M(0, j) + M(i-1,j) + M(i, j+1) + M(i,j-1))+2*H*M(i,j);}
    else if (i==0){
              dE = +2*J*M(i,j)*(M(i+1, j) + M(dimv-1,j) + M(i, j+1) + M(i,j-1))+ 2*H*M(i,j);}
    else if (j== (dims-1)){
              dE = +2*J*M(i,j)*(M(i+1, j) + M(i-1,j) + M(i, 0) + M(i,j-1))+ 2*H*M(i,j);}
    else if (j== 0){
              dE = +2*J*M(i,j)*(M(i+1, j) + M(i-1,j) + M(i, j+1) + M(i,dimv-1))+ 2*H*M(i,j);}
    else{
            dE = +2*J*M(i,j)*(M(i+1, j) + M(i-1,j) + M(i, j+1) + M(i,j-1))+ 2*H*M(i,j);
          }
    if (dE<0){
      M(i,j) = (-1)*M(i,j);
      E_kon+= dE;
    }

    else if (distribution(generator)< exp(-dE/T)){
       M(i,j) = (-1)*M(i,j);
       E_kon+= dE;
      }
  }

  for (size_t k = 0; k < st_tock; k++) {

    lilmc(M,J,H,T);

    //std::cout << Ei << '\n';
    //Ekvad += pow(Ei, 2);
    //Eav += Ei;
    double magni = abs(magnetiz(M));
    mag += magni;
    Mkvad += pow(magni, 2);

  }


  //E_povp2 = E_povp2/(dimv*dims);
  //double c = (Ekvad/st_tock- pow((Eav/st_tock), 2))/(dims*dimv * T*T);
  double chi = (Mkvad/st_tock- pow((mag/st_tock), 2))/(dims*dimv * T);
  //E_kon/(dims*dimv);
  //Eav/(st_tock*dims*dimv)

  return(mag/(st_tock*dims*dimv));
}

MatrixXi nakljucna_matrika(int N, int M){
  MatrixXi mat = MatrixXi::Zero(N,M);
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution(0,1);
  for (size_t i = 0; i < mat.rows(); i++) {
    for (size_t j = 0; j < mat.cols(); j++) {
      if (distribution(generator)==0){
        mat(i,j) = -1;
      }
      else {
        mat(i,j)=1;
      }

    }
  }


  return(mat);

}



void narisi_en(){
  int st_tock = 50;
  int st_mag = 4;
  VectorXd T = VectorXd::LinSpaced(30,1 ,3);
  Vector4d h(0,0.01,0.5,1);
  int N = 1000000;

  MatrixXd EN(st_mag, st_tock);
  for (size_t j = 0; j < st_mag; j++) {
    for (size_t i = 0; i < st_tock; i++) {
      MatrixXi matr = nakljucna_matrika(100,100);

      EN(j,i) = resitev(matr,N,1, h[j], T(i),100 );

    }
}

  std::ofstream file("podatki.txt");
  for (size_t i = 0; i < st_tock; i++) {
    file << T(i) << ' '<< EN(0,i)<<'\n';
  }
  file.close();
}






int main() {
  auto Nak = nakljucna_matrika(100,100);
  //auto X = resitev(Nak,1000000,1, 0, 10);
  narisi_en();

  //std::cout << resitev(Nak, 5000000, 1,0,0.5, 1000) << '\n';


    return 0;
}
