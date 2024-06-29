#ifndef ISING
#define ISING
#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;
#include <math.h>
#include <cmath>
#include <stdlib.h>
#include <complex>
#include <bitset>
#include <time.h>
using std::ofstream;
#include <random>
#include <omp.h>

//random_device rd{};
//mt19937 gen{rd()};

//ising MODEL

template<unsigned int NN>
struct ising {

  static const unsigned int N= NN; // size of square lattice

  double J; // exchange integral
  double E=0.0;
  double Esq = 0.0;
  double beta; // beta is 1/T !!
  double M = 0.0;
  double Msq = 0.0;
  double H = 0.0;

  MatrixXd state = MatrixXd::Zero(NN, NN);

  ising(double J_, double beta_,double H_, double seed_): J(J_),
   beta(beta_),H(H_){
     gen.seed(seed_);
     uniform_int_distribution<> dist(0, 1);
     for (size_t i = 0; i < N; i++) {
       for (size_t j = 0; j < N; j++) {
         state(i,j) = dist(gen)==0? -1:1;
       }
     }
     //calculate energy and magnetization of random state
     double Htemp = H;
     H = 0;
     for (size_t i = 0; i < N; i++) {
       for (size_t j = 0; j < N; j++) {
         E += energy_one(i, j );
       }
     }

     E = E*0.5;
     H = Htemp;
     for (size_t i = 0; i < N; i++) {
       for (size_t j = 0; j < N; j++) {
         E -= H*state(i,j);
       }
     }
     Esq = pow(E, 2);

     for (size_t i = 0; i < N; i++) {
       for (size_t j = 0; j < N; j++) {

         M += state(i,j) ;

       }
    }

     Msq = pow(M, 2);
   }
   double energy_Alocal(int &i, int &j){
     return(-J*(state(i,j)*state((i+1)%N, j)+
           state(i,j)*state(i, ((j-1)==-1? N-1 : j-1))+
           (state(i,j)*state(i, (j+1)%N))) - H*state(i,j));
   }

   double energy_Blocal(int &i, int &j){
     return(-J*(state(i,j)*state(((i-1)==-1? N-1 : i-1), j)+
           state(i,j)*state(i, ((j-1)==-1? N-1 : j-1))+
           state(i,j)*state(i, (j+1)%N)) - H*state(i,j) );
   }

   /*
   double energy_one(int i, int j ){
     return(-J* (((state(i,j)== state((i+1)%N, j))? 1.0:0)
     +((state(i,j)== state(((i-1)==-1? N-1 : i-1), j))? 1.0:0)
     +((state(i,j)== state(i, ((j-1)==-1? N-1 : j-1)))? 1.0:0)
     +((state(i,j)== state(i, (j+1)%N))? 1.0:0)));
   }
   */
   double energy_one(int i, int j){
     if (i%2 == 0){
       if (j%2 ==0){
         return energy_Blocal(i, j);
       }
       else{
         return energy_Alocal(i, j);
       }
     }
     else{
       if (j%2 ==0){
         return energy_Alocal(i, j);
       }
       else{
         return energy_Blocal(i, j);
       }
     }
   }

};






template<unsigned int NN>
double energy_Alocal(ising<NN> & obj, int &i, int &j){
  return(-obj.J*(obj.state(i,j)*obj.state((i+1)%obj.N, j)+
        obj.state(i,j)*obj.state(i, ((j-1)==-1? obj.N-1 : j-1))+
        (obj.state(i,j)*obj.state(i, (j+1)%obj.N))) - obj.H*obj.state(i,j) );
}

template<unsigned int NN>
double energy_Blocal(ising<NN> & obj, int &i, int &j){
  return(-obj.J*(obj.state(i,j)*obj.state(((i-1)==-1? obj.N-1 : i-1), j)+
        obj.state(i,j)*obj.state(i, ((j-1)==-1? obj.N-1 : j-1))+
        (obj.state(i,j)*obj.state(i, (j+1)%obj.N))) - obj.H*obj.state(i,j));
}


//THIS ENERGY IS FOR HEXAGONAL LATTICE
template<unsigned int NN>
double energy_one(ising<NN> & obj, int i, int j){
  if (i%2 == 0){
    if (j%2 ==0){
      return energy_Blocal(obj, i, j);
    }
    else{
      return energy_Alocal(obj, i, j);
    }
  }
  else{
    if (j%2 ==0){
      return energy_Alocal(obj, i, j);
    }
    else{
      return energy_Blocal(obj, i, j);
    }
  }
}


/*
//THIS ENERGY IS FOR SQUARE LATTICE without external field
template<unsigned int NN>
double energy_one(ising<NN> & obj, int i, int j ){
  return(-obj.J* (((obj.state(i,j)== obj.state((i+1)%obj.N, j))? 1.0:0)
  +((obj.state(i,j)== obj.state(((i-1)==-1? obj.N-1 : i-1), j))? 1.0:0)
  +((obj.state(i,j)== obj.state(i, ((j-1)==-1? obj.N-1 : j-1)))? 1.0:0)
  +((obj.state(i,j)== obj.state(i, (j+1)%obj.N))? 1.0:0)));
}
*/


/*
template<unsigned int NN>
void calc_energy(ising<NN> &obj){
    obj.E = 0.0;
    for (size_t i = 0; i < obj.N; i++) {
      for (size_t j = 0; j < obj.N; j++) {
        obj.E -= obj.J* (((obj.state(i,j)== obj.state((i+1)%obj.N, j))? 1.0:0)
        +((obj.state(i,j)== obj.state(((i-1)==-1? obj.N-1 : i-1), j))? 1.0:0)
        +((obj.state(i,j)== obj.state(i, ((j-1)==-1? obj.N-1 : j-1)))? 1.0:0)
        +((obj.state(i,j)== obj.state(i, (j+1)%obj.N))? 1.0:0));
      }
    }
    obj.E = obj.E*0.5;
}
*/
template<unsigned int NN>
void calc_energy(ising<NN> &obj){

    obj.E = 0.0;
    double Htemp = obj.H;
    obj.H = 0;
    for (size_t i = 0; i < obj.N; i++) {
      for (size_t j = 0; j < obj.N; j++) {
        obj.E += energy_one<NN>(obj, i, j );
      }
    }
    obj.E = obj.E*0.5;
    obj.H = Htemp;
    for (size_t i = 0; i < obj.N; i++) {
      for (size_t j = 0; j < obj.N; j++) {
        obj.E -= obj.H*obj.state(i,j);
      }
    }
}

template<unsigned int NN>
void calc_mag(ising<NN> &obj){
  obj.M = 0;
  for (size_t i = 0; i < obj.N; i++) {
    for (size_t j = 0; j < obj.N; j++) {
      obj.M += obj.state(i,j);

    }
  }
}

template<unsigned int NN>
void mc_onemove(ising<NN> &obj, int &ri, int &rj, double &ksi){
  double E_state = energy_one(obj, ri, rj);

  auto old_q = obj.state(ri,rj);
  obj.state(ri, rj) = (-1)* old_q;

  double E_state_new = energy_one(obj, ri, rj);
  double dE = E_state_new-E_state;

  double dM = obj.state(ri, rj) - old_q;
  double faktor = obj.beta;

  if (obj.beta > 0.8){
    faktor = 0.7*obj.beta;
  }

  if (dE<0){
    obj.E += dE;
    obj.M += dM;
  }
  else if (ksi < exp(-faktor*dE)){
    obj.E += dE;
    obj.M += dM;
  }
  else {
    obj.state(ri,rj) *= (-1) ;
  }

}

template<unsigned int NN>
void N_sq_flips(ising<NN> &obj){
  uniform_int_distribution<> dist_ij(0, obj.N-1);
  uniform_real_distribution<> ksi(0.0, 1.0);
  for (size_t i = 0; i < 10000; i++) {
    int ri = dist_ij(gen);
    int rj = dist_ij(gen);
    double ks = ksi(gen);

    mc_onemove(obj, ri, rj, ks);
  }
}

template<unsigned int NN> // N_relax * N^2 steps!
void relax(ising<NN> &obj, int N_relax){
  for (size_t i = 0; i < N_relax; i++) {
    N_sq_flips(obj);
  }
}

template<unsigned int NN>
void ising_simulate_and_flush_beta( double J,double H,double beta_st, int beta_steps, double step_size,
  int Nrelax, int Nsample, std::string ime_dat){
    fstream myfile;
    myfile.open(ime_dat,fstream::out);
    myfile<<"N_relax:"<<Nrelax << "\t Nsample:"<<Nsample<< "\n";
    double beta = beta_st;
    for (size_t i = 0; i < beta_steps; i++) {
      ising<NN> pot(J,beta,H,i);
      relax(pot, Nrelax);
      double avgmag = 0;
      double avgen = 0;
      double avgmagsq = 0;
      double avgensq =0;
      for (size_t j = 0; j < Nsample; j++) {
          N_sq_flips(pot);
          avgmag += abs(pot.M);
          avgen += pot.E;
          avgensq += pow(pot.E, 2);
          avgmagsq += pow(abs(pot.M), 2);
        }

      avgmag = avgmag/(Nsample);
      avgen = avgen/(Nsample);
      avgmagsq = avgmagsq/(Nsample);
      avgensq = avgensq/(Nsample);
      myfile << beta << "\t"<< avgen << "\t" << avgmag << "\t" << avgensq << "\t" << avgmagsq << "\n";
      beta += step_size;

    }
  }

template<unsigned int NN>
void flush_state(ising<NN> &obj, std::string ime_dat){
  fstream myfile;
  myfile.open(ime_dat,fstream::out);
  myfile<<"beta:"<<obj.beta << '\t'<<"H"<<obj.H<< '\t'<<"J:"<<obj.J<< "\n";
  for (size_t i = 0; i < obj.N; i++) {
    for (size_t j = 0; j < obj.N; j++) {
      myfile << obj.state(i,j) << "\t";
    }
    myfile << "\n";
  }
}

#endif
