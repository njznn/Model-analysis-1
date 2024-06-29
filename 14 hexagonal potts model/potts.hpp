#ifndef POTTS
#define POTTS
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

random_device rd{};
mt19937 gen{rd()};

//POTTS MODEL

template<unsigned int NN>
struct pottshex {

  int q;
  static const unsigned int N= NN; // size of square lattice

  double J; // exchange integral
  double E=0.0;
  double H;
  double Esq = 0.0;
  double beta; // beta is 1/T !!
  complex<double> M = 0.0;
  double Msq = 0.0;

  MatrixXd state = MatrixXd::Zero(NN, NN);

  pottshex(int q_, double J_,double H_, double beta_, double seed_): q(q_),J(J_),
   H(H_),beta(beta_){
     gen.seed(seed_);
     uniform_int_distribution<> dist(1, q);
     for (size_t i = 0; i < N; i++) {
       for (size_t j = 0; j < N; j++) {
         state(i,j) = dist(gen);
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

     for (size_t i = 0; i <N; i++) {
       for (size_t j = 0; j < N; j++) {
          M+= exp((2*numbers::pi * 1i *(state(i,j)-1.0))/((double) q));
       }
     }

     Msq = pow(abs(M), 2);
   }

   double energy_Alocal(int &i, int &j){
     return(-J*(((state(i,j)== state((i+1)%N, j))? 1.0:0)+
           ((state(i,j)== state(i, ((j-1)==-1? N-1 : j-1)))? 1.0:0)+
           ((state(i,j)== state(i, (j+1)%N))? 1.0:0)) - H*state(i,j));
   }

   double energy_Blocal(int &i, int &j){
     return(-J*(((state(i,j)== state(((i-1)==-1? N-1 : i-1), j))? 1.0:0)+
           ((state(i,j)== state(i, ((j-1)==-1? N-1 : j-1)))? 1.0:0)+
           ((state(i,j)== state(i, (j+1)%N))? 1.0:0)) - H*state(i,j));
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
double energy_Alocal(pottshex<NN> & obj, int &i, int &j){
  return(-obj.J*(((obj.state(i,j)== obj.state((i+1)%obj.N, j))? 1.0:0)+
        ((obj.state(i,j)== obj.state(i, ((j-1)==-1? obj.N-1 : j-1)))? 1.0:0)+
        ((obj.state(i,j)== obj.state(i, (j+1)%obj.N))? 1.0:0) ) - obj.H*obj.state(i,j));
}

template<unsigned int NN>
double energy_Blocal(pottshex<NN> & obj, int &i, int &j){
  return(-obj.J*(((obj.state(i,j)== obj.state(((i-1)==-1? obj.N-1 : i-1), j))? 1.0:0)+
        ((obj.state(i,j)== obj.state(i, ((j-1)==-1? obj.N-1 : j-1)))? 1.0:0)+
        ((obj.state(i,j)== obj.state(i, (j+1)%obj.N))? 1.0:0) )- obj.H*obj.state(i,j));
}


//THIS ENERGY IS FOR HEXAGONAL LATTICE
template<unsigned int NN>
double energy_one(pottshex<NN> & obj, int i, int j){
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
double energy_one(pottshex<NN> & obj, int i, int j ){
  return(-obj.J* (((obj.state(i,j)== obj.state((i+1)%obj.N, j))? 1.0:0)
  +((obj.state(i,j)== obj.state(((i-1)==-1? obj.N-1 : i-1), j))? 1.0:0)
  +((obj.state(i,j)== obj.state(i, ((j-1)==-1? obj.N-1 : j-1)))? 1.0:0)
  +((obj.state(i,j)== obj.state(i, (j+1)%obj.N))? 1.0:0)));
}
*/


/*
template<unsigned int NN>
void calc_energy(pottshex<NN> &obj){
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
void calc_energy(pottshex<NN> &obj){
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
void calc_mag(pottshex<NN> &obj){
  complex<double> MM = 0;
  for (size_t i = 0; i < obj.N; i++) {
    for (size_t j = 0; j < obj.N; j++) {

      MM += exp((2*numbers::pi * 1i *(obj.state(i,j)-1.0))/((double) obj.q));
      std::cout << exp((2*numbers::pi * 1i *(obj.state(i,j)-1.0))/((double) obj.q)) << '\n';
    }
  }
  obj.M = abs(MM);
}

template<unsigned int NN>
void mc_onemove(pottshex<NN> &obj, int &ri, int &rj , int &new_q, double &ksi){
  double E_state = energy_one(obj, ri, rj);

  auto old_q = obj.state(ri,rj);
  obj.state(ri, rj) = new_q;

  double E_state_new = energy_one(obj, ri, rj);
  double dE = E_state_new-E_state;

  complex<double> dM = exp((2*numbers::pi * 1i *(new_q-1.0))/((double) obj.q)) -
  exp((2*numbers::pi * 1i *(old_q-1.0))/((double) obj.q));


  if (dE<0){
    obj.E += dE;
    obj.M += dM;
  }
  else if (ksi < exp(-obj.beta*dE)){
    obj.E += dE;
    obj.M += dM;
  }
  else {
    obj.state(ri,rj) = old_q;
  }

}

template<unsigned int NN>
void N_sq_flips(pottshex<NN> &obj){
  uniform_int_distribution<> dist_ij(0, obj.N-1);
  uniform_real_distribution<> ksi(0.0, 1.0);
  uniform_int_distribution<> qnew(1, obj.q);
  for (size_t i = 0; i < 10000; i++) {
    int q_st = qnew(gen);
    int ri = dist_ij(gen);
    int rj = dist_ij(gen);
    double ks = ksi(gen);
    while(obj.state(ri, rj)==q_st){
      q_st = qnew(gen);
    }
    mc_onemove(obj, ri, rj, q_st, ks);
  }
}

template<unsigned int NN> // N_relax * N^2 steps!
void relax(pottshex<NN> &obj, int N_relax){
  for (size_t i = 0; i < N_relax; i++) {
    N_sq_flips(obj);
  }
}

template<unsigned int NN>
void potts_simulate_and_flush_beta(int q, double J,double H, double beta_st, int beta_steps, double step_size,
  int Nrelax, int Nsample, std::string ime_dat){
    fstream myfile;
    myfile.open(ime_dat,fstream::out);
    myfile<<"N_relax:"<<Nrelax << "\t Nsample:"<<Nsample<< "\n";
    double beta = beta_st;
    for (size_t i = 0; i < beta_steps; i++) {
      pottshex<NN> pot(q,J,H, beta,12);
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
void flush_state(pottshex<NN> &obj, std::string ime_dat){
  fstream myfile;
  myfile.open(ime_dat,fstream::out);
  for (size_t i = 0; i < obj.N; i++) {
    for (size_t j = 0; j < obj.N; j++) {
      myfile << obj.state(i,j) << "\t";
    }
    myfile << "\n";
  }
}

#endif
