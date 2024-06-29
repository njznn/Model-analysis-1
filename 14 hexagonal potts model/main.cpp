using namespace std;
#include <fstream>
#include <bitset>
using std::ofstream;
#include "potts.hpp"
#include "ising.hpp"
#include <iomanip>
#include <chrono>
using namespace std::chrono;
#include <omp.h>
#include <Eigen/Geometry>




int main() {
  std::string ime_dat = "/home/ziga/Desktop/FMF/magisterij/modelska_1/"
  "Zakljucna_potts_na_heksagonalni_mrezi/isingstate_128_3_0.txt";
  //std::string ime_dat2 = "/home/ziga/Desktop/FMF/magisterij/modelska_1/"
  //"Zakljucna_potts_na_heksagonalni_mrezi/potts_q5_128_J1_more.txt";

  //potts_simulate_and_flush_beta<128>(4,1,0,0.01,30,0.1,
  //100000, 500, ime_dat);
  //potts_simulate_and_flush_beta<128>(3,1,0.01,30,0.1,
  //300000, 500, ime_dat2);
  //ising_simulate_and_flush_beta<128>(1,0,0.01,15,0.1, 10000, 500, ime_dat);
  /*
  VectorXd H(10);
  H<<0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2;
  for (size_t i = 5; i < 10; i++) {
    std::string ime_dat = "/home/ziga/Desktop/FMF/magisterij/modelska_1/"
    "Zakljucna_potts_na_heksagonalni_mrezi/potts32_q5_h_"+std::to_string(H(i)).substr(0, 4)+".txt";
    potts_simulate_and_flush_beta<32>(5,1,H(i),0.01,30,0.1, 50000, 500, ime_dat);
  }
  */

  auto pot = ising<128>(1,3,0,11);
  relax<128>(pot, 100000);
  std::cout << pot.E << '\n';
  flush_state<128>(pot, ime_dat);



  //fstream myfile;
  //myfile.open(ime_dat,fstream::out);
  //for (size_t i = 0; i < 1000; i++) {
    /* code */
  //}

  //N_sq_flips(pot);

  return 0;
}
