/*
    init_variables.cpp: struct with static members to compute and store all constant variables used in the program.
    Change physical and numerical parameters here.
   
    Author: Théo Monpouet Ekeram
    As part of a master thesis in Computational Mathematics, KTH Royal Institute of Technology
    Thesis name: 'Parallel implementation and analysis of the discrete-velocity BGK Boltzmann method'
    Date: 2025/06/01
*/



#include <iostream>
#include <math.h>
#include <complex>
#include <valarray>
#include <algorithm>
#include <string>

#include <mpi.h>
#include <fftw3-mpi.h>

using namespace std;


// Custom mod function
int mod(int a, int b) {
    return (a % b + b) % b; // -6%12 would be -6 using c++:s mod, "should" be +6
}


struct Constants {
    // Adjustable physical parameters
    static const double nu;
    static const double epsilon;
    static const double Lx;
    static const double Ly;
    static const double T0;
    static const double T1;

    // Adjustable Numerical parameters
    static const int N;
    static const double dt;

    // Initial conditions
    static const string init_condition;
    static const string init_file_path;

    // Saving solution
    static const string result_file_path;
    static const string error_file_suffix;
    static const string saving_sol;
    static const int Nsave;

    // Fixed parameters
    static const int N_half;
    static const double c_s;
    static const double dx;
    static const double dy;
    static const int Nd;
    static const int TSCREEN;


    // MPI constants
    static ptrdiff_t local_alloc_ps;
    static ptrdiff_t local_alloc_fs;
    static ptrdiff_t local_N_ps;
    static ptrdiff_t local_N_fs;
    static ptrdiff_t local_start_ps;
    static ptrdiff_t local_start_fs;
    static int rank;
    static int size;

    // Wavenumbers
    static valarray<complex<double>> Kx;
    static valarray<complex<double>> Ky;

    // Operators
    static valarray<complex<double>> Lap_hat;
    static valarray<complex<double>> Pois_hat;
    static valarray<complex<double>> FTh;

    // Dealias filter
    static valarray<complex<double>> dealias;

    // Initial conditions
    static valarray<double> w_0;
    
    // Weights
    static const int lattice_number;
    static valarray<double> weights;


    
    // set_sizes(): Static method to compute and set sizes of operator arrays
    // Inputs:
    //    - int size_: How many total processes there are.
    //    - int rank_: Which process (0-size_) is the current running process.
    static void set_sizes(int size_, int rank_) {

        // Load distribution between processes
        local_alloc_ps = fftw_mpi_local_size_2d(N, N_half, MPI_COMM_WORLD, &local_N_ps, &local_start_ps);
        local_alloc_fs = fftw_mpi_local_size_2d(N_half, N, MPI_COMM_WORLD, &local_N_fs, &local_start_fs);

        size = size_;
        rank = rank_;
        
        // initializing operator sizes
        Kx.resize(local_alloc_fs);
        Ky.resize(local_alloc_fs);
        Lap_hat.resize(local_alloc_fs);
        Pois_hat.resize(local_alloc_fs);
        dealias.resize(local_alloc_fs);
        FTh.resize(local_alloc_fs * lattice_number);

        w_0.resize(2 * local_alloc_ps);
    }



    // set_initials(): Static method to set the inital condition using the zero-mean condition.
    static void set_initials() {
        valarray<complex<double>> w_0hat(local_alloc_fs);
        valarray<double> inverse_output(2 * local_alloc_ps);
        valarray<double> w_0copy(2 * local_alloc_ps);

        // Initializing temporary fftw plans
        fftw_plan forward_plan_init = fftw_mpi_plan_dft_r2c_2d(N, N, &w_0copy[0], reinterpret_cast<fftw_complex*>(&w_0hat[0]), MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT);
        fftw_plan inverse_plan_init = fftw_mpi_plan_dft_c2r_2d(N, N, reinterpret_cast<fftw_complex*>(&w_0hat[0]), &inverse_output[0], MPI_COMM_WORLD, FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN);
 
        // Set closed form initial condition
        for (int i = 0; i < local_N_ps; i++) {
            for (int j = 0; j < N; j++) {
                double x = j * dx;
                double y = (local_start_ps + i ) * dy;

                if (init_condition == "tg") {
                    // Taylor-Green Vortex
                    w_0[i * 2*(N/2+1) + j] = 10 * sin(2*M_PI*2*x) * sin(2*M_PI*2*y);
                } else {
                    // Perturbed Taylor-Green Vortex
                    w_0[i * 2*(N/2+1) + j] = (-sin(2 * M_PI * x) * sin(2 * M_PI * y)) + exp(-(pow(x-0.5, 2)+pow(y-0.5,2)) / (0.02));
                }
            }
        }

        // Zero-mean condition
        w_0copy = w_0;
        fftw_execute(forward_plan_init);

        w_0hat[0] = complex<double>(0.0, 0.0);

        fftw_execute(inverse_plan_init);
        w_0 = inverse_output / (Constants::N * Constants::N);

        // Free memory
        fftw_destroy_plan(forward_plan_init);
        fftw_destroy_plan(inverse_plan_init);
    }


    
    // set_operators(): Static method to compute and set operators Kx, Ky, Lap_hat, Pois_hat, FTh and dealias
    static void set_operators() {

        valarray<double> kmod(local_alloc_fs);


        // Kx, Ky
        for (int i = 0; i < local_N_fs; i++) {   
            int tempx = mod((int) ((local_start_fs + i + 1) - ceil(N / 2.0 + 1)), N) - floor(N / 2.0);
            if (local_start_fs + i == round(N/2)) tempx = round(N/2);

            for (int j = 0; j < N; j++) {
                int tempy = mod((int) ((j + 1) - ceil(N / 2.0 + 1)), N) - floor(N / 2.0);
                if (j == round(N/2)) tempy = round(N/2);

                int index = i * N + j;

                Kx[index] = complex<double>(0, (2*M_PI / Lx) * tempx);
                Ky[index] = complex<double>(0, (2*M_PI / Lx) * tempy);



                // Lap_hat
                double temp = real((Kx[index]*Kx[index] + Ky[index]*Ky[index]));
                Lap_hat[index] = complex<double>(temp, 0);

                // Pois_hat
                Pois_hat[index] = complex<double>(temp, 0);
                if (index == 0 && rank == 0) Pois_hat[index] = complex<double>(1.0, 0);;

                // kmod
                kmod[index] = sqrt(abs(Kx[index])*abs(Kx[index]) + abs(Ky[index])*abs(Ky[index]));


                if (local_start_fs + i == round(N/2)) Kx[index] *= 0;
                if (j == round(N/2)) Ky[index] *= 0;
            }
        }

        
        // FTh
        int offset = Constants::local_alloc_fs;
        FTh[slice(0*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (Kx) / epsilon);
        FTh[slice(1*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (Ky) / epsilon);
        FTh[slice(2*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (-Kx) / epsilon);
        FTh[slice(3*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (-Ky) / epsilon);
        FTh[slice(4*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (Kx + Ky) / epsilon);
        FTh[slice(5*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (-Kx + Ky) / epsilon);
        FTh[slice(6*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (-Kx - Ky) / epsilon);
        FTh[slice(7*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (Kx - Ky) / epsilon);
        FTh[slice(8*offset, offset, 1)] = -(1/(epsilon*epsilon*nu) + (Kx - Kx) / epsilon); // 0

        // kcut
        // find global max, Allreduce with MPI_MAX to reduce and broadcast
        double local_max = kmod.max();
        double kcut;
        MPI_Allreduce(&local_max, &kcut, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        kcut *= 2.0/3.0;

        // dealias
        for (int i = 0; i < local_N_fs; i++) {     
            for (int j = 0; j < N; j++) {
                int index = i * N + j;
                dealias[index] = complex<double>(exp(-36.0 * pow(kmod[index]/kcut, 36)), 0);
            }
        }

    }

    
    // initialize(): Static method to initialize sizes, initial conditions and operators
    static void initialize(int size_, int rank_, double e0) {
        epsilon = e0;
        set_sizes(size_, rank_);
        if (init_condition != "fromfile") set_initials();
        set_operators();
    }
};



// Adjustable physical parameters
const double Constants::nu = pow(10, -4);
double Constants::epsilon;
const double Constants::Lx = 1.0;
const double Constants::Ly = 1.0;
const double Constants::T0 = 0.0;
const double Constants::T1 = 1;

// Adjustable Numerical parameters
const int Constants::N = 128;
const double Constants::dt = 2.0*pow(10, -4);


// Initial condition
// "tg":  Taylor-Green Vortex
// "ptg": Perturbed Taylor-Green Vortex
// "fromfile": Read the input from file specified by variable init_filepath
const string Constants::init_condition = "tg";
const string Constants::init_file_path = "/"; // File path (folder) of where to read initial condition


// Saving solution
const string Constants::result_file_path = "/cfs/klemming/projects/supr/latticeboltzmann_2025/extension1/testD2Q9Tg/"; // File path of where to save solution (end with "/")
const string Constants::error_file_suffix = "_" + Constants::init_condition + "_e" + to_string(Constants::epsilon);
const string Constants::saving_sol = "e"; // "w": Save vorticity, "e": save error, "we": save both (only applicable to IC "tg", for "ptg" only vorticity gets saved)
const int Constants::Nsave = 1000; // Number of steps saved



// Fixed parameters
const int Constants::N_half = Constants::N/2 + 1;
const double Constants::c_s = pow(1.0/3.0, 1.0/2.0);
const double Constants::dx = Lx/N;
const double Constants::dy = Ly/N;
const int Constants::Nd = round(T1/dt);
const int Constants::TSCREEN = floor(T1 / (dt * Nsave));


// MPI constants
ptrdiff_t Constants::local_alloc_ps;
ptrdiff_t Constants::local_alloc_fs;
ptrdiff_t Constants::local_N_ps;
ptrdiff_t Constants::local_N_fs;
ptrdiff_t Constants::local_start_ps;
ptrdiff_t Constants::local_start_fs;
int Constants::rank;
int Constants::size;


// Wavenumbers
valarray<complex<double>> Constants::Kx;
valarray<complex<double>> Constants::Ky;

// Operators
valarray<complex<double>> Constants::Lap_hat;
valarray<complex<double>> Constants::Pois_hat;
valarray<complex<double>> Constants::FTh;


// Dealias filter
valarray<complex<double>> Constants::dealias;


// Initial conditions
valarray<double> Constants::w_0;


// Weights
const int Constants::lattice_number = 9;
valarray<double> Constants::weights{1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 4.0/9.0};


