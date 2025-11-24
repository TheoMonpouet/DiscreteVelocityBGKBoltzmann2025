/*
    functions.cpp: the functions used in the program for the timestepping and saving of solutions.
   
    Author: Théo Monpouet Ekeram
    As part of a master thesis in Computational Mathematics, KTH Royal Institute of Technology
    Thesis name: 'Parallel implementation and analysis of the discrete-velocity BGK Boltzmann method'
    Date: 2025/06/01
*/




#include <iostream>
#include <math.h>
#include <complex>
#include <valarray>
#include <algorithm>  // For std::max_element
#include <fstream>
#include <iomanip>

#include <mpi.h>
#include <fftw3-mpi.h>

using namespace std;



// u2geq_hat(): function that uses the velocity components, density, lattice weights and Knudsen number
//              to compute the particle velocity distribution g_eq in Fourier space. Using dealiasing for 
//              non-linear terms.
// Inputs:
//       - valarray<complex<double>>& geqh: array of complex<double> to be overwritten with the output of the function
void u2geq_hat(valarray<complex<double>>& geqh) {

    // Coefficients with the speed of sound constant
    double C2 = 1.0 / (Constants::c_s* Constants::c_s);
    double C4 = 0.5 * C2 * C2;

    // Transform velocity to physical space
    FFTHandler::execute_bkwd(Temp::u1h, Temp::u1);
    FFTHandler::execute_bkwd(Temp::u2h, Temp::u2);

    // Pre-compute products in physical space
    for (int i = 0; i < 2*Constants::local_alloc_ps; i++) {
        Temp::u11[i]  = Temp::u1[i] * Temp::u1[i];
        Temp::u12[i]  = Temp::u1[i] * Temp::u2[i];
        Temp::u22[i]  = Temp::u2[i] * Temp::u2[i];
        Temp::uabs[i] = (C2/2.0) * (Temp::u22[i] + Temp::u11[i]);
    }

    // Transform the products to Fourier space
    FFTHandler::execute_fwd(Temp::u11, Temp::u11h);
    FFTHandler::execute_fwd(Temp::u22, Temp::u22h);
    FFTHandler::execute_fwd(Temp::u12, Temp::u12h);
    FFTHandler::execute_fwd(Temp::uabs, Temp::uabsh);

    // Dealias non-linear terms
    for (int i = 0; i < Constants::local_alloc_fs; i++) {
        Temp::u11h[i]  = Constants::epsilon * Temp::u11h[i]  * Constants::dealias[i];
        Temp::u22h[i]  = Constants::epsilon * Temp::u22h[i]  * Constants::dealias[i];
        Temp::u12h[i]  = Constants::epsilon * Temp::u12h[i]  * Constants::dealias[i];
        Temp::uabsh[i] = Constants::epsilon * Temp::uabsh[i] * Constants::dealias[i];
        Temp::rhoh[i]  = Temp::rhoh[i] * Constants::dealias[i];
    }

    // Compute geqh from velocity and density fields using the weights
    int offset = Constants::local_alloc_fs;
    for (int i = 0; i < Constants::local_alloc_fs; i++) {
        geqh[i + 0*offset] = (Temp::rhoh[i] + C2*Temp::u1h[i]                    - Temp::uabsh[i] + C4 * Temp::u11h[i]                                      ) * Constants::weights[0];
        geqh[i + 1*offset] = (Temp::rhoh[i] + C2*Temp::u2h[i]                    - Temp::uabsh[i] + C4 * Temp::u22h[i]                                      ) * Constants::weights[1];
        geqh[i + 2*offset] = (Temp::rhoh[i] - C2*Temp::u1h[i]                    - Temp::uabsh[i] + C4 * Temp::u11h[i]                                      ) * Constants::weights[2];
        geqh[i + 3*offset] = (Temp::rhoh[i] - C2*Temp::u2h[i]                    - Temp::uabsh[i] + C4 * Temp::u22h[i]                                      ) * Constants::weights[3];

        geqh[i + 4*offset] = (Temp::rhoh[i] + C2*( Temp::u1h[i] + Temp::u2h[i])  - Temp::uabsh[i] + C4*2*Temp::u12h[i] + C4*Temp::u11h[i] + C4*Temp::u22h[i]) * Constants::weights[4];
        geqh[i + 5*offset] = (Temp::rhoh[i] + C2*(-Temp::u1h[i] + Temp::u2h[i])  - Temp::uabsh[i] - C4*2*Temp::u12h[i] + C4*Temp::u11h[i] + C4*Temp::u22h[i]) * Constants::weights[5];
        geqh[i + 6*offset] = (Temp::rhoh[i] + C2*(-Temp::u1h[i] - Temp::u2h[i])  - Temp::uabsh[i] + C4*2*Temp::u12h[i] + C4*Temp::u11h[i] + C4*Temp::u22h[i]) * Constants::weights[6];
        geqh[i + 7*offset] = (Temp::rhoh[i] + C2*( Temp::u1h[i] - Temp::u2h[i])  - Temp::uabsh[i] - C4*2*Temp::u12h[i] + C4*Temp::u11h[i] + C4*Temp::u22h[i]) * Constants::weights[7];

        geqh[i + 8*offset] = (Temp::rhoh[i]                                      - Temp::uabsh[i]                                                           ) * Constants::weights[8];
    }
}


// calc_rho(): Function to calculate the density field from the probability distribution functions
void calc_rho() {
    Temp::rho *= 0;
    for (int m = 0; m < Constants::lattice_number; m++) {
        for (int i = 0; i < Constants::local_N_ps; i++) {
            for (int j = 0; j < Constants::N; j++) {
                int index = i * 2*(Constants::N/2+1) + j;
                Temp::rho[index] += Temp::g[index + 2*Constants::local_alloc_ps * m];
            }
        }
    }
}




// rhou1u2(): function that computes the density field and reconstructs the velocity field in Fourier space from ghat
// Inputs:
//       - valarray<complex<double>>& ghat: Particle velocity probability distribution
void rhou1u2(const valarray<complex<double>>& ghat) {
    // Compute IFFT
    for (int i = 0; i < Constants::lattice_number; i++) {
        FFTHandler::execute_bkwd_index(ghat, Temp::g, i);
    }
    
    // Summing g to get density, stored in Temp::rho
    calc_rho();
    FFTHandler::execute_fwd(Temp::rho, Temp::rhoh);


    // NS velocity components
    int offset = Constants::local_alloc_fs;
    for (size_t i = 0; i < Constants::local_alloc_fs; i++) {
        Temp::u1h[i] = ghat[i + 0 * offset] - ghat[i + 2 * offset] + ghat[i + 4 * offset] 
                     - ghat[i + 5 * offset] - ghat[i + 6 * offset] + ghat[i + 7 * offset];
        Temp::u2h[i] = ghat[i + 1 * offset] - ghat[i + 3 * offset] + ghat[i + 4 * offset] 
                     + ghat[i + 5 * offset] - ghat[i + 6 * offset] - ghat[i + 7 * offset];
    }

    // Leray projection for divergence-free solution
    for (int i = 0; i < Constants::local_alloc_fs; i++) {
        Temp::laP_hat[i] = Temp::u1h[i] * Constants::Kx[i] / Constants::Pois_hat[i] + Temp::u2h[i] * Constants::Ky[i] / Constants::Pois_hat[i];
        Temp::u1h[i] -= Constants::Kx[i] * Temp::laP_hat[i];
        Temp::u2h[i] -= Constants::Ky[i] * Temp::laP_hat[i];
    }
}


// RHS(): Function to compute the right-hand-side in the RK4 timestepping formulation
// Inputs:
//      - valarray<complex<double>>& ghat: Particle velocity probability distribution
//      - valarray<complex<double>>& kn: The steps in the RK4 formulation, this variable is to be overwritten with the output of the function
void RHS(const valarray<complex<double>>& ghat, valarray<complex<double>>& kn) {
    // Computes u1h, u2h, rhoh
    rhou1u2(ghat);

    // Computes geqh
    u2geq_hat(Temp::geqh);

    // Computes RHS and overwrite result variable
    for (int i = 0; i < Constants::local_alloc_fs * Constants::lattice_number; i++) {
        kn[i] = Constants::dt * (Constants::FTh[i] * ghat[i] + Temp::geqh[i] / (Constants::epsilon * Constants::epsilon * Constants::nu));
    }
}


// RK4stepping(): Function to step forwards with the RK4 method
// Inputs:
//      - valarray<complex<double>>& ghat: Particle velocity probability distribution, to be updated with the result of the stepping
void RK4stepping(valarray<complex<double>>& ghat) {
    // k1
    RHS(ghat, Temp::k1);

    // k2
    RHS(ghat + 0.5*Temp::k1, Temp::k2);

    // k3
    RHS(ghat + 0.5*Temp::k2, Temp::k3);

    // k4
    RHS(ghat + Temp::k3, Temp::k4);

    // Update ghat
    for (int i = 0; i < Constants::local_alloc_fs * Constants::lattice_number; i++) {
        ghat[i] += (Temp::k1[i] + 2.0*Temp::k2[i] + 2.0*Temp::k3[i] + Temp::k4[i]) / 6.0;
    }
}


// set_initial_ghat_from_closed_form(): Function to set the initial condition of ghat from the vorticity inital condition defined in "init_variables.cpp"
// Inputs:
//      - valarray<complex<double>>& ghat: Particle velocity probability distribution, to be overwritten with the initial condition of ghat
void set_initial_ghat_from_closed_form(valarray<complex<double>>& ghat) {
    // FFT of initial vorticity
    valarray<complex<double>> w_0hat(Constants::local_alloc_fs);
    FFTHandler::execute_fwd(Constants::w_0, w_0hat);


    // Compute velocity in x and y from the vorticity
    Temp::u1h = -(Constants::Ky / Constants::Pois_hat) * w_0hat;
    Temp::u2h =  (Constants::Kx / Constants::Pois_hat) * w_0hat;

    // Initial density 
    valarray<double> rho(1.0 / (Constants::N * Constants::N), 2 * Constants::local_alloc_ps);
    FFTHandler::execute_fwd(rho, Temp::rhoh);

    //Compute geq (used as initial condition for Boltzmann)
    u2geq_hat(ghat);
}


// set_initial_ghat_from_file(): Function to set the initial condition of ghat from the vorticity inital condition defined in file
// Inputs:
//      - valarray<complex<double>>& ghat: Particle velocity probability distribution, to be overwritten with the initial condition of ghat
void set_initial_ghat_from_file(valarray<complex<double>>& ghat) {
    valarray<double> w0_local(2*Constants::local_alloc_ps);
    valarray<double> w0_global(Constants::N*Constants::N);

    // Opening file
    ifstream infile(Constants::init_file_path);
    if (!infile) std::cerr << "Failed to open file\n";

    // Reading whole file
    for (size_t i = 0; i < Constants::N*Constants::N; ++i) {
        if (!(infile >> w0_global[i])) {
            std::cerr << "Error reading value at index " << i << "\n";
        }
    }

    // Splitting file so that every processor only takes one part
    for (int i = 0; i < Constants::local_N_ps; i++) {
        for (int j = 0; j < Constants::N; j++) {
            w0_local[i * 2*(Constants::N/2+1) + j] = w0_global[((i + Constants::local_start_ps) * Constants::N) + j];
        }
    }

    // FFT of initial vorticity
    valarray<complex<double>> w0_local_hat(Constants::local_alloc_fs);
    FFTHandler::execute_fwd(w0_local, w0_local_hat);


    // Compute velocity in x and y from the vorticity
    Temp::u1h = -(Constants::Ky / Constants::Pois_hat) * w0_local_hat;
    Temp::u2h =  (Constants::Kx / Constants::Pois_hat) * w0_local_hat;

    // Initial density 
    valarray<double> rho(1.0 / (Constants::N * Constants::N), 2 * Constants::local_alloc_ps);
    FFTHandler::execute_fwd(rho, Temp::rhoh);

    //Compute geq (used as initial condition for Boltzmann)
    u2geq_hat(ghat);
}



// save_vort_to_file(): Function to save the vorticity field to file
// Inputs:
//      - valarray<double>& w: vorticity field array
//      - int ti: current time index
void save_vort_to_file(valarray<double>& w, int ti) {
    // Filepath of result
    ostringstream oss;
    oss << Constants::result_file_path << Constants::init_condition << "_eps" << Constants::epsilon << "_N" << Constants::N << "_dt" << Constants::dt << "_T" << Constants::T1 << "_nu" << Constants::nu << "_Nsave" << Constants::Nsave << "_ti" << ti << ".txt";
    ofstream outFile(oss.str());

    // Write file with full precision
    outFile << std::fixed << std::setprecision(16);

    for (const auto& val : w) {
        outFile << val << " ";
    }
    outFile << std::endl;


    outFile.close();
}


// save_err_to_file(): Function to save the relative error field to file
// Inputs:
//      - valarray<double>& w: vorticity field array
//      - double t: current time
//      - int ti: current time index
void save_err_to_file(valarray<double>& w, double t, int ti) {
    if (Constants::init_condition == "ptg") return;

    valarray<double> ref_w(Constants::N*Constants::N);
    valarray<double> w_diff(Constants::N*Constants::N);

    // Computing reference solution and difference to simulation approximation
    for (int i = 0; i < Constants::N; i++) {
        for (int j = 0; j < Constants::N; j++) {
            int index = i * 2*(Constants::N/2+1) + j;
            int index_filtered = i * Constants::N + j;

            double x = j * Constants::dx;
            double y = (Constants::local_start_ps + i ) * Constants::dy;

            ref_w[index_filtered] = 10 * sin(2*M_PI*2*x) * sin(2*M_PI*2*y) * exp(-4*M_PI*M_PI * (4+4) * Constants::c_s * Constants::c_s * Constants::nu * t);
            w_diff[index_filtered] = abs(w[index_filtered] - ref_w[index_filtered]);
        }
    }

    // Computing relative error
    valarray<double> prod_diff = abs(w-ref_w)*abs(w-ref_w);
    valarray<double> ref_diff  = ref_w*ref_w;

    double E_diff = sqrt(prod_diff.sum() * Constants::dx * Constants::dy);
    double E_ref  = sqrt(ref_diff.sum()  * Constants::dx * Constants::dy);

    // Writing error to file
    string file_name = Constants::result_file_path + "/error" + Constants::error_file_suffix + ".txt";
    ofstream out_error(file_name, std::ios::app);

    out_error << E_diff/E_ref << "\n";
    out_error.close();
}





// save_to_file_routine(): Routine to save chosen info to file
// Inputs:
//      - valarray<complex<double>>& ghat: Particle velocity probability distribution
//      - double t: current time
//      - int ti: current time index
void save_to_file_routine(valarray<complex<double>>& ghat, double t, int ti) {
    // Computing velocity from ghat
    rhou1u2(ghat);

    valarray<double> w_local(2 * Constants::local_alloc_ps);
    valarray<double> w_local_filtered(Constants::local_N_ps * Constants::N);

    // Transforming from velocity to vorticity
    valarray<complex<double>> w_local_hat(Constants::local_alloc_fs);
    for (int i = 0; i < Constants::local_alloc_fs; i++) {
        w_local_hat[i] = Constants::Kx[i] * Temp::u2h[i] - Constants::Ky[i] * Temp::u1h[i];
    }

    // Transforming to physical space
    FFTHandler::execute_bkwd(w_local_hat, w_local);

    // Filter extra components
    for (int i = 0; i < Constants::local_N_ps; i++) {
        for (int j = 0; j < Constants::N; j++) {
            int index = i * 2*(Constants::N/2+1) + j;
            int index_filtered = i * Constants::N + j;
            w_local_filtered[index_filtered] = w_local[index];
        }
    }

    // Gather all the local vorticity batches to rank 0
    valarray<double> w_global(Constants::N * Constants::N);
    MPI_Gather(&w_local_filtered[0], Constants::local_N_ps * Constants::N, MPI_DOUBLE, &w_global[0], Constants::local_N_ps * Constants::N, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    // Saving vorticity and/or error depending on user choice in "init_variable.cpp"
    if(Constants::rank == 0) {
        if (Constants::saving_sol == "we") {
            save_vort_to_file(w_global, ti);
            save_err_to_file(w_global, t, ti);
        } else if (Constants::saving_sol == "w") {
            save_vort_to_file(w_global, ti);
        } else if (Constants::saving_sol == "e") {
            save_err_to_file(w_global, t, ti);
        }
    }
}
