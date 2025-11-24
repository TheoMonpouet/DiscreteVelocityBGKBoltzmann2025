# Discrete-Velocity BGK Boltzmann implementation
##### Author: Théo Monpouet Ekeram
##### Date: June 2025
##### Thesis link: [Discrete-Velocity BGK Boltzmann implementation](https://kth.diva-portal.org/smash/record.jsf?dswid=-4344&pid=diva2%3A1990584&c=1&searchType=SIMPLE&language=en&query=theo+monpouet&af=%5B%5D&aq=%5B%5B%5D%5D&aq2=%5B%5B%5D%5D&aqe=%5B%5D&noOfRows=50&sortOrder=author_sort_asc&sortOrder2=title_sort_asc&onlyFullText=false&sf=all)

##### Description:
Parallel implemetation of the discrete velocity BGK Boltzmann method for the lattices D2Q7, D2Q9, D2Q13 in two dimensions and D3Q19 in three dimensions. Implemented in C++ using the Message Passing Interface ([MPI](https://www.mpich.org/static/docs/v3.1/www3/)) and Fastest Fourier Transform in the West ([FFTW](https://www.fftw.org)). In the cases where closed form reference solutions exists, the program can output the error in order to determine the rate of the hydrodynamic limit. In the case where no closed form exact solutions exist, the user can choose to save the simulated vorticity or velocity field to file, to analyse outside of the program.

# Dependencies
- Message Passing Interface ([MPI](https://www.mpich.org/static/docs/v3.1/www3/))
- Fastest Fourier Transform in the West ([FFTW](https://www.fftw.org))

# Files
One lattice per folder according to folder name. The file structure is the same for all lattices:
- `init_variables.cpp`: Initializes the struct `Constants` to include all constants needed in the program. This includes physical and numerical parameters, operators, initial conditions etc. This is also where the user input to the program is. The variables in the `Adjustable physical parameters`, `Adjustable numerical parameters`, `Initial condition`, and `Saving solution` can be changed to specify how the simulation should be run and where the result should be saved .
- `FFTHandler.cpp`: Initializes a struct `FFTHandler` with wrappers to execute the different forwards and inverse Fourier transforms.
- `temp_variables.cpp`: Initializes a struct `Temp` to store all variables that can be overwritten during the program.
- `functions.cpp`: Includes all the functions regarding the time stepping and the saving to file.
- `main.cpp`: The main program, includes setting up MPI, setting initial conditions, and the main time loop.

# How to use
Set parameters (viscosity, Knudsen number, spatial domain, time interval, spatial resolution, initial condition, file path to initial condition (optional), file path to result, which results to save, and the number of saving points). This is done in `init_variables.cpp`.

Compile `main.cpp` using optimization flags such as `-O3` for the `CC` wrapper, and run the `main.exe` file that results from the compilation. The program then runs and save the results in the path that is specified in the user-set parameters.
