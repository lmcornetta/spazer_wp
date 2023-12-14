#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <complex>
#include <vector>
#include <cmath>
#include "fftw3.h"

#define MAX_NPTS (int) 1024
#define XMAX 10.0
#define XMIN -10.0
#define PI 3.1415926535

using namespace std;

/* Computes de 1-D Fast Fourier transform for an array of length MAX_NPTS */
void fft(fftw_complex * in, fftw_complex * out){
    /* Create a Discrete Fourier transform plan */
    fftw_plan plan = fftw_plan_dft_1d(MAX_NPTS, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    /* Execute the plan */
    fftw_execute(plan);
    /* Do some cleaning */
    fftw_destroy_plan(plan);
    fftw_cleanup();
}

/* Computes de 1-D Fast Fourier transform for an array of generic length npts */
void fft(fftw_complex * in, fftw_complex * out, int npts){
    /* Create a Discrete Fourier transform plan */
    fftw_plan plan = fftw_plan_dft_1d(npts, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    /* Execute the plan */
    fftw_execute(plan);
    /* Do some cleaning */
    fftw_destroy_plan(plan);
    fftw_cleanup();
}

/* Computes de 1-D inverse Fast Fourier transform for an array of length MAX_NPTS */
void ifft(fftw_complex * in, fftw_complex * out){
    /* Create a "I" Discrete Fourier transform plan */
    fftw_plan plan = fftw_plan_dft_1d(MAX_NPTS, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    /* Execute the plan */
    fftw_execute(plan);
    /* Do some cleaning */
    fftw_destroy_plan(plan);
    fftw_cleanup();
    /* Scale the output to obtain the exact inverse */
    for (int i = 0; i < MAX_NPTS; i++){
        out[i][0] /= MAX_NPTS;
        out[i][1] /= MAX_NPTS;
    }
}

/* Computes de 1-D inverse Fast Fourier transform for an array of generic length npts */
void ifft(fftw_complex * in, fftw_complex * out, int npts){
    /* Create a "I" Discrete Fourier transform plan */
    fftw_plan plan = fftw_plan_dft_1d(npts, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    /* Execute the plan */
    fftw_execute(plan);
    /* Do some cleaning */
    fftw_destroy_plan(plan);
    fftw_cleanup();
    /* Scale the output to obtain the exact inverse */
    for (int i = 0; i < npts; i++){
        out[i][0] /= npts;
        out[i][1] /= npts;
    }
}

/* Print the complex numbers (length MAX_NPTS) */
void displayComplex(fftw_complex * a){
    for (int i = 0; i < MAX_NPTS; i++){
        if (a[i][1] < 0)
            cout << a[i][0] << " - " << abs(a[i][1]) << "i" << endl;
        else
            cout << a[i][0] << " + " << abs(a[i][1]) << "i" << endl;
    }
}

/* Print the complex numbers (length npts) */
void displayComplex(fftw_complex * a, int npts){
    for (int i = 0; i < npts; i++){
        if (a[i][1] < 0) {
            cout << a[i][0] << " - " << abs(a[i][1]) << "i" << endl;
        }
        else {
            cout << a[i][0] << " + " << abs(a[i][1]) << "i" << endl;
        }
    }
}

/* Print grid (double values) */
void displayGrid(double * a, int npts){
    for (int i = 0; i < npts; i++){
        cout << i << " "  << a[i] << endl;
    }
}

/* Print the absolute values (length MAX_NPTS) */
void displayAbs(fftw_complex * a){
    for (int i = 0; i < MAX_NPTS; i++)
        cout << a[i][0]*a[i][0] + a[i][1]*a[i][1] << endl;
}

/* Print the absolute values (length npts) */
void displayAbs(fftw_complex * a, int npts){
    for (int i = 0; i < npts; i++)
        cout << a[i][0]*a[i][0] + a[i][1]*a[i][1] << endl;
}

/* Print Amplitude and Phase (length MAX_NPTS) */
void displayAmpPhase(fftw_complex * a){
    double phase;
    for (int i = 0; i < MAX_NPTS; i++){
        if (a[i][0] == 0.0)
            phase = 0.5*PI*a[i][1]/abs(a[i][1]);
        else 
            phase = atan(a[i][1]/a[i][0]);
        cout << sqrt(a[i][0]*a[i][0] + a[i][1]*a[i][1]) << "\t" << phase << endl;
    }
}

/* Print the Amplitude and Phase (length npts) */
void displayAmpPhase(fftw_complex * a, int npts){
    double phase;
    for (int i = 0; i < npts; i++){
        if (a[i][0] == 0.0)
            phase = 0.5*PI*a[i][1]/abs(a[i][1]);
        else 
            phase = atan(a[i][1]/a[i][0]);
        cout << sqrt(a[i][0]*a[i][0] + a[i][1]*a[i][1]) << "\t" << phase << endl;
    }        
}

/* Proint (export) to file */
void printToFile(fftw_complex * psi, int npts, string filename){
    ofstream myfile;
    myfile.open(filename + ".dat");
    for(int i = 0; i < npts - 1; i++)
        myfile << psi[i][0] << "\t" << psi[i][1] << "\t" << psi[i][0]*psi[i][0] + psi[i][1]*psi[i][1] << endl;
    myfile << psi[npts - 1][0] << "\t" << psi[npts - 1][1] << "\t" << psi[npts - 1][0]*psi[npts - 1][0] + psi[npts - 1][1]*psi[npts - 1][1];
    myfile.close();
}

class Grid{
public:
    double * x;
    double * k;
    double dx, dk;
    Grid() {
        x = (double*) malloc(sizeof(double) * MAX_NPTS);
        k = (double*) malloc(sizeof(double) * MAX_NPTS);     
        dx = (XMAX - XMIN)/MAX_NPTS;
        dk = 2*PI/(MAX_NPTS*dx);
        for (int i = 0; i < MAX_NPTS; i++){
            x[i] = XMIN + i*dx;
            if (i < MAX_NPTS/2)
                k[i] = i*dk;
            else
                k[i] = -(MAX_NPTS - i)*dk;
        }
    }

    Grid(double xmin, double xmax, int npts) {
        x = (double*) malloc(sizeof(double) * npts); 
        k = (double*) malloc(sizeof(double) * npts); 
        dx = (xmax - xmin)/npts;
        dk = 2*PI/(npts*dx);
        for (int i = 0; i < npts; i++){
            x[i] = xmin + i*dx;
            if (i < npts/2)
                k[i] = i*dk;
            else
                k[i] = -(npts - i)*dk;
        }
    }

    ~Grid() {
        free(x);
        free(k);
    }
};

double harmonic(double x){
    return 0.5*x*x;
}

class Potential{
public:
    fftw_complex * V;
    Potential() {
        V = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * MAX_NPTS);
        double dx = (XMAX - XMIN)/MAX_NPTS;    
        for (int i = 0; i < MAX_NPTS; i++){
            V[i][0] = harmonic(XMIN + i*dx);
            V[i][1] = 0.0;
        }
    }

    Potential(double xmin, double xmax, int npts) {
        V = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);        
        double dx = (xmax - xmin)/npts;
        for (int i = 0; i < npts; i++){
            V[i][0] = harmonic(xmin + i*dx);
            V[i][1] = 0.0;
        }
    }

    Potential(double xmin, double xmax, int npts, fftw_complex * pot) {
        V = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts); 
        for (int i = 0; i < npts; i++){
            V[i][0] = pot[i][0];
            V[i][1] = pot[i][1];
        }
    }

    ~Potential() {
        fftw_free(V);
    }
};

class EvolutionOP {/* Check cos and sin functions (Imag and Real parts)*/
public:
    fftw_complex * kinetic_evolution;
    fftw_complex * kinetic_half_evolution;
    fftw_complex * potential_evolution;
    Potential * pot;
    Grid * grid;
    double dt;
    int npts;

    /* Default constructors for if no potential list is given */
    EvolutionOP(){
        kinetic_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * MAX_NPTS);
        kinetic_half_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * MAX_NPTS);
        potential_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * MAX_NPTS);
        dt = 0.01;
        pot = new Potential();
        grid = new Grid();
        for (int i = 0; i < MAX_NPTS; i++){
            kinetic_evolution[i][0] = cos(-0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_evolution[i][1] = sin(-0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_half_evolution[i][0] = cos(-0.5*0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_half_evolution[i][1] = sin(-0.5*0.5*grid->k[i]*grid->k[i]*dt);
            potential_evolution[i][0] = cos(-pot->V[i][0]*dt)*exp(-0.5*pot->V[i][1]*dt);
            potential_evolution[i][1] = sin(-pot->V[i][0]*dt)*exp(-0.5*pot->V[i][1]*dt);
        }
    }

    EvolutionOP(double dt_){
        kinetic_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * MAX_NPTS);
        kinetic_half_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * MAX_NPTS);
        potential_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * MAX_NPTS);
        dt = dt_;
        pot = new Potential();
        grid = new Grid();
        for (int i = 0; i < MAX_NPTS; i++){
            kinetic_evolution[i][0] = cos(-0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_evolution[i][1] = sin(-0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_half_evolution[i][0] = cos(-0.5*0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_half_evolution[i][1] = sin(-0.5*0.5*grid->k[i]*grid->k[i]*dt);
            potential_evolution[i][0] = cos(-pot->V[i][0]*dt)*exp(-0.5*pot->V[i][1]*dt);
            potential_evolution[i][1] = sin(-pot->V[i][0]*dt)*exp(-0.5*pot->V[i][1]*dt);
        }
    }

    EvolutionOP(double xmin, double xmax, int npts){
        kinetic_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        kinetic_half_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        potential_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        dt = 0.01;
        grid = new Grid(xmin, xmax, npts);
        pot = new Potential(xmin, xmax, npts);
        for (int i = 0; i < npts; i++){
            kinetic_evolution[i][0] = cos(-0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_evolution[i][1] = sin(-0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_half_evolution[i][0] = cos(-0.5*0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_half_evolution[i][1] = sin(-0.5*0.5*grid->k[i]*grid->k[i]*dt);
            potential_evolution[i][0] = cos(-pot->V[i][0]*dt)*exp(-0.5*pot->V[i][1]*dt);
            potential_evolution[i][1] = sin(-pot->V[i][0]*dt)*exp(-0.5*pot->V[i][1]*dt);
        }
    }

    EvolutionOP(double dt_, double xmin, double xmax, int npts){
        kinetic_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        kinetic_half_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        potential_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        dt = dt_;
        grid = new Grid(xmin, xmax, npts);
        pot = new Potential(xmin, xmax, npts);
        for (int i = 0; i < npts; i++){
            kinetic_evolution[i][0] = cos(-0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_evolution[i][1] = sin(-0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_half_evolution[i][0] = cos(-0.5*0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_half_evolution[i][1] = sin(-0.5*0.5*grid->k[i]*grid->k[i]*dt);
            potential_evolution[i][0] = cos(-pot->V[i][0]*dt)*exp(-0.5*pot->V[i][1]*dt);
            potential_evolution[i][1] = sin(-pot->V[i][0]*dt)*exp(-0.5*pot->V[i][1]*dt);
        }
    }
    
    /* Constructors for a user given potential list */
    EvolutionOP(double xmin, double xmax, int npts_, fftw_complex * pot_){
        dt = 0.01;
        npts = npts_;
        kinetic_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        kinetic_half_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        potential_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        grid = new Grid(xmin, xmax, npts);
        pot = new Potential(xmin, xmax, npts, pot_);
        for (int i = 0; i < npts; i++){
            kinetic_evolution[i][0] = cos(-0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_evolution[i][1] = sin(-0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_half_evolution[i][0] = cos(-0.5*0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_half_evolution[i][1] = sin(-0.5*0.5*grid->k[i]*grid->k[i]*dt);
            potential_evolution[i][0] = cos(-pot->V[i][0]*dt)*exp(-0.5*pot->V[i][1]*dt);
            potential_evolution[i][1] = sin(-pot->V[i][0]*dt)*exp(-0.5*pot->V[i][1]*dt);
        }
    }

    EvolutionOP(double dt_, double xmin, double xmax, int npts_, fftw_complex * pot_){
        dt = dt_;
        npts = npts_;
        kinetic_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        kinetic_half_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        potential_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        grid = new Grid(xmin, xmax, npts);
        pot = new Potential(xmin, xmax, npts, pot_);
        for (int i = 0; i < npts; i++){
            kinetic_evolution[i][0] = cos(-0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_evolution[i][1] = sin(-0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_half_evolution[i][0] = cos(-0.5*0.5*grid->k[i]*grid->k[i]*dt);
            kinetic_half_evolution[i][1] = sin(-0.5*0.5*grid->k[i]*grid->k[i]*dt);
            potential_evolution[i][0] = cos(-pot->V[i][0]*dt)*exp(-0.5*pot->V[i][1]*dt);
            potential_evolution[i][1] = sin(-pot->V[i][0]*dt)*exp(-0.5*pot->V[i][1]*dt);
        }
    }

    EvolutionOP(double dt_, double xmin, double xmax, int npts_, double mass, fftw_complex * pot_){
        dt = dt_;
        npts = npts_;
        kinetic_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        kinetic_half_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        potential_evolution = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        grid = new Grid(xmin, xmax, npts);
        pot = new Potential(xmin, xmax, npts, pot_);
        for (int i = 0; i < npts; i++){
            kinetic_evolution[i][0] = cos(-(0.5/mass)*grid->k[i]*grid->k[i]*dt);
            kinetic_evolution[i][1] = sin(-(0.5/mass)*grid->k[i]*grid->k[i]*dt);
            kinetic_half_evolution[i][0] = cos(-0.5*(0.5/mass)*grid->k[i]*grid->k[i]*dt);
            kinetic_half_evolution[i][1] = sin(-0.5*(0.5/mass)*grid->k[i]*grid->k[i]*dt);
            potential_evolution[i][0] = cos(-pot->V[i][0]*dt)*exp(-0.5*pot->V[i][1]*dt);
            potential_evolution[i][1] = sin(-pot->V[i][0]*dt)*exp(-0.5*pot->V[i][1]*dt);
        }
    }


    /* Default destructor */
    ~EvolutionOP(){ 
        fftw_free(kinetic_evolution);
        fftw_free(kinetic_half_evolution);
        fftw_free(potential_evolution);

        delete pot; 
        delete grid;
        }

};

class Psi {
public:
    /* Public attributes */
    fftw_complex *psi, *psi_k, *psi0;
    EvolutionOP * evol;
    int npts;
    double dt, t, mass;

    Psi(){/* Generic constructor*/
        npts = MAX_NPTS;
        psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        psi_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        psi0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        dt = 0.01;
        t = 0.0;

        evol = new EvolutionOP();
        
        double s = 0.0;
        for (int i = 0; i < npts; i++){
            psi[i][0] = exp(-(evol->grid->x[i] - 2.4)*(evol->grid->x[i] - 2.4)/2)/sqrt(2*PI);
            psi[i][1] = 0.0;
            psi0[i][0] = exp(-(evol->grid->x[i] - 2.4)*(evol->grid->x[i] - 2.4)/2)/sqrt(2*PI);
            psi0[i][1] = 0.0;
            s += psi0[i][0]*psi0[i][0] + psi0[i][1]*psi0[i][1];
        }

        /* Normalizing initial wave packet */
        for (int i = 0; i < npts; i++){
            psi0[i][0] = psi0[i][0]/sqrt(s*evol->grid->dx);
            psi0[i][1] = psi0[i][1]/sqrt(s*evol->grid->dx);
            psi[i][0] = psi[i][0]/sqrt(s*evol->grid->dx);
            psi[i][1] = psi[i][1]/sqrt(s*evol->grid->dx);
        }
    }

    Psi(double dt_){/* Generic constructor */
        npts = MAX_NPTS;
        psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        psi_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        psi0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        t = 0.0;
        dt = dt_;

        evol = new EvolutionOP(dt);
        
        double s = 0.0;
        for (int i = 0; i < npts; i++){
            psi[i][0] = exp(-(evol->grid->x[i] - 2.4)*(evol->grid->x[i] - 2.4)/2.0)/sqrt(sqrt(PI));
            psi[i][1] = 0.0;
            psi0[i][0] = exp(-(evol->grid->x[i] - 2.4)*(evol->grid->x[i] - 2.4)/2.0)/sqrt(sqrt(PI));
            psi0[i][1] = 0.0;
            s += psi0[i][0]*psi0[i][0] + psi0[i][1]*psi0[i][1];
        }

        /* Normalizing initial wave packet */
        for (int i = 0; i < npts; i++){
            psi0[i][0] = psi0[i][0]/sqrt(s*evol->grid->dx);
            psi0[i][1] = psi0[i][1]/sqrt(s*evol->grid->dx);
            psi[i][0] = psi[i][0]/sqrt(s*evol->grid->dx);
            psi[i][1] = psi[i][1]/sqrt(s*evol->grid->dx);
        }

    }

    Psi(double xmin, double xmax, int npts_, fftw_complex * psi_){/* Constructor */        
        npts = npts_;
        psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        psi_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        psi0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        t = 0.0;

        evol = new EvolutionOP(xmin, xmax, npts);
        
        double s = 0.0;
        for (int i = 0; i < npts; i++){
            psi[i][0] = psi_[i][0];
            psi[i][1] = psi_[i][1];
            psi0[i][0] = psi_[i][0];
            psi0[i][1] = psi_[i][1];
            s += psi0[i][0]*psi0[i][0] + psi0[i][1]*psi0[i][1];
        }

        /* Normalizing initial wave packet */
        for (int i = 0; i < npts; i++){
            psi0[i][0] = psi0[i][0]/sqrt(s*evol->grid->dx);
            psi0[i][1] = psi0[i][1]/sqrt(s*evol->grid->dx);
            psi[i][0] = psi[i][0]/sqrt(s*evol->grid->dx);
            psi[i][1] = psi[i][1]/sqrt(s*evol->grid->dx);
        }
    }

    Psi(double dt_, double xmin, double xmax, int npts_, fftw_complex * psi_){/* Constructor */
        npts = npts_; 
        psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        psi_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        psi0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        t = 0.0;
        dt = dt_;

        evol = new EvolutionOP(dt, xmin, xmax, npts);
        
        double s = 0.0;
        for (int i = 0; i < npts; i++){
            psi[i][0] = psi_[i][0];
            psi[i][1] = psi_[i][1];
            psi0[i][0] = psi_[i][0];
            psi0[i][1] = psi_[i][1];
            s += psi0[i][0]*psi0[i][0] + psi0[i][1]*psi0[i][1];
        }

        /* Normalizing initial wave packet */
        for (int i = 0; i < npts; i++){
            psi0[i][0] = psi0[i][0]/sqrt(s*evol->grid->dx);
            psi0[i][1] = psi0[i][1]/sqrt(s*evol->grid->dx);
            psi[i][0] = psi[i][0]/sqrt(s*evol->grid->dx);
            psi[i][1] = psi[i][1]/sqrt(s*evol->grid->dx);
        }
    }

    /* Psi constructors for a given potential list */
    Psi(double xmin, double xmax, int npts_, double mass_, fftw_complex * pot_, fftw_complex * psi_){/* Constructor without dt*/
        npts = npts_;
        mass = mass_; 
        psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        psi_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        psi0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        t = 0.0;

        evol = new EvolutionOP(xmin, xmax, npts, mass, pot_);
        
        double s = 0.0;
        for (int i = 0; i < npts; i++){
            psi[i][0] = psi_[i][0];
            psi[i][1] = psi_[i][1];
            psi0[i][0] = psi_[i][0];
            psi0[i][1] = psi_[i][1];
            s += psi0[i][0]*psi0[i][0] + psi0[i][1]*psi0[i][1];
        }

        /* Normalizing initial wave packet */
        for (int i = 0; i < npts; i++){
            psi0[i][0] = psi0[i][0]/sqrt(s*evol->grid->dx);
            psi0[i][1] = psi0[i][1]/sqrt(s*evol->grid->dx);
            psi[i][0] = psi[i][0]/sqrt(s*evol->grid->dx);
            psi[i][1] = psi[i][1]/sqrt(s*evol->grid->dx);
        }
    }

    Psi(double dt_, double xmin, double xmax, int npts_, double mass_, fftw_complex * pot_, fftw_complex * psi_){/* Constructor with dt */
        npts = npts_; 
        mass = mass_;
        psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        psi_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        psi0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
        t = 0.0;
        dt = dt_;
        
        evol = new EvolutionOP(dt, xmin, xmax, npts, mass, pot_);

        double s = 0.0;
        for (int i = 0; i < npts; i++){
            psi[i][0] = psi_[i][0];
            psi[i][1] = psi_[i][1];
            psi0[i][0] = psi_[i][0];
            psi0[i][1] = psi_[i][1];
            s += psi0[i][0]*psi0[i][0] + psi0[i][1]*psi0[i][1];
        }

        /* Normalizing initial wave packet */
        for (int i = 0; i < npts; i++){
            psi0[i][0] = psi0[i][0]/sqrt(s*evol->grid->dx);
            psi0[i][1] = psi0[i][1]/sqrt(s*evol->grid->dx);
            psi[i][0] = psi[i][0]/sqrt(s*evol->grid->dx);
            psi[i][1] = psi[i][1]/sqrt(s*evol->grid->dx);
        }
    }

    ~Psi() {/* Destructor */   
        fftw_free(psi);
        fftw_free(psi_k);
        fftw_free(psi0);
        delete evol; 
    }

    void normalize(){
        double s = 0.0;
        for (int i = 0; i < npts; i++)
            s += psi[i][0]*psi[i][0] + psi[i][1]*psi[i][1];
        for (int i = 0; i < npts; i++){
            psi[i][0] = psi[i][0]/sqrt(s*evol->grid->dx);
            psi[i][1] = psi[i][1]/sqrt(s*evol->grid->dx);
        }
    }

    void dump(){
        double xi;
        for (int i = 0; i < npts; i++){
            xi = this->evol->grid->x[i];
            if (xi <= 0){
                psi[i][0] = psi[i][0]*0.0;
                psi[i][1] = psi[i][1]*0.0;
            }
            else if (xi <= 0.5)
            {
                psi[i][0] = psi[i][0]*xi*xi;
                psi[i][1] = psi[i][1]*xi*xi;
            }
            
        }
    }

    complex<double> correlation(double gamma, double freq){
        double creal = 0.0;
        double cimag = 0.0;
        for (int i = 0; i < npts; i++){
            creal += psi0[i][0]*psi[i][0] + psi0[i][1]*psi[i][1];
            cimag += psi0[i][0]*psi[i][1] - psi0[i][1]*psi[i][0];
        }
    
        complex<double> c;
        c.real(exp(-gamma*t)*(creal*evol->grid->dx*cos(freq*t) - cimag*evol->grid->dx*sin(freq*t)));
        c.imag(exp(-gamma*t)*(cimag*evol->grid->dx*cos(freq*t) + creal*evol->grid->dx*sin(freq*t)));
        return c;
    }

    void time_evol(){

        fft(psi, psi_k, npts);
        for (int i = 0; i < npts; i++){
            psi_k[i][0] = evol->kinetic_half_evolution[i][0]*psi_k[i][0] - evol->kinetic_half_evolution[i][1]*psi_k[i][1];
            psi_k[i][1] = evol->kinetic_half_evolution[i][0]*psi_k[i][1] + evol->kinetic_half_evolution[i][1]*psi_k[i][0];
        }

        ifft(psi_k, psi, npts);
        for (int i = 0; i < npts; i++){
            psi[i][0] = evol->potential_evolution[i][0]*psi[i][0] - evol->potential_evolution[i][1]*psi[i][1];
            psi[i][1] = evol->potential_evolution[i][0]*psi[i][1] + evol->potential_evolution[i][1]*psi[i][0];
        }

        fft(psi, psi_k, npts);
        for (int i = 0; i < npts; i++){
            psi_k[i][0] = evol->kinetic_half_evolution[i][0]*psi_k[i][0] - evol->kinetic_half_evolution[i][1]*psi_k[i][1];
            psi_k[i][1] = evol->kinetic_half_evolution[i][0]*psi_k[i][1] + evol->kinetic_half_evolution[i][1]*psi_k[i][0];
        }

        ifft(psi_k, psi, npts);
        this->dump();
        this->normalize();
        t += dt;
    }
};

int main(){
    double xmin = -2.99;
    double xmax = 16;
    int npts = 2048;
    
    double tfinal = 1240.24; /* 30 fs */
    int ntimesteps = 262144; /* 2^18 */
    
    double dt = tfinal/ntimesteps;
    cout << 2*ntimesteps << "\t" << dt << "\t" << tfinal << endl;
    fftw_complex *corr, *spec, *pot, *psi0;

    corr = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 2*ntimesteps);
    spec = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 2*ntimesteps);
    pot = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
    psi0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);


    /* Reading potential */
    ifstream potfile("potential.dat", ios::in);
    vector<double> v;
    double num = 0.0;
    while(potfile >> num)
        v.push_back(num);
    potfile.close();

    cout << v.size() << endl;

    double wc0 = 20.1386184; /* Vertical excitation energy */
    for (int k = 0; k < npts; k++){
        pot[k][0] = v[k];
        pot[k][1] = 0.0;
    }

    /* Building initial wave packet from scratch */
    double mass = 1836.15267389*7.9995;
    double freq = 0.005078180043;

    ifstream psi0file("psi0.dat", ios::in);
    vector<double> psi0_;
    num = 0.0;
    while(psi0file >> num)
        psi0_.push_back(num);
    psi0file.close();

    cout << psi0_.size() << endl;

    for (int k = 0; k < npts; k++){
        psi0[k][0] = psi0_[k];
        psi0[k][1] = 0.0;
    }

    /* Creating Psi object */
    Psi * wf = new Psi(dt, xmin, xmax, npts, mass, pot, psi0);
    double gamma = 0.15/27.2114;

    complex<double> c = wf->correlation(gamma, freq - wc0);
    corr[ntimesteps][0] = c.real();
    corr[ntimesteps][1] = c.imag();

    int i = 0;
    string filename;
    //filename = "t_" + to_string(i);
    while(wf->t < tfinal){
        cout << i << endl;
        if (i%1000 == 0){
           filename = "t_" + to_string(i/1000);
           printToFile(wf->psi, npts, filename);
        }
        //printToFile(wf->psi, npts, filename);
       
        /* Time ellapse */
        wf->time_evol();
        i++;
        //filename = "t_" + to_string(i);

        /* Evaluate correlation function at time t */
        c = wf->correlation(gamma, freq - wc0);
        corr[ntimesteps + i][0] = c.real();
        corr[ntimesteps - i][0] = c.real();
        corr[ntimesteps + i][1] = c.imag();
        corr[ntimesteps - i][1] = -c.imag();
    }
    /* Final timestep */
    printToFile(wf->psi, npts, filename);
    c = wf->correlation(gamma, freq - wc0);
    corr[ntimesteps + i][0] = c.real();
    corr[ntimesteps - i - 1][0] = c.real();
    corr[ntimesteps + i][1] = c.imag();
    corr[ntimesteps - i - 1][1] = -c.imag();

    cout << "propagation done" << endl;
    /* Delete wave packet objetct */
    
    delete wf;
    cout << "memory free" << endl;
    
    /* Write correlation function to file */
    string correlationFile = "correlation";
    ofstream mycorrelationFile;
    mycorrelationFile.open(correlationFile + ".dat");
    for (int i = 0; i < 2*ntimesteps-1; i++){
        mycorrelationFile << i*dt - tfinal << "\t" << corr[i][0] << "\t" << corr[i][1] << "\t" << corr[i][0]*corr[i][0] + corr[i][1]*corr[i][1] << endl;
    }
    i = (2*ntimesteps - 1);
    mycorrelationFile << i*dt - tfinal << "\t" << corr[i][0] << "\t" << corr[i][1] << "\t" << corr[i][0]*corr[i][0] + corr[i][1]*corr[i][1];
    mycorrelationFile.close();

    /* Calcuate spectrum from correlation function via FFT */

    fft(corr, spec, 2*ntimesteps);
    
    /* Write spectrum function to file */
    string spectrumFile = "spectrum";
    ofstream myspectrumFile;
    myspectrumFile.open(spectrumFile + ".dat");
    for (int i = 0; i < 2*ntimesteps-1; i++){
        myspectrumFile << i*dt - tfinal << "\t" << spec[i][0] << "\t" << spec[i][1] << "\t" << spec[i][0]*spec[i][0] + spec[i][1]*spec[i][1] << endl;
    }
    i = (2*ntimesteps - 1);
    myspectrumFile << i*dt - tfinal << "\t" << spec[i][0] << "\t" << spec[i][1] << "\t" << spec[i][0]*spec[i][0] + spec[i][1]*spec[i][1];
    myspectrumFile.close();

    fftw_free(corr);
    fftw_free(spec);

    return 0;
}
