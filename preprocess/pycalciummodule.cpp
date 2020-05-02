//
// Created by Albert Qu on 4/28/20.
//
#define PY_SSIZE_T_CLEAN
//#include <Python.h>
#include "te-datainit.h"
using namespace std;

#define IOSTREAMH std::ostream& output
#define IOSTREAMC output
#define IOSTREAMV output
#define IOSTREAMENDL std::endl
#define GSL_RANDOM_NUMBER_GENERATOR gsl_rng_ranlxs2

//void apply_light_scattering_to_time_series(double** data, unsigned int size, long samples, std::string YAMLfilename, double sigma_scatter, double amplitude_scatter, IOSTREAMH);


int get_size_file(std::string filename) {
    std::string s = "scott>=tiger>=mushroom";
    std::string delimiter = ">=";

    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        std::cout << token << std::endl;
        s.erase(0, pos + delimiter.length());
    }
    std::cout << s << std::endl;
    size_t last = 0; size_t next = 0; while ((next = s.find(delimiter, last)) != string::npos) { cout << s.substr(last, next-last) << endl; last = next + 1; } cout << s.substr(last) << endl;
    return 0;
}


string pathAppend(const string& p1, const string& p2) {

    char sep = '/';
    string tmp = p1;

#ifdef _WIN32
    sep = '\\';
#endif

    if (p1[p1.length()] != sep) { // Need to add a
        tmp += sep;                // path separator
        return(tmp + p2);
    }
    else
        return(p1 + p2);
}


int main(int argc, char** argv)
{
    // C code for generating calcium fluorescence series
    // time file; index file; # neurons; tauF, number of calcium frames, FluorescenceModel, std of noise, saturation
    // NECESSARY PARAMETERS FOR MODEL
    std::string inputfile_spiketimes; // time file
    std::string inputfile_spikeindices; // index file
    std::string outputfile_results_name; //outfile
    unsigned int tauImg = 100; //ms;
    std::string fluorescence_model = "Leogang";
    double std_noise = 0.03;
    double fluorescence_saturation = 300.;
    double cutoff = 1000.;
    double DeltaCalciumOnAP = 50.; // uM
    double tauCa = 400.; // ms
    gsl_rng* GSLrandom;
    gsl_rng_env_setup();
    GSLrandom = gsl_rng_alloc(GSL_RANDOM_NUMBER_GENERATOR);
    gsl_rng_set(GSLrandom, 288);
    unsigned int size;
    long samples;
    if( argc == 6 ) {
        std::string in1(argv[1]);
        std::string in2(argv[2]);
        std::string out1(argv[3]);
        inputfile_spiketimes = in1;
        inputfile_spikeindices = in2;
        size = atoi(argv[4]); // # of neurons
        samples = atoi(argv[5]); // sample
    }
    else {
        cout << "Usage: ./cppfile InputSpikeTime InputSpikeIndex OutputFile size samples\n";
        return 1;
    }

    // use python imp source for similar file loading
    double** xdatadouble = generate_time_series_from_spike_data(inputfile_spiketimes, inputfile_spikeindices, size,
    tauImg, samples, fluorescence_model, std_noise, fluorescence_saturation, cutoff, DeltaCalciumOnAP, tauCa, GSLrandom,
    std::cout);
    write_result(xdatadouble, size, outputfile_results_name, std::cout, MX);
    if(xdatadouble!=NULL) free_time_series_memory(xdatadouble, size);
}