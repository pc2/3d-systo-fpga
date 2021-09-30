/*
* MIT License
* 
* Copyright (c) 2021 Paolo Gorlani 
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE. 
*/

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

#ifdef FAST_EMU
  #define CL_CHANNEL_1_INTELFPGA 0
  #define CL_CHANNEL_2_INTELFPGA 0
  #define CL_CHANNEL_3_INTELFPGA 0
  #define CL_CHANNEL_4_INTELFPGA 0
  #define CL_MEM_HETEROGENEOUS_INTELFPGA 0
  #define PLATFORM_ID 0
#else
  #define PLATFORM_ID 1
  #include <CL/cl_ext_intelfpga.h>
#endif

#include <iostream>
#include <iomanip>
#include <fstream>

#include <vector>
#include <string>

#include <algorithm>
#include <limits>
#include <cmath>

#include <float_classifier.hpp>

#define DEVICE_ID 0

#define PRINT_ULP_HIST
//#define CHECK_WITH_ONES

unsigned int float_to_uint(float);
unsigned int float_ulp_distance(float, float);
int check_dims(size_t dim_i, size_t dim_j, size_t dim_k);

int main(int argc, char* argv[])
{

  if(argc != 5)
  {
    std::cout<<"Wrong number of parameters!\n";
    std::cout<<"Usage: "<<argv[0]<<" <dim_i> <dim_j> <dim_k> <.aocx file path>\n";
    std::cout<<"A(i,k)*B(k,j)=C(i,j)"<<std::endl;

    return EXIT_FAILURE;
  }

  const size_t dim_i = std::atoi(argv[1]);
  const size_t dim_j = std::atoi(argv[2]);
  const size_t dim_k = std::atoi(argv[3]);
  const char* aocx_filename = argv[4];

  if(check_dims(dim_i, dim_j, dim_k))
    return EXIT_FAILURE;

  const size_t LENGTH_A = dim_i*dim_k;
  const size_t LENGTH_B = dim_k*dim_j;
  const size_t LENGTH_C = dim_i*dim_j;

  const size_t buffer_size_A = sizeof(float)*LENGTH_A; 
  const size_t buffer_size_B = sizeof(float)*LENGTH_B; 
  const size_t buffer_size_C = sizeof(float)*LENGTH_C; 

  float *source_a, *source_b, *sp_fpga_result;

  int err;
  err  = posix_memalign((void**)&source_a, 64*sizeof(float), buffer_size_A);
  err |= posix_memalign((void**)&source_b, 64*sizeof(float), buffer_size_B);
  err |= posix_memalign((void**)&sp_fpga_result, 64*sizeof(float), buffer_size_C);

  if(err)
  {
    std::cout<<"Error: host memory allocation." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout<<"\n Matrix multiplication sizes"<<std::endl;
  std::cout<<"  A = ("<<dim_i<<", "<<dim_k<<")"<<std::endl;
  std::cout<<"  B = ("<<dim_k<<", "<<dim_j<<")"<<std::endl;
  std::cout<<"  C = ("<<dim_i<<", "<<dim_j<<")"<<std::endl;

#ifdef CHECK_WITH_ONES

  std::vector<double> dp_host_result (LENGTH_C, dim_k);

  #pragma omp parallel for
  for(size_t i=0; i < LENGTH_A; i++)
    source_a[i] = 1; 

  #pragma omp parallel for
  for(size_t i=0; i < LENGTH_B; i++)
     source_b[i] = 1;

  std::cout<<"\nMatrices are filled with ones, result = "<<dim_k<<std::endl;
 
#else

  std::cout<<"\nComputing matrix multiplication on host (it can take long) ... ";

  const float normalize = 1.0f/float(RAND_MAX);

  std::srand(1);

  for(size_t i=0; i < LENGTH_A; i++)
    source_a[i] = normalize*float(rand());

  for(size_t i=0; i < LENGTH_B; i++)
    source_b[i] = normalize*float(rand());

  std::vector<double> dp_host_result (LENGTH_C, 0.0);

  // B and C are row-major, A is col-major
  #pragma omp parallel for
  for(size_t i=0; i < dim_i; i++)
    for(size_t k=0; k < dim_k; k++)
      for(size_t j=0; j < dim_j; j++)
        dp_host_result[i*dim_j+j] += source_a[k*dim_i+i]*source_b[k*dim_j+j]; 

  std::cout<<"done!"<<std::endl;

#endif

  // OpenCL host code starts --------------------------------------------------

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  std::cout<<"\nFound "<<platforms.size()<<" OpenCL platforms.\n";
  std::cout<<"\nSelecting OpenCL device (PLATFORM_ID="
           <<PLATFORM_ID<<", DEVICE_ID="<<DEVICE_ID<<")"
           <<std::endl;

  cl::Platform platform = platforms[PLATFORM_ID];
  std::cout<<"  Platform: "<<platform.getInfo<CL_PLATFORM_NAME>()<<std::endl;

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);

  cl::Device device = devices[DEVICE_ID];
  std::cout<<"  Device: "<<device.getInfo<CL_DEVICE_NAME>()<<std::endl;

  cl::Context context(devices);

  cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

  std::cout<<"\nLoading bitstream form "<<aocx_filename<<" ..."<<std::endl;
  std::ifstream bin_file(aocx_filename, std::ifstream::binary);
  if(bin_file.fail())
  {
    std::cout<<"Error reading "<<aocx_filename<<std::endl;
    return EXIT_FAILURE;
  }

  bin_file.seekg (0, bin_file.end);
  size_t nb = bin_file.tellg();
  bin_file.seekg (0, bin_file.beg);

  char *buf = new char [nb];
  bin_file.read(buf, nb);

  cl::Program::Binaries bins;
  bins.push_back({buf,nb});

  devices.resize(1);

  std::cout<<"\nProgram creation";
  cl_int prog_err;
  cl::Program program(context, devices, bins, NULL, &prog_err);

  if (prog_err != CL_SUCCESS)
  {
    std::cout<<"  KO"<<std::endl;
    std::cout<<"Error in program creation."<<std::endl;
    return EXIT_FAILURE;
  } else std::cout<<"  OK"<<std::endl;
  
  program.build();

  std::cout<<"\nKernel creation";
  
  std::string kernel_name("krnl_systo_0");

  cl_int kernel_err;
  cl::Kernel kernel(program, kernel_name.c_str(), &kernel_err);

  if (kernel_err != CL_SUCCESS)
  {
    std::cout<<"  KO"<<std::endl;
    std::cout<<"Error in kernel creation."<<std::endl;
    return EXIT_FAILURE;
  } else std::cout<<"  OK"<<std::endl;

  cl_int cbuff_err, errb;
  
  cl::Buffer buffer_a(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA|CL_MEM_HETEROGENEOUS_INTELFPGA,  buffer_size_A/*, &cbuff_err*/); errb = cbuff_err;
  cl::Buffer buffer_b(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA|CL_MEM_HETEROGENEOUS_INTELFPGA,  buffer_size_B/*, &cbuff_err*/); errb |= cbuff_err;
  cl::Buffer buffer_c(context, CL_MEM_WRITE_ONLY|CL_CHANNEL_3_INTELFPGA|CL_MEM_HETEROGENEOUS_INTELFPGA, buffer_size_C/*, &cbuff_err*/); errb |= cbuff_err;  

  if(errb)
  {
    std::cout<<"Error: device memory allocation." << std::endl;
    return EXIT_FAILURE;
  }


  std::cout<<"\nSet arguments";
  int arg_err;
  arg_err = kernel.setArg(0, buffer_a);
  arg_err += kernel.setArg(1, buffer_b);
  arg_err += kernel.setArg(2, buffer_c);
  arg_err += kernel.setArg(3, unsigned(dim_i));
  arg_err += kernel.setArg(4, unsigned(dim_j));
  arg_err += kernel.setArg(5, unsigned(dim_k));

  if (arg_err != CL_SUCCESS)
  {
    std::cout<<"  KO"<<std::endl;
    std::cout<<"Error in setting kernel arguments."<<std::endl;
    return EXIT_FAILURE;
  } else std::cout<<"  OK"<<std::endl;

  cl::Event kernel_event;

  q.enqueueWriteBuffer(buffer_a, CL_TRUE, 0, buffer_size_A, source_a);
  q.enqueueWriteBuffer(buffer_b, CL_TRUE, 0, buffer_size_B, source_b);

  q.finish();
  std::cout<<"\nKernel execution start"<<std::endl;

  q.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(1),cl::NDRange(1),NULL,&(kernel_event));

  q.finish();
  std::cout<<"Kernel execution finish"<<std::endl;

  q.enqueueReadBuffer(buffer_c, CL_TRUE, 0, buffer_size_C, sp_fpga_result);

  unsigned long start_time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  unsigned long end_time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  unsigned long elapsed_ns = end_time - start_time;

  std::cout<<"\nKernel times [ns]"
           <<"\tstart="<<start_time<<"\tstop="<<end_time
           <<"\telapsed="<<elapsed_ns<<std::endl;


  // OpenCL host code ends ----------------------------------------------------

  // Result Check

  std::vector<size_t> ulp_histogram(16,0); 

  double max_rel_err = 0;
  unsigned long max_ulp = 0;
  unsigned long min_ulp = std::numeric_limits<unsigned long>::max();
  unsigned long mean_ulp = 0;

  std::vector<double> v_rel_err;
  v_rel_err.reserve(LENGTH_C);
  std::vector<unsigned long> v_ulp;
  v_ulp.reserve(LENGTH_C);

  bool mean_ulp_wrap = false;    

  float_classifier<float> _fpclass;

  for(size_t i = 0; i < LENGTH_C; i++) 
  {
    _fpclass.eval(sp_fpga_result[i]);

    double ferr = std::abs((float(dp_host_result[i]) - sp_fpga_result[i])/float(dp_host_result[i]));
    unsigned long ulp = float_ulp_distance(float(dp_host_result[i]), sp_fpga_result[i]);
    if(ulp < ulp_histogram.size()) ++ulp_histogram[ulp];

    v_rel_err.push_back(ferr);
    v_ulp.push_back(ulp);

    max_ulp = std::max(max_ulp, ulp);
    min_ulp = std::min(min_ulp, ulp);
    max_rel_err = std::max(max_rel_err, ferr);
    mean_ulp_wrap |= (mean_ulp + ulp < mean_ulp);
    mean_ulp += ulp;
  }

  std::cout<<"\nResult check"<<std::endl;
  std::cout<<"\n  Floating-point class output summary"<<std::endl;
  const double perc_tot = 100.0/_fpclass.total();
  std::cout<<"    #NaNs:\t"<<_fpclass.nans()<<" ("<<perc_tot*_fpclass.nans()<<" %)"<<std::endl;
  std::cout<<"    #infs:\t"<<_fpclass.infs()<<" ("<<perc_tot*_fpclass.infs()<<" %)"<<std::endl;
  std::cout<<"    #normals:\t"<<_fpclass.normals()<<" ("<<perc_tot*_fpclass.normals()<<" %)"<<std::endl;
  std::cout<<"    #subnormals:\t"<<_fpclass.subnormals()<<" ("<<perc_tot*_fpclass.subnormals()<<" %)"<<std::endl;
  std::cout<<"    #zeros:\t"<<_fpclass.zeros()<<" ("<<perc_tot*_fpclass.zeros()<<" %)"<<std::endl;

  auto max_rel_err0 = std::max_element(v_rel_err.begin(), v_rel_err.end());
  size_t pos_max_rel_err0 = max_rel_err0 - v_rel_err.begin();
  auto max_ulp0 = std::max_element(v_ulp.begin(), v_ulp.end());
  size_t pos_max_ulp0 = max_ulp0 - v_ulp.begin(); 

  const int max_precision = std::numeric_limits<float>::digits10 + 1;
 
  std::cout<<"\n  Correctness"<<std::endl;

  std::cout<<"    Max ULP distance: "<<*max_ulp0
            <<"  at position: ("<<pos_max_ulp0/dim_j<<", "<<pos_max_ulp0%dim_j<<")"
            <<"  host: "<<std::setprecision(max_precision)<<float(dp_host_result[pos_max_ulp0])
            <<"  fpga: "<<std::setprecision(max_precision)<<sp_fpga_result[pos_max_ulp0]
            <<std::endl;

  std::cout<<"    ULP distance: min("<<min_ulp<<") max("<<max_ulp<<") mean("<<double(mean_ulp)/double(LENGTH_C)<<")";
  if(mean_ulp_wrap) std::cout<<"[ERROR mean ulp wrapped!]";
  std::cout<<std::endl; 

  std::cout<<"\n    Max relative error: "<<*max_rel_err0
             <<"  at position: ("<<pos_max_rel_err0/dim_j<<", "<<pos_max_rel_err0%dim_j<<")"
             <<"  host: "<<std::setprecision(max_precision)<<float(dp_host_result[pos_max_rel_err0])
             <<"  fpga: "<<std::setprecision(max_precision)<<sp_fpga_result[pos_max_rel_err0]
             <<std::endl;

#ifdef PRINT_ULP_HIST
  std::cout<<"\n  ULP histogram"<<std::endl;
  for(size_t i=0; i<ulp_histogram.size(); ++i)
    std::cout<<"    "<<i<<"  ULP:\t"<<ulp_histogram[i]<<"\t("<<100.0*double(ulp_histogram[i])/double(LENGTH_C)<<"%)"<<std::endl; 
#endif

  std::cout<<"\nKernels performances"<<std::endl;
  const size_t ops = size_t(dim_i)*size_t(dim_j)*size_t(dim_k+dim_k-1); // (dim) multiplications + (dim-1) add for each matrix element 
  const double GFLOPS = double(ops)/double(elapsed_ns); 
  std::cout<<"  Overall kernel execution time: "<<elapsed_ns<<" ns"<<std::endl;
  std::cout<<"  Overall kernel floating-point performance: "<<GFLOPS<<" GFLOPS"<<std::endl;


  //delete buf;
  free(source_a);
  free(source_b);
  free(sp_fpga_result);

  return 0;

}

unsigned int float_to_uint(float f_a)
{

  static_assert(sizeof(float) == sizeof(unsigned int), "unsigned int/float sizes differ"); 

  unsigned int u_a;
  std::memcpy(&u_a, &f_a, sizeof(float));

  return u_a; 
}

unsigned int float_ulp_distance(float f_a, float f_b)
{

  static_assert(sizeof(float) == sizeof(unsigned int), "unsigned int/float sizes differ"); 

  unsigned int ulp;
  unsigned int a, b;

  std::memcpy(&a, &f_a, sizeof(float));
  std::memcpy(&b, &f_b, sizeof(float));

  if (a > b)
    ulp = a - b;
    else
      ulp = b - a;

  return ulp; 
}

#include"config.h"

#ifndef long_main_idx_bits 
#define long_main_idx_bits (32u)
#else
#define long_main_idx_bits (64u)
#endif

#ifndef nb2_idx_bits
#define nb2_idx_bits (16u)
#else
#define nb2_idx_bits (32u)
#endif

int check_dims(size_t dim_i, size_t dim_j, size_t dim_k)
{

  const size_t RA = dim0_i*dim0_k/GMEM_WIDTH_A;
  const size_t RB = dim0_j*dim0_k/GMEM_WIDTH_B;
  const size_t dim1_i = RB*dim0_i;
  const size_t dim1_j = RA*dim0_j;

  if( (dim_i%dim1_i) || (dim_j%dim1_j) || (dim_k%dim0_k) )
  {
    std::cout<<"Error: dim_i needs to be a multiple of "<<dim1_i;
    std::cout<<", dim_j of "<<dim1_j;
    std::cout<<" and dim_k of "<<dim0_k<<"."<<std::endl;
    return 1; 
  } 

  const size_t NB2_i = dim_i/dim1_i;
  const size_t NB2_j = dim_j/dim1_j;

  const size_t NB2_i_bits = std::ceil(std::log2(double(NB2_i)));
  const size_t NB2_j_bits = std::ceil(std::log2(double(NB2_j)));

  const size_t long_main_it = (dim_i/dim0_i)*(dim_j/dim0_j)*((dim_k/dim0_k)+1+dim0_i);
  const size_t long_main_it_bits = std::ceil(std::log2(double(long_main_it)));

  int err = 0;

  if( (NB2_i_bits > nb2_idx_bits) || (NB2_j_bits > nb2_idx_bits) )
  {
    std::cout<<"Error: nb2_idx_t too small.\n";
    std::cout<<"set #define nb2_idx_bits "<<std::max(NB2_i_bits,NB2_j_bits)<<" in config.h"<<std::endl;
    err += 2;
  }  

  if(long_main_it_bits > long_main_idx_bits)
  {
    std::cout<<"Error: long_main_idx_t too small.\n";
    std::cout<<"set #define long_main_idx_bits "<<long_main_it_bits<<" in config.h"<<std::endl;
    err += 4;
  } 

  return err;

}


