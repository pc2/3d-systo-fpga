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

#include"config.h"

#define RA (dim0_i*dim0_k/GMEM_WIDTH_A)
#define RB (dim0_j*dim0_k/GMEM_WIDTH_B)

#define NB1_i RB 
#define NB1_j RA

#define SHIFT (NB1_i*NB1_j)
#define dim1_i (NB1_i*dim0_i) 
#define dim1_j (NB1_j*dim0_j)  

typedef float A0_type[dim0_k_p][dim0_i_p]; // attention A must be saved col-major
typedef float B0_type[dim0_k_p][dim0_j_p];
typedef float C0_type[dim0_i][dim0_j];

#include"index_type_selection_macros.h"
#include"error_check_macros.h"
#include"memory_functions.h"
#include"systolic_mmm.h"

#define _inc(x1, mod1, x2, mod2, x3, mod3, x4) { if(++x1==mod1) { x1=0; if(++x2==mod2) { x2=0;  if(++x3==mod3) { x3=0; ++x4; } } } }
#define _inc1(x1, mod1, x2, mod2) { if(++x1==mod1) { x1=0; if(++x2==mod2) { x2=0;  } } }
#define _inc2(x1, mod1, x2, mod2, x3, mod3) { if(++x1==mod1) { x1=0; if(++x2==mod2) { x2=0;  if(++x3==mod3) x3=0; } } }

__kernel void 
__attribute__((max_work_group_size(1,1,1)))
__attribute__((uses_global_work_offset(0)))
krnl_systo_0( __global volatile const float * restrict gA,
              __global volatile const float * restrict gB,
              __global volatile float * restrict gC,
              const uint dim2_i, const uint dim2_j, const uint dim2_k )
{

  A0_type A1[2*NB1_i] __attribute((numbanks(dim0_k_p*dim0_i_p)));
  B0_type B1[2*NB1_j] __attribute((numbanks(dim0_k_p*dim0_j_p)));
  C0_type C0[SHIFT] __attribute((register));

  const nb2_idx_t NB2_i = dim2_i/dim1_i;
  const nb2_idx_t NB2_j = dim2_j/dim1_j;

  const main_idx_t write_it = dim0_i*dim0_j/dim0_j; // number of iterations to write a block of C
  const main_idx_t read_it = dim2_k/dim0_k; // number of iterations to compute a block of C
  
  const main_idx_t lev1_mod = (1 + read_it + write_it);
  const main_idx_t lev1_lenght = NB1_i*NB1_j*lev1_mod; // total number of iterations to compute a block of C
  const main_idx_t lev0_lenght = NB2_i*NB2_j*lev1_lenght; // total number of iterations to compute entire C

#ifdef DEBUG_INFO 

  printf("\n\t**** DEBUG INFO START ****\n\n");

  printf("\t---> SHIFT=%u=%u=%u\n", SHIFT, dim1_i*dim0_k/GMEM_WIDTH_A, dim1_j*dim0_k/GMEM_WIDTH_B);
  printf("\t---> GMEM_WIDTH_A=%u GMEM_WIDTH_B=%u\n", GMEM_WIDTH_A, GMEM_WIDTH_B);
  printf("\t---> dim1_i=%u dim1_j=%u\n", dim1_i, dim1_j);
  printf("\t---> NB1_i=%u NB1_j=%u\n", NB1_i, NB1_j);
  printf("\t---> NB2_i=%u NB2_j=%u\n", NB2_i, NB2_j);
  printf("\t---> A1 banks=%u depth=%u\n", dim0_k*dim0_i, 2*NB1_i);
  printf("\t---> B1 banks=%u depth=%u\n", dim0_k*dim0_j, 2*NB1_j);
  printf("\t---> comp %f\n", ((double)(read_it))/((double) lev1_lenght));
  printf("\t---> %u %u\n", lev1_lenght, lev0_lenght);

  printf("\n\t**** DEBUG INFO END ****\n\n");

#endif

  shift_idx_t s_idx = 0; main_idx_t m_idx = 0; nb2_idx_t I2 = 0; nb2_idx_t J2 = 0;
  shift_idx_t p_idx = 0; dim2_idx_t q_idx = 0;
  nb2_idx_t c1_idx = 0; nb2_idx_t c2_idx = 0; dim0_idx_t c3_idx = 0; 

  for(main_idx_t I=0; I<lev0_lenght; ++I)
  {
    const uchar XX = ((I/(NB1_i*NB1_j))%2u); 

    const bool zero_shift = !m_idx;
    const bool read = m_idx < read_it;
    const bool write = (m_idx >= read_it+1);
  
    //read      
    float volatile __attribute((register)) tmp_A[GMEM_WIDTH_A] = {0};
    float volatile __attribute((register)) tmp_B[GMEM_WIDTH_B] = {0};

    {
      const dim2_idx_t y1 = q_idx; //(I/(SHIFT/dim0_k))%dim2_k; 
      const dim1_idx_t x1 = p_idx; //I%(SHIFT/dim0_k); 
   
      if(read) read_gmem(y1, x1, GMEM_WIDTH_A, dim2_i, I2, dim1_i, gA, tmp_A); // read from gmem
      if(read) read_gmem(y1, x1, GMEM_WIDTH_B, dim2_j, J2, dim1_j, gB, tmp_B);
    }

    {
      const dim0_idx_t _ii = (I/(SHIFT/dim0_k))%dim0_k; 

      const dim0_idx_t _jjA = I%(dim0_i/GMEM_WIDTH_A);
      const dim0_idx_t _jjB = I%(dim0_j/GMEM_WIDTH_B);

      const A1_idx_t _wwA = ((I/(dim0_i/GMEM_WIDTH_A))%(NB1_i))*2u + XX;
      const B1_idx_t _wwB = ((I/(dim0_j/GMEM_WIDTH_B))%(NB1_j))*2u + XX;

      write_lmem2(_ii, _jjA, _wwA, GMEM_WIDTH_A, tmp_A, A1_idx_t, A1, dim1_i, dim0_k, dim0_i); // write to local memory
      write_lmem2(_ii, _jjB, _wwB, GMEM_WIDTH_B, tmp_B, B1_idx_t, B1, dim1_j, dim0_k, dim0_j);
    }

    // write
    float volatile tmp_C[dim0_j] __attribute((register));

    const dim0_idx_t _i0w = (I/(NB1_i*NB1_j))%dim0_i; // row selected within the shift register to be written to gmem
    dim0_idx_t i0w[dim0_i][dim0_j];

    #pragma unroll
    for(dim0_idx_t i=0; i<dim0_i; ++i)
      #pragma unroll
      for(uint j=0; j<dim0_j; ++j)
      {
        i0w[i][j] = ((!i) ? ((!j) ? __fpga_reg(_i0w) : __fpga_reg(i0w[i][j-1])) : __fpga_reg(i0w[i-1][j])); 

        if(i == i0w[i][j])         
          tmp_C[j]=C0[0][i][j];
      }


    {
      const dim1_idx_t y1 = dim0_i*c2_idx /*((I/NB1_j)%NB1_i)*/ + /*(I/SHIFT)%dim0_i*/ c3_idx;
      const dim1_idx_t x1 = c1_idx; //I%NB1_j; 
      const ulong gmem_idx = (((ulong)dim1_j)*(dim1_i*(I2*dim2_j/dim1_j) + (dim2_j/dim1_j)*y1 + J2) + dim0_j*x1); // burst-aligned 
  
      if(write) 
      #pragma unroll
      for(uchar j=0; j<dim0_j; ++j)
        gC[gmem_idx+j]=tmp_C[j];
    }

    // compute
    const A1_idx_t I1 = (I/NB1_j)%NB1_i;  
    const B1_idx_t J1 = I%NB1_j;  

    const A1_idx_t A1_idx = I1*2u + !XX;
    const B1_idx_t B1_idx = J1*2u + !XX;

    systolic_mmm(A1[A1_idx], B1[B1_idx], C0[0]); 

    // shift
    bool zs[dim0_i][dim0_j];

    #pragma unroll 
    for(uint i=0; i<dim0_i; ++i)
      #pragma unroll
      for(uint j=0; j<dim0_j; ++j)
      {
        zs[i][j] = ((!i) ? ((!j) ? __fpga_reg(zero_shift) : __fpga_reg(zs[i][j-1])) : __fpga_reg(zs[i-1][j])); 
        const float _t = (zs[i][j]) ? 0.0f : __fpga_reg(C0[0][i][j]);
        #pragma unroll 
        for(uint _s=0; _s<SHIFT-1; ++_s)
          C0[_s][i][j] =  C0[_s+1][i][j];
        C0[SHIFT-1][i][j] = _t;
      }

    _inc(s_idx, SHIFT, m_idx, lev1_mod, J2, NB2_j, I2);
    _inc1(p_idx, SHIFT/dim0_k, q_idx, dim2_k);
    _inc2(c1_idx, NB1_j, c2_idx, NB1_i, c3_idx, dim0_i);

  }

}

