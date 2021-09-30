#if dim0_i%GMEM_WIDTH_A
  #error  dim1_i is not a multiple of GMEM_WIDTH_A
#endif

#if dim0_j%GMEM_WIDTH_B
  #error  dim1_j is not a multiple of GMEM_WIDTH_B
#endif

#if !GMEM_WIDTH_A
  #error GMEM_WIDTH_A is zero
#endif

#if !GMEM_WIDTH_B
  #error GMEM_WIDTH_B is zero
#endif

#if  (SHIFT!=dim1_i*dim0_k/GMEM_WIDTH_A)|(SHIFT!=dim1_j*dim0_k/GMEM_WIDTH_B)
  #error dimensions do not match!
#endif

#if (GMEM_WIDTH_A>8)|(GMEM_WIDTH_B>8)
  #warning Possible stall on memory readings!
#endif

