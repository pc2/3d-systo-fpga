#define write_lmem2(_ii, _jj, write_idx, GMEM_WIDTH, GMEM_VALS, __idx_type__, M, dim1_x, dim0_y, dim0_x) \
{ \
\
  __idx_type__ __write_idx[dim0_y][dim0_x]; \
\
  dim0_idx_t __ii[dim0_y][dim0_x]; \
  const dim0_idx_t _idx = _jj*dim0_y + _ii ; \
\
  _Pragma("unroll") \
  for(dim0_idx_t _i=0; _i<dim0_y; ++_i) \
    _Pragma("unroll") \
    for(dim0_idx_t _j=0; _j<dim0_x/GMEM_WIDTH; ++_j ) \
      _Pragma("unroll") \
      for(dim0_idx_t _k=0; _k<GMEM_WIDTH; ++_k) \
      { \
\
        __ii[_i][(GMEM_WIDTH*_j+_k)] = ((!_j) ? __fpga_reg(_idx) : __fpga_reg(__ii[_i][(GMEM_WIDTH*_j+_k)-1])); \
        __write_idx[_i][(GMEM_WIDTH*_j+_k)] = ((!_j) ? __fpga_reg(write_idx) : __fpga_reg(__write_idx[_i][(GMEM_WIDTH*_j+_k)-1])); \
\
        if(__ii[_i][(GMEM_WIDTH*_j+_k)] == _j*dim0_y + _i ) \
          M[__write_idx[_i][(GMEM_WIDTH*_j+_k)]][_i][(GMEM_WIDTH*_j+_k)] = GMEM_VALS[_k]; \
\
      } \
\
} \
\

void read_gmem( const dim2_idx_t y1, const dim1_idx_t x1, const uchar GMEM_SIZE,
                const dim2_idx_t dim2_x, const nb2_idx_t X2,
                const dim1_idx_t dim1_x, 
                __global volatile const float * restrict gmem_data, float data[] )
{

  const ulong gmem_idx = ((ulong)dim1_x)*((dim2_x/dim1_x)*y1 + X2) + GMEM_SIZE*x1; // burst-aligned

  #pragma unroll 
  for(uchar _i=0; _i<GMEM_SIZE; ++_i)
    data[_i] = gmem_data[gmem_idx+_i];

}

