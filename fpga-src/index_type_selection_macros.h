#if (dim0_i<256)&(dim0_j<256)&(dim0_k<256)
  typedef uchar dim0_idx_t;
#else
  typedef ushort dim0_idx_t;
#endif

#if (dim1_i<256)&(dim1_j<256)&(dim1_k<256)
  typedef uchar dim1_idx_t;
#else
  typedef ushort dim1_idx_t;
#endif

typedef uint dim2_idx_t;

#if (2*NB1_i)<256 
  typedef uchar A1_idx_t;
#else
  typedef ushort A1_idx_t;
#endif

#if (2*NB1_j)<256 
  typedef uchar B1_idx_t;
#else
  typedef ushort B1_idx_t;
#endif

#if SHIFT<65536 
  typedef ushort shift_idx_t;
#else
  typedef uint shift_idx_t;
  #error unespected 
#endif

#ifndef main_idx_bits 
  typedef uint main_idx_t;
#else
  typedef unsigned int __attribute__((__ap_int(long_main_idx_bits))) main_idx_t;
#endif

#ifndef nb2_idx_bits
  typedef ushort nb2_idx_t;
#else
  typedef unsigned int __attribute__((__ap_int(nb2_idx_bits))) nb2_idx_t;
#endif


