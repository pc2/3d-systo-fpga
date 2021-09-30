#define __fpga_reg2(x) __fpga_reg(__fpga_reg(x)) 
#define __fpga_reg4(x) __fpga_reg(__fpga_reg(__fpga_reg(__fpga_reg(x))))
#define __fpga_reg8(x) __fpga_reg4(__fpga_reg4(x)) 


#define _inc(x, mod1, y, mod2, z) { if(++x==mod1) { x=0; if(++y==mod2) { y=0; ++z;  } } }

