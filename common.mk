# Simple Makefile for testing

HOST_SRC:=../host-src/host.cpp
FPGA_SRC:=../fpga-src/krnl_systo.cl
INCLUDES:=../host-includes

BOARD:=-board=p520_hpc_sg280l
AOCL_COMPILE_CONFIG:=$(shell aocl compile-config)
AOCL_LINK_CONFIG:=$(shell aocl link-config)
AOCLFLAGS:= $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG)
CXXFLAGS:=$(CXXFLAGS) -O3 -std=c++11 -fopenmp
FPGAFLAGS:=-fp-relaxed -fpc -no-interleaving=default -global-ring -duplicate-ring

# Host targets

host: $(HOST_SRC) 
	$(CXX) $(CXXFLAGS) -I$(INCLUDES) -I. $(AOCLFLAGS) $< -o $@ 

clean:
	$(RM) host

# FPGA targets

krnl_systo.aocr: 
	aoc -rtl -v -v -v -report -g -W $(BOARD) $(FPGAFLAGS)  -I. -o krnl_systo.aocr $(FPGA_SRC)

krnl_systo-s$(SEED).aocx: krnl_systo.aocr
	aoc -high-effort -seed=$(SEED) -v -v -v -report -g -W $(BOARD) $(FPGAFLAGS) krnl_systo.aocr -o krnl_systo-s$(SEED).aocx

aocr: krnl_systo.aocr

aocx: krnl_systo-s$(SEED).aocx
