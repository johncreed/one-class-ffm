CXX = g++
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CXX = g++
endif

#MKLPATH = /home/skypole/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64_lin
#MKLINCLUDE = /home/skypole/intel/compilers_and_libraries_2018.2.199/linux/mkl/include
MKLROOT = /home/ybw/intel/compilers_and_libraries_2018.2.199/linux/mkl
CXXFLAGS = -Wall -g -std=c++0x -march=native
MKLFLAGS = -m64 -I${MKLROOT}/include -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
# comment the following flags if you do not want to use OpenMP
DFLAG += -DUSEOMP
CXXFLAGS += -fopenmp
CXXFLAGS += $(MKLFLAGS)

all: train


train: train.cpp ffm.o
	$(CXX) $(CXXFLAGS) -o $@ $^

ffm.o: ffm.cpp ffm.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

clean:
	rm -f train predict ffm.o *.bin.*
