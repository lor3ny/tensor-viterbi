# =========================================================================== #
#  Makefile — tensor-viterbi
# =========================================================================== #

# ── Toolchain ────────────────────────────────────────────────────────────── #
CUDA_HOME ?= /leonardo/prod/opt/compilers/cuda/12.6/none/bin/nvcc
CXX      := $(CUDA_HOME)/bin/nvcc
CXXFLAGS := -std=c++17 -O2 -Xcompiler "-Wall -Wextra -Wno-unused-parameter -Wno-unused-variable"
LDFLAGS  := -L$(CUDA_HOME)/lib64
ARCH 	 := -arch=sm_80 # Leonardo CINECA — NVIDIA A100

# ── Sources & target ─────────────────────────────────────────────────────── #
TARGET   := tensor-viterbi
SRCS_CPP := src/main.cpp
SRCS_CU  := src/hsmm.cu src/kernels.cu
OBJS     := $(SRCS_CPP:.cpp=.o) $(SRCS_CU:.cu=.o)
HEADERS  := src/hsmm.hpp src/json.hpp src/kernels.cuh


# ── Default rule ─────────────────────────────────────────────────────────── #
.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

# ── Compilation rules ────────────────────────────────────────────────────── #
src/main.o: src/main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

src/hsmm.o: src/hsmm.cu src/hsmm.hpp src/kernels.cuh
	$(CXX) $(CXXFLAGS) $(ARCH) -c -o $@ $<

src/kernels.o: src/kernels.cu src/kernels.cuh
	$(CXX) $(CXXFLAGS) $(ARCH) -c -o $@ $<

# ── Convenience targets ──────────────────────────────────────────────────── #
.PHONY: debug
debug: CXXFLAGS := -std=c++17 -g -DDEBUG -Xcompiler "-Wall -Wextra"
debug: $(TARGET)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(TARGET)