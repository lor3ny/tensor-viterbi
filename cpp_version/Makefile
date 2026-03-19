# =========================================================================== #
#  Makefile — tensor-viterbi
# =========================================================================== #

# ── Toolchain ────────────────────────────────────────────────────────────── #
CUDA_HOME ?= /leonardo/prod/opt/compilers/cuda/12.6/none/bin/nvcc
CXX      := $(CUDA_HOME)/bin/nvcc
CXXFLAGS := -std=c++17 -O2 -Xcompiler "-Wall -Wextra"
LDFLAGS  := -L$(CUDA_HOME)/lib64

# ── Sources & target ─────────────────────────────────────────────────────── #
TARGET  := tensor-viterbi
SRCS    := src/main.cpp src/hsmm.cpp
OBJS    := $(SRCS:.cpp=.o)
HEADERS := src/hsmm.hpp src/json.hpp

# ── GPU detection ─────────────────────────────────────────────────────────── #
GPU_COUNT := $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
ifneq ($(GPU_COUNT), 0)
    CXXFLAGS += -DGPU_DETECTED
    $(info GPU detected: $(GPU_COUNT) device(s) found)
else
    $(info No GPU detected, building CPU-only)
endif

# ── Default rule ─────────────────────────────────────────────────────────── #
.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

# ── Compilation rules ────────────────────────────────────────────────────── #
src/main.o: src/main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

src/hsmm.o: src/hsmm.cpp src/hsmm.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# ── Convenience targets ──────────────────────────────────────────────────── #
.PHONY: debug
debug: CXXFLAGS := -std=c++17 -g -DDEBUG -Xcompiler "-Wall -Wextra"
debug: $(TARGET)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(TARGET)