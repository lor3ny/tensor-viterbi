# =========================================================================== #
#  Makefile — tensor-viterbi
# =========================================================================== #

# ── Toolchain ────────────────────────────────────────────────────────────── #
CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2
LDFLAGS  :=

# ── Sources & target ─────────────────────────────────────────────────────── #
TARGET  := tensor-viterbi
SRCS    := src/main.cpp src/hsmm.cpp
OBJS    := $(SRCS:.cpp=.o)
HEADERS := src/hsmm.hpp src/json.hpp

# ── Default rule ─────────────────────────────────────────────────────────── #
.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

# ── Compilation rules ────────────────────────────────────────────────────── #
src/main.o: src/main.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

src/hsmm.o: src/hsmm.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# ── Convenience targets ──────────────────────────────────────────────────── #
.PHONY: debug
debug: CXXFLAGS := -std=c++17 -Wall -Wextra -g -DDEBUG
debug: $(TARGET)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(TARGET)