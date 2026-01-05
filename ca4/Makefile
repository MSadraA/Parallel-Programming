# ====================================================================
# Compiler & Flags
# ====================================================================
CXX := g++

# Compiler Flags
# -fopenmp فقط اگر واقعا OpenMP داری استفاده کن
CXXFLAGS := -g -mavx -mavx2 -mfma -std=c++17 -O3 -pthread -fopenmp

# ====================================================================
# OpenCV Configuration (Linux)
# ====================================================================
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4)

# ====================================================================
# Combined Settings
# ====================================================================
CPPFLAGS := $(OPENCV_CFLAGS)
LDFLAGS  := $(OPENCV_LIBS)

# ====================================================================
# Automatic Targets
# ====================================================================
# All cpp files
SRCS := $(wildcard *.cpp)

# Linux executables (no .exe)
TARGETS := $(SRCS:.cpp=)

# ====================================================================
# Rules
# ====================================================================
.PHONY: all clean

all: $(TARGETS)

# Build rule
%: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o *.avi
