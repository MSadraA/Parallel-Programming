# ====================================================================
# Compiler & Flags
# ====================================================================
CXX      := C:/msys64/mingw64/bin/g++

# Compiler Flags
# Added -fopenmp because we are working on an OpenMP project
CXXFLAGS := -g -mavx -mfma -std=c++17 -pthread -O2 -fopenmp

# ====================================================================
# OpenCV Configuration
# ====================================================================
OPENCV_CPPFLAGS := -IC:/msys64/mingw64/include/opencv4
OPENCV_LIBS     := -LC:/msys64/mingw64/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

# ====================================================================
# Combined Settings
# ====================================================================
# Preprocessor Flags (Includes + Definitions)
CPPFLAGS := $(OPENCV_CPPFLAGS) -D_GLIBCXX_HAS_GTHREADS=1

# Linker Flags (Libraries)
LDFLAGS  := $(OPENCV_LIBS)

# ====================================================================
# Automatic Targets
# ====================================================================
# 1. Find all .cpp files in the current directory
SRCS := $(wildcard *.cpp)

# 2. Convert .cpp filenames to .out filenames for targets
TARGETS := $(SRCS:.cpp=.out)

# ====================================================================
# Rules
# ====================================================================
.PHONY: all clean

# Default target: build all executables
all: $(TARGETS)

# Rule to build a specific .out from a .cpp
# $< : The source file (e.g., main.cpp)
# $@ : The target file (e.g., main.out)
%.out: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS)

# Clean up generated files
clean:
	rm -f $(TARGETS) *.o *.exe