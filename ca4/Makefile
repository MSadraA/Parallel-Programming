# ====================================================================
# Compiler & Flags
# ====================================================================
CXX      := C:/msys64/mingw64/bin/g++

# Compiler Flags
# -fopenmp is essential for the parallel version
CXXFLAGS := -g -mavx -mavx2 -mfma -std=c++17 -pthread -O2 -fopenmp

# ====================================================================
# OpenCV Configuration
# ====================================================================
# Include Path
OPENCV_CPPFLAGS := -IC:/msys64/mingw64/include/opencv4

# Library Path & Libs
# ADDED: -lopencv_videoio (Required for Webcam/VideoWriter)
OPENCV_LIBS     := -LC:/msys64/mingw64/lib \
                   -lopencv_core \
                   -lopencv_imgproc \
                   -lopencv_highgui \
                   -lopencv_imgcodecs \
                   -lopencv_videoio

# ====================================================================
# Combined Settings
# ====================================================================
CPPFLAGS := $(OPENCV_CPPFLAGS) -D_GLIBCXX_HAS_GTHREADS=1
LDFLAGS  := $(OPENCV_LIBS)

# ====================================================================
# Automatic Targets
# ====================================================================
# 1. Find all .cpp files in the current directory
SRCS := $(wildcard *.cpp)

# 2. Convert .cpp filenames to .exe filenames for Windows
TARGETS := $(SRCS:.cpp=.exe)

# ====================================================================
# Rules
# ====================================================================
.PHONY: all clean

# Default target: build all executables
all: $(TARGETS)

# Rule to build a specific .exe from a .cpp
# $< : The source file (e.g., parallel_bonus.cpp)
# $@ : The target file (e.g., parallel_bonus.exe)
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS)

# Clean up generated files
clean:
	rm -f *.exe *.o *.avi