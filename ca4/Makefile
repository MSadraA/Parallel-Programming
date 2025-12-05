# ====================================================================
# Compiler & Flags
# ====================================================================
CXX      := C:/msys64/mingw64/bin/g++

# Compiler Flags
# -pthread is crucial for POSIX threads
CXXFLAGS = -g -mavx -mavx2 -mfma -std=c++17 -pthread -O2 -fopenmp

# ====================================================================
# OpenCV Configuration
# ====================================================================
OPENCV_CPPFLAGS := -IC:/msys64/mingw64/include/opencv4

# Added -lopencv_videoio specifically for VideoWriter support
OPENCV_LIBS     := -LC:/msys64/mingw64/lib \
                   -lopencv_core -lopencv_imgproc -lopencv_highgui \
                   -lopencv_imgcodecs -lopencv_videoio

# ====================================================================
# Combined Settings
# ====================================================================
CPPFLAGS := $(OPENCV_CPPFLAGS) -D_GLIBCXX_HAS_GTHREADS=1
LDFLAGS  := $(OPENCV_LIBS)

# ====================================================================
# Automatic Targets
# ====================================================================
# Finds all .cpp files (Assuming you put everything in main.cpp or single files)
SRCS := $(wildcard *.cpp)

# Convert .cpp filenames to .exe filenames
TARGETS := $(SRCS:.cpp=.exe)

# ====================================================================
# Rules
# ====================================================================
.PHONY: all clean

all: $(TARGETS)

# Rule to build .exe from .cpp
# $< : Source file
# $@ : Target file
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o *.exe