# --- Compiler ---
CC = g++

# --- Flags ---
# CXXFLAGS = -g -msse3 -mavx -mavx2 -mfma -std=c++17 -fopenmp
CXXFLAGS = -g -msse3 -mavx -mavx2 -mfma -std=c++17 -fopenmp -O2

# --- Paths---
IPP_INCLUDE_PATH = /opt/intel/oneapi/2025.3/include
IPP_LIB_PATH = /opt/intel/oneapi/2025.3/lib
OCV_INCLUDE_PATH = /usr/include/opencv4

# --- Linker ---
LDFLAGS = -L$(IPP_LIB_PATH)
LDFLAGS += -Wl,-rpath,$(IPP_LIB_PATH)
LDLIBS = -lippcore -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

# --- Preprocessor ---
CPPFLAGS = -I$(IPP_INCLUDE_PATH) -I$(OCV_INCLUDE_PATH)

# --- Targets---
SRCS_CPP = $(wildcard *.cpp)
SRCS_CXX = $(wildcard *.c++)
SRCS = $(SRCS_CPP) $(SRCS_CXX)

BINS_CPP = $(patsubst %.cpp, %.out, $(SRCS_CPP))
BINS_CXX = $(patsubst %.c++, %.out, $(SRCS_CXX))
BINS = $(BINS_CPP) $(BINS_CXX)

# --- Rules ---
all: $(BINS)

%.out: %.o
	$(CC) $(CXXFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)

%.o: %.cpp
	$(CC) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

%.o: %.c++
	$(CC) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(BINS)
	rm -f *.jpg