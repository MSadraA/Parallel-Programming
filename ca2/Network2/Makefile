# --- Compiler ---
CXX := C:/msys64/mingw64/bin/g++

# --- Flags ---
CXXFLAGS = -g -mavx -mfma -std=c++17 -pthread 

# ====================================================================
# OpenCV Paths
OPENCV_CPPFLAGS = -IC:/msys64/mingw64/include/opencv4
OPENCV_LIBS = -LC:/msys64/mingw64/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
# ====================================================================

# --- Combined Flags ---
CPPFLAGS = $(OPENCV_CPPFLAGS) -D_GLIBCXX_HAS_GTHREADS=1

# --- Automatic Targets and Sources ---
# 1. Find all .cpp files in the directory
SRCS = $(wildcard *.cpp)
# 2. Create a list of target .out files (e.g., sobel.cpp -> sobel.out)
TARGETS = $(SRCS:.cpp=.out)

# --- Rules ---

# Default rule: build all targets found
all: $(TARGETS)

# Pattern Rule to link executables
# This rule says: "To make a file like '%.out', you need '%.o'"
# (e.g., To make 'sobel.out', you need 'sobel.o')
%.out: %.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_LIBS)

# Pattern Rule to compile .cpp to .o
# This rule says: "To make a file like '%.o', you need '%.cpp'"
%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Clean rule: remove all .o files and all target executables
clean:
	rm -f *.o $(TARGETS)

# Run rules
# Since 'run' is ambiguous, we create specific run rules
run-network: network.out
	@echo "--- Running network ---"
	@./network.out

run-test: test.out
	@echo "--- Running Test ---"
	@./test.out

# Make 'make run' default to running network.out
run: run-network