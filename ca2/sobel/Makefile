# --- Compiler ---
CXX := C:/msys64/mingw64/bin/g++

# --- Target ---
TARGET = sobel.out

# --- Source ---
SRC = sobel.cpp

# --- Flags ---
CXXFLAGS = -g -mavx -std=c++17 -pthread -O2

# ====================================================================
# OpenCV Paths
OPENCV_CPPFLAGS = -IC:/msys64/mingw64/include/opencv4
OPENCV_LIBS = -LC:/msys64/mingw64/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
# ====================================================================

# --- Combined Flags ---
CPPFLAGS = $(OPENCV_CPPFLAGS) -D_GLIBCXX_HAS_GTHREADS=1

# --- Rules ---

# Default rule: build the target
all: $(TARGET)

# Rule to link the final executable
$(TARGET): $(SRC:.cpp=.o)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_LIBS)

# Rule to compile .cpp to .o
%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f *.o $(TARGET)

# Run rule
run: all
	@echo "--- Running Sobel ---"
	@./$(TARGET)
