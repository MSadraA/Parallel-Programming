CC=clang++

CFLAGS=-c `pkg-config --cflags opencv4`
LIBFLAGS=`pkg-config --libs opencv4`

INCDIR=-I/opt/intel/ipp/include -I/usr/local/Cellar/opencv/4.10.0_11/include/opencv4

LIBDIR=-L/opt/intel/ipp/lib -L/usr/local/Cellar/opencv/4.10.0_11/lib 

CST11=-std=c++11

all: main

main: main.o 
	$(CC) $(LIBFLAGS) $(LIBDIR) $(CST11) main.o -o main
#	$(CC) $(LIBFLAGS) $(LIBDIR) $(LIBFILE) $(CST11) main.o -o main

main.o: main.cpp
	$(CC) $(CFLAGS) $(INCDIR) $(CST11) main.cpp

clean:
	rm -f *o main

