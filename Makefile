export CC = g++

export FLAGS = -DNDEBUG

test : test.o NNet.o Util.o
	$(CC) NNet/NNet.o Util/Util.o test.o -o test -O3

test.o : test.cpp
	$(CC) -c test.cpp $(FLAGS)

NNet.o :
	$(MAKE) -C NNet --print-directory

Util.o :
	$(MAKE) -C Util --print-directory




.PHONY : clean
clean :
	rm *.o */*.o test

.PHONY : all
all : test
