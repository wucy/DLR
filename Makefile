export CC = g++

export FLAGS = -O3 -Wall -DNDEBUG

BIN = oDLRTrain test

oDLRTrain : oDLRTrain.o Trainer.o NNet.o Util.o
	$(CC) oDLRTrain.o Trainer/Trainer.o Trainer/oDLRTrainer.o NNet/NNet.o Util/Util.o -o oDLRTrain -O3 -DNDEBUG

oDLRTrain.o : oDLRTrain.cpp
	$(CC) -c oDLRTrain.cpp $(FLAGS)

test : test.o NNet.o Util.o
	$(CC) NNet/NNet.o Util/Util.o test.o -o test -O3

test.o : test.cpp
	$(CC) -c test.cpp $(FLAGS)


Trainer.o :
	$(MAKE) -C Trainer --print-directory

NNet.o :
	$(MAKE) -C NNet --print-directory

Util.o :
	$(MAKE) -C Util --print-directory




.PHONY : clean
clean :
	rm *.o */*.o $(BIN)

.PHONY : all
all : $(BIN)
	rm *.o */*.o
