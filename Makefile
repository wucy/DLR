export CC = g++

<<<<<<< HEAD
export FLAGS = -O3 -Wall -DNDEBUG
=======
export FLAGS = -Wall #-DNDEBUG
>>>>>>> 40429d0bff00def6e25281793c213bde38f8ac2d

BIN = oDLRTrain test

oDLRTrain : oDLRTrain.o Trainer.o NNet.o Util.o
<<<<<<< HEAD
	$(CC) oDLRTrain.o Trainer/Trainer.o Trainer/oDLRTrainer.o NNet/NNet.o Util/Util.o -o oDLRTrain -O3 -DNDEBUG
=======
	$(CC) oDLRTrain.o Trainer/oDLRTrainer.o NNet/NNet.o Util/Util.o -o oDLRTrain -O3
>>>>>>> 40429d0bff00def6e25281793c213bde38f8ac2d

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
