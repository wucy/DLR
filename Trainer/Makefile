.PHONY : all
all : Trainer.o oDLRTrainer.o


Trainer.o : Trainer.cpp Trainer.h
	$(CC) -c Trainer.cpp ${FLAGS}

oDLRTrainer.o : oDLRTrainer.cpp oDLRTrainer.h Trainer.o
	$(CC) -c oDLRTrainer.cpp ${FLAGS}

