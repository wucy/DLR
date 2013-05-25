#include "Trainer/oDLRTrainer.h"
#include "NNet/NNet.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
using namespace std;


int main(int argc, char * argv [])
{


	if (argc != 9)
	{
		cerr << "Usage: oDLRTrain NN_Name TFeats TLabs FeatsDim SilMode eps round out_dir" << endl;
		return 0;
	}


	
	char * nn = argv[1];
	char * train_fea = argv[2];
	char * train_lab = argv[3];

	int fea_dim = atoi(argv[4]);

	int sil_mode = atoi(argv[5]);

	float eps = atof(argv[6]);
	
	int round = atoi(argv[7]);
	
	string out_dir = argv[8];
	

	NNet nnet(nn);
	oDLRTrainer trainer(nnet, train_lab, train_fea, fea_dim, sil_mode);

	trainer.SGD_only_c_Train(eps, round, out_dir);
	//trainer.SGD_only_c_all_tied_Train(eps, round, out_dir);	
	return 0;
}
