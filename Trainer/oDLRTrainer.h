#pragma once


#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>


#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>

#include "../Util/Util.h"
#include "../NNet/NNet.h"
#include "Trainer.h"

using namespace boost::numeric::ublas;

using std::cin;
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;



class oDLRTrainer : public Trainer {
	public: 
	
	oDLRTrainer(const NNet & n, char * ti_fn):Trainer(n,ti_fn)
	{
	}

	
	struct Result
	{
		vector< float > M;
		vector< float > c;
		vector< float > a;
		int result_type;
	};
	Result SGD_only_c_Train(float eps, int round);
	//void BGD_only_c_train(float eps, int round);
	//void BGD_diagM_AND_c_Train(float eps, int round);
	//void BGD_a_Train(float eps, int round);

	private:

	float TotalErr(const NNet::Transform & trans, int round);
};
