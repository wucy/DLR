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


using namespace boost::numeric::ublas;

using std::cin;
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;



class Trainer {
	public: 
	
	Trainer(const NNet & n, char * ti_fn):nnet(n) 
	{
		InputProxy(ti_fn);
	}


	void InputProxy(char * fn);

	struct TrainItem
	{
		int label;
		vector< float > feature;
		TrainItem(int ll, const vector< float > & fea):feature(fea)
		{
			label = ll;
		}
	};

	protected:
	
	std::vector< TrainItem > training_items;

	NNet nnet;
};
