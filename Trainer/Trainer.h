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
	
	Trainer(const NNet & n, char * fn_lab, char * fn_fea, int dim, int sil_use_mode = 0):nnet(n) 
	{
		InputProxy(fn_lab, fn_fea, dim, sil_use_mode);
	}


	void InputProxy(char * fn_lab, char * fn_fea, int dim, int sil_use_mode);

	struct TrainItem
	{
		int label;
		vector< float > feature;
		TrainItem(int ll, const vector< float > & fea):feature(fea)
		{
			label = ll;
		}
	};
	

	struct CriteriaItem
	{
		float obj;
		float acc;
		CriteriaItem(float a, float o)
		{
			acc = a;
			obj = o;
		}
	};

	protected:

	//virtual ErrItem TotalErr(const NNet::Transform & trans) {}
	
	std::vector< TrainItem > training_items;
	
	NNet nnet;


};
