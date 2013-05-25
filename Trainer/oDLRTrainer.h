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
<<<<<<< HEAD
using std::string;
=======

>>>>>>> 40429d0bff00def6e25281793c213bde38f8ac2d


class oDLRTrainer : public Trainer {
	public: 
	
<<<<<<< HEAD
	oDLRTrainer(const NNet & n, char * fn_lab, char * fn_fea, int dim, int sil_use_mode = 0):Trainer(n, fn_lab, fn_fea, dim, sil_use_mode)
	{
		GenOutputFeature(training_items);
	}


=======
	oDLRTrainer(const NNet & n, char * ti_fn):Trainer(n,ti_fn)
	{
	}

>>>>>>> 40429d0bff00def6e25281793c213bde38f8ac2d
	
	struct Result
	{
		vector< float > M;
		vector< float > c;
		vector< float > a;
		int result_type;
	};
<<<<<<< HEAD
	void SGD_only_c_Train(float eps, int round, const string & out_dir);
	void SGD_only_c_all_tied_Train(float eps, int round, const string & out_dir);
=======
	Result SGD_only_c_Train(float eps, int round);
>>>>>>> 40429d0bff00def6e25281793c213bde38f8ac2d
	//void BGD_only_c_train(float eps, int round);
	//void BGD_diagM_AND_c_Train(float eps, int round);
	//void BGD_a_Train(float eps, int round);

<<<<<<< HEAD
	protected:
	

	private:
	
	void DumpAdaptNN(const string & out_dir, int round, const CriteriaItem & obj_item, const NNet::Transform & tf, bool isbest) const;
	

	std::vector< vector< float > > output_feature;
	void GenOutputFeature(const std::vector< TrainItem > & items);
	CriteriaItem CalcObj(const NNet::Transform & trans);
=======
	private:

	float TotalErr(const NNet::Transform & trans, int round);
>>>>>>> 40429d0bff00def6e25281793c213bde38f8ac2d
};
