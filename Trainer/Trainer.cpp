#include "Trainer.h"

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>


#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <cassert>

#include "../Util/Util.h"
#include "../NNet/NNet.h"


using namespace boost::numeric::ublas;

using std::cin;
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;


void Trainer::InputProxy(char * fn)
{
	ifstream ifs(fn);

	int dim, tot;
	ifs >> dim >> tot;
	for (int i = 0; i < tot; ++ i)
	{
		int label;
		vector< float > fea(dim);
		ifs >> label;
		for (int j = 0; j < dim; ++ j) ifs >> fea(j);
		TrainItem ti(label, fea);
		training_items.push_back(ti);
	}

	ifs.close();
}
