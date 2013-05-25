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


void Trainer::InputProxy(char * fn_lab, char * fn_fea, int fea_dim, int sil_use_mode)
{
	ifstream ifs_fea(fn_fea);
	ifstream ifs_lab(fn_lab);
	int label;
	while (ifs_lab >> label)
	{
		vector< float > fea(fea_dim);
		for (int j = 0; j < fea_dim; ++ j) ifs_fea >> fea(j);
	
		if (sil_use_mode == 1)
		{
			if (label == 1560) continue;
		}
		else if (sil_use_mode == 2)
		{
			if (label <= 1562 && label >= 1560) continue;
		}
			
		TrainItem ti(label, fea);
		training_items.push_back(ti);
	}
	ifs_fea.close();
	ifs_lab.close();
	cerr << "Finish reading features.[" << training_items.size() << "]" << endl;
}
