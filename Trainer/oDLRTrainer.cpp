#include "oDLRTrainer.h"


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



float oDLRTrainer::TotalErr(const NNet::Transform & trans, int round)
{
	int tot_layer = nnet.GetTotalLayer();
	float err = 0;
	for (int i = 0; i < training_items.size(); ++ i)
	{
		const TrainItem & now_item = training_items[i];
		const vector< float > o_fea = nnet.GetNLayerOutput(tot_layer - 1, now_item.feature);
		const vector< float > posteriors = trans.get_output(o_fea);

		const vector< float > df = posteriors - unit_vector< float >(posteriors.size(), now_item.label);
		float loss = inner_prod(df, df);

		err += loss;
	}
	cerr << "=======================================\n";
	cerr << "ROUND=" << round << "\tLOSS=" << err << "\n";
	cerr << "=======================================\n";
	return err;
}


oDLRTrainer::Result oDLRTrainer::SGD_only_c_Train(float eps, int round)
{
	assert(training_items.size() > 0);
	Result result;
	vector< float > c(training_items[0].feature.size());
	for (int i = 0; i < c.size(); ++ i) c(i) = 1;

	int tot_layer = nnet.GetTotalLayer();

	for (int rd = 0; rd < round; ++ rd)
	{
		std::vector< int > seq = random_seq(training_items.size());
		for (int i = 0; i < seq.size(); ++ i)
		{
			NNet::Transform trans(nnet.transforms[tot_layer - 1]);
			trans.b += prod(trans.W, c);

			const TrainItem & now_item = training_items[seq[i]];

			const vector< float > o_fea = nnet.GetNLayerOutput(tot_layer - 1, now_item.feature);
			const vector< float > posteriors = trans.get_output(o_fea);

			const float lrate = eps / norm_2(c);
			
			vector< float > gred = row(trans.W, now_item.label);
			for (int j = 0; j < c.size(); ++ j)
			{
				gred -= posteriors[j] * row(trans.W, j);
			}
			
			c += lrate * gred;
		}
	}
	result.result_type = 0;
	result.c = c;
	return result;
}
