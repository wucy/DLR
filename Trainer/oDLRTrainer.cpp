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
#include <sstream>

#include "../Util/Util.h"
#include "../NNet/NNet.h"


using namespace boost::numeric::ublas;

using std::cin;
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;
using std::ostringstream;

Trainer::CriteriaItem oDLRTrainer::CalcObj(const NNet::Transform & trans)
{
	float obj = 0;
	int right_tot = 0;
	

	int tot = training_items.size();

	cerr << "==============CalcObj==================\n";
	for (int i = 0; i < tot; ++ i)
	{
		const TrainItem & now_item = training_items[i];
		vector< float > posteriors = trans.get_output(output_feature[i]);
		
		int max_id = 0;
		float max = posteriors(0);
		for (int j = 1, lj = posteriors.size(); j < lj; ++ j)
		{
			if (max < posteriors(j))
			{
				max = posteriors(j);
				max_id = j;
			}
		}
		
		if (max_id == now_item.label) right_tot ++;
		//cout << max_id << "\t" << now_item.label << endl;
		obj -= log(posteriors(now_item.label));
		if (i % 1000 == 0) cerr << i << ", ";
	}
	float acc = ((float) right_tot) / tot;
	cerr << "\n=======================================\n";
	cerr << "OBJ=" << obj << "\tACC=" << acc << "\n";
	cerr << "=======================================\n";
	

	Trainer::CriteriaItem ret(acc, obj);
	return ret;
}



void oDLRTrainer::GenOutputFeature(const std::vector< TrainItem > & items)
{
	int tot_layer = nnet.GetTotalLayer();
	cerr << "===Generate OutputFeature===\n";
	for (int i = 0, li = items.size(); i < li; ++ i)
	{
		output_feature.push_back(nnet.GetNLayerOutput(tot_layer - 1, items[i].feature));
		if (i % 1000 == 0) cerr << i << ", ";
	}
	cerr << "\n============================\n";
}


void oDLRTrainer::DumpAdaptNN(const string & out_dir, int round, const CriteriaItem & cri_item, const NNet::Transform & tf, bool isbest) const
{
	ostringstream oss;
	oss << out_dir << "/nnet_adapt_iter" << round << "_acc" << cri_item.acc << "_obj" << cri_item.obj;
	if (isbest) oss << "_final";
	
	string ofn = oss.str();
	ofstream ofs(ofn.c_str());

	cerr << ofn << endl;

	for (int i = 0; i < nnet.GetTotalLayer() - 1; ++ i)
	{
		ofs << nnet.transforms[i].toString();
	}

	ofs << tf.toString();

	ofs.close();
}

void oDLRTrainer::SGD_only_c_Train(float eps, int round, const string & out_dir)
{

	assert(false && "NDEBUG Mode is required.");
	assert(training_items.size() > 0);
	int tot_layer = nnet.GetTotalLayer();
	
	vector< float > c(nnet.transforms[tot_layer - 2].b.size());
	for (int i = 0; i < c.size(); ++ i) c(i) = 0;

	
	NNet::Transform trans(nnet.transforms[tot_layer - 1]);
	

	vector< float > raw_b(trans.b);

	trans.b = prod(trans.W, c) + raw_b;

	vector< float > Wrows[trans.W.size1()];


	
	for (int i = 0; i < trans.W.size1(); ++ i)
	{
		Wrows[i] = row(trans.W, i);
	}

	CriteriaItem obj = CalcObj(trans);
	//ErrItem err(0,0);
	CriteriaItem bestobj = obj;
	NNet::Transform besttf = trans;


	DumpAdaptNN(out_dir, 0, bestobj, besttf, false);

	for (int rd = 1; rd <= round; ++ rd)
	{
		cerr << "#######################################\n";
		cerr << "ROUND=" << rd << "\tEPS=" << eps << endl;

		
		std::vector< int > seq = random_seq(training_items.size());
		for (int i = 0, li = seq.size(); i < li; ++ i)
		{
			int id = seq[i];

			const int now_label = training_items[id].label;
			const vector< float > posteriors = trans.get_output(output_feature[id]);


			vector< float > gred = Wrows[now_label];
			for (int j = 0, lj = posteriors.size(); j < lj; ++ j)
			{
				gred -= posteriors[j] * Wrows[j];
			}
			
			//const float lrate = norm_2(gred) < 1e-15 ? 0 : eps / norm_2(gred);
			
			const float lrate = eps;

			c += lrate * gred;
			trans.b = prod(trans.W, c) + raw_b;
			
			
			
			if (i % 1000 == 0) cerr << i << ", "; 
		}

		cerr << seq.size() << endl;

		obj = CalcObj(trans);
		
		if (bestobj.obj > obj.obj)
		{
			bestobj = obj;
			besttf = trans;
		}
		else
		{
			eps /= 2;
		}

	

		DumpAdaptNN(out_dir, rd, obj, trans, false);
	}
	DumpAdaptNN(out_dir, round + 1, bestobj, besttf, true);
}

void oDLRTrainer::SGD_only_c_all_tied_Train(float eps, int round, const string & out_dir)
{

	assert(false && "NDEBUG Mode is required.");
	assert(training_items.size() > 0);
	int tot_layer = nnet.GetTotalLayer();
	
	vector< float > c(nnet.transforms[tot_layer - 2].b.size());
	for (int i = 0; i < c.size(); ++ i) c(i) = 0;

	
	NNet::Transform trans(nnet.transforms[tot_layer - 1]);
	

	vector< float > raw_b(trans.b);

	trans.b = prod(trans.W, c) + raw_b;

	vector< float > Wrows[trans.W.size1()];


	
	for (int i = 0; i < trans.W.size1(); ++ i)
	{
		Wrows[i] = row(trans.W, i);
	}

	CriteriaItem obj = CalcObj(trans);
	//ErrItem err(0,0);
	CriteriaItem bestobj = obj;
	NNet::Transform besttf = trans;


	DumpAdaptNN(out_dir, 0, bestobj, besttf, false);

	for (int rd = 1; rd <= round; ++ rd)
	{
		cerr << "#######################################\n";
		cerr << "ROUND=" << rd << "\tEPS=" << eps << endl;

		
		std::vector< int > seq = random_seq(training_items.size());
		for (int i = 0, li = seq.size(); i < li; ++ i)
		{
			int id = seq[i];

			const int now_label = training_items[id].label;
			const vector< float > posteriors = trans.get_output(output_feature[id]);


			vector< float > gred = Wrows[now_label];
			for (int j = 0, lj = posteriors.size(); j < lj; ++ j)
			{
				gred -= posteriors[j] * Wrows[j];
			}
			

			//const float lrate = norm_2(gred) < 1e-15 ? 0 : eps / norm_2(gred);
			
			const float lrate = eps;
				
			float sum_gred = 0;
			for (int j = 0, lj = gred.size(); j < lj; ++ j) sum_gred += gred[j];
				
			for (int k = 0, lk = c.size(); k < lk; ++ k)
			{
				c[k] += lrate * sum_gred;
			}
			
			trans.b = prod(trans.W, c) + raw_b;
			
			
			
			if (i % 1000 == 0) cerr << i << ", "; 
		}

		cerr << seq.size() << endl;

		obj = CalcObj(trans);
		
		if (bestobj.obj > obj.obj)
		{
			bestobj = obj;
			besttf = trans;
		}
		else
		{
			eps /= 2;
		}

	

		DumpAdaptNN(out_dir, rd, obj, trans, false);
	}
}
