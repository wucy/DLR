//#define NDEBUG

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>

#include "Util/Util.h"
#include "NNet/NNet.h"

using namespace boost::numeric::ublas;

using std::cin;
using std::cout;
using std::cerr;
using std::endl;

using std::ifstream;
using std::ofstream;

void test_Util();
void test_NNet();
void test_NNet_inst();

int main()
{
	//cerr << RAND_MAX << endl;
#ifdef NDEBUG
	//cerr << "NDEBUG is defined" << endl;
#endif
	//test_Util();
	test_NNet_inst();
	//test_NNet();
	//test_Util();
	return 0;
}


void test_Util()
{
	vector< float > vect(4);
	for (int i = 0; i < vect.size(); ++ i) vect(i) = i;
	cerr << "RAW:\t" << vect << endl;
	cerr << "SIGMOID:\t" << vector_sigmoid(vect) << endl;
	cerr << "SOFTMAX:\t" << vector_softmax(vect) << endl;
	std::vector< int > seq = random_seq(10000);
	//for (int i = 0; i < 100; ++ i) cerr << seq[i] << "\t"; cerr << endl;
}

void test_NNet()
{
	NNet nnet("testcase/nn_full.tc");
	//cerr << nnet.transforms[0].W << endl;
	//cerr << nnet.transforms[0].b << endl;
	vector< float > v(429);
	for (int i = 0; i < 429; i ++) v(i) = i;
	cerr << "LOAD FINISHED\n";
	
	cout << nnet.GetNLayerOutput(3, v) << endl;
	vector< float > u = nnet.GetNLayerOutputFromM(0, 2, v);
	cout << nnet.GetNLayerOutputFromM(2, 3, u);
}

void test_NNet_inst()
{
	NNet nnet("/home/slhome/cyw56/workdir/slfs3/baseline_qyzj/dnn/backprop/tr_L0_L1_L2/weights/nnet_tr_L0_L1_L2_final_iters13_tr63.145_cv60.871");
	ifstream ifs("testcase/fea2.in");
	vector< float > v(429);
	for (int i =0 ; i < 429; ++ i) ifs >> v(i);
	ifs.close();
	ofstream ofs("testcase/fea2.pred");
	vector< float > out_fea=nnet.GetNLayerOutput(3, v);
	NNet::Transform last(nnet.transforms[3]);
	zero_vector < float > c(out_fea.size());
	last.b = prod(last.W, c) + last.b;
	vector< float > out = last.get_output(out_fea);
	int id = 0; float max = out(0);
	for (int i = 1; i < out.size(); i ++)
	{
		if (max < out(i))
		{
			max = out(i);
			id = i;
		}
	}
	ofs.close();
}
