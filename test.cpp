//#define NDEBUG

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <iostream>
#include <fstream>

#include "Util/Util.h"
#include "NNet/NNet.h"

using namespace boost::numeric::ublas;

using std::cin;
using std::cout;
using std::cerr;
using std::endl;

void test_Util();
void test_NNet();

int main()
{
#ifdef NDEBUG
	//cerr << "NDEBUG is defined" << endl;
#endif
	test_NNet();
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
