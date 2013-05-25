#include "Util.h"

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <cmath>
#include <vector>

#include <cstdlib>
#include <ctime>



using namespace boost::numeric::ublas;
using std::exp;

vector< float > vector_sigmoid(const vector< float > & vect)
{
	vector< float > ret(vect.size());
	for (int i = 0; i < vect.size(); ++ i)
	{
		ret(i) = 1 / (1 + exp(-vect(i)));
	}
	return ret;
}

vector< float > vector_softmax(const vector< float > & vect)
{
	vector< float > ret(vect.size());
	float regulizer = 0;
	float max = vect(0);
	float min = vect(0);
	for (int i = 1; i < vect.size(); ++ i) 
	{
		if (vect(i) > max) max = vect(i);
		if (vect(i) < min) min = vect(i);
	}
	for (int i = 0; i < vect.size(); ++ i) regulizer += exp(vect(i)-max);
	//std::cout << regulizer << "\t" << max << "\t" << min << std::endl;
	for (int i = 0; i < vect.size(); ++ i) ret(i) = exp(vect(i)-max) / regulizer;
	return ret;
}


std::vector< int > random_seq(int n)
{
	srand(time(NULL)^1357924680);
	std::vector< int > ret;
	for (int i = 0; i < n; ++ i) ret.push_back(i);
	for (int step = 0; step < n * 5; ++ step)
	{
		int id1 = rand() % n, id2 = rand() % n;
		int tmp = ret[id1];
		ret[id1] = ret[id2];
		ret[id2] = tmp;
	}
	return ret;
}


