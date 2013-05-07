#include "Util.h"

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <cmath>

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
	for (int i = 0; i < vect.size(); ++ i) regulizer += exp(vect(i));
	for (int i = 0; i < vect.size(); ++ i) ret(i) = exp(vect(i)) / regulizer;
	return ret;
}
