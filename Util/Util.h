#pragma once

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <cmath>

using namespace boost::numeric::ublas;

extern vector< float > vector_sigmoid(const vector< float > & vect);
extern vector< float > vector_softmax(const vector< float > & vect);
