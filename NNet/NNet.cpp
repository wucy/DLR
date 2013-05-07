#include "NNet.h"
#include "../Util/Util.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include<iostream>
#include<fstream>

using namespace boost::numeric::ublas;

using std::string;
using std::ifstream;

NNet::NNet(char * fn)
{
	TNetReaderProxy(fn);
}

void NNet::TNetReaderProxy(char * fn)
{

	ifstream ifs(fn);
	string ltype;
	string tmp;
	int tot_out, tot_in;
	while (ifs >> ltype >> tot_out >> tot_in)
	{
		assert(ltype == "<biasedlinearity>");
		Transform tf;
		tf.tot_input = tot_in;
		tf.tot_output = tot_out;
		
		//NEXT(matrix input) m, height, weight;
		ifs >> tmp >> tot_out >> tot_in;
		assert(tmp != "m" || tot_out != tf.tot_output || tot_in != tf.tot_input);
		
		matrix< float > W(tot_out, tot_in);
		for (int i = 0; i < tot_out; ++ i)
			for (int j = 0; j < tot_in; ++ j)
				ifs >> W(i, j);
		
		//NEXT(vect input) v, height;
		ifs >> tmp >> tot_out;
		assert(tmp != "v" || tot_out != tf.tot_output);


		vector< float > b(tot_out);
		for (int i = 0; i < tot_out; ++ i)
			ifs >> b(i);


		//NEXT(output type) type, out, in;
		ifs >> tmp >> tot_out >> tot_in;
		assert(tot_out != tf.tot_output || tot_in != tf.tot_input);

		if (tmp == "<sigmoid>") tf.output_type = SIGMOID;
		else if (tmp == "<softmax>") tf.output_type = SOFTMAX;
		else {
			std::cerr << "parse failed: " << tmp << std::endl; 
			exit(-1);
		}

		tf.W = W;
		tf.b = b;

		this->transforms.push_back(tf);
	}
	ifs.close();
}

vector< float > NNet::GetNLayerOutput(int n, vector< float > input)
{
	assert(n >= this->transforms.size());

	vector< float > now(input);

	for (int i = 0; i < n; i ++)
	{
		now = prod(transforms[i].W, now);
		now += transforms[i].b;
		
		if (transforms[i].output_type == SIGMOID) vector_sigmoid(now);
		else if (transforms[i].output_type == SOFTMAX) vector_softmax(now);
	}

	return now;
}
