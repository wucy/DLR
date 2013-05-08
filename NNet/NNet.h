#pragma once


#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <vector>

#include "../Util/Util.h"

using namespace boost::numeric::ublas;

class NNet
{
public:
	NNet(char * fn);
	vector< float > GetNLayerOutputFromM(int m, int n, const vector< float > & input);
	vector< float > GetNLayerOutput(int n, const vector< float > & input);
	int GetTotalLayer() { return transforms.size(); }
	
	enum LayerType
	{
		LINEAR,
		SIGMOID,
		SOFTMAX
	};

	struct Transform
	{
		matrix< float > W;
		vector< float > b;
		int tot_input, tot_output;
		LayerType output_type;

		Transform(){}

		Transform(const matrix< float > & WW, const vector< float > & bb, LayerType ot):W(WW),b(bb),output_type(ot){}
		
		vector< float > get_output(const vector< float > & input) const
		{
			vector< float > now = prod(W, input) + b;
			if (output_type == SIGMOID) now = vector_sigmoid(now);
			else if (output_type == SOFTMAX) now = vector_softmax(now);
			return now;
		}
	};


	std::vector< Transform > transforms;

private:
	void TNetReaderProxy(char * fn);
};

