#pragma once


#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <vector>

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
		matrix < double > W;
		vector < float > b;
		int tot_input, tot_output;
		LayerType output_type;
	};


	std::vector< Transform > transforms;

private:
	void TNetReaderProxy(char * fn);
};

