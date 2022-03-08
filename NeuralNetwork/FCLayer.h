#pragma once

#include "LayerHeader.h"

class FCLayer : public BaseLayer {
public:
	FCLayer() {}
	FCLayer(int input_size, int output_size, LAYER_TYPE type, ACT_TYPE act_type=ACT_TYPE::LRELU, std::string name="");

	~FCLayer()
	{
		delete [] bl_w;
		delete[] bl_grad;
		delete[] bl_output;
	}	

	
	virtual void forward();
	virtual void update_gradient();
	virtual void update_weight();
};