#pragma once
#include "LayerHeader.h"

class ConvLayer : public BaseLayer {
public:

	ConvLayer() {}
	~ConvLayer() 
	{
		delete[] bl_output;
		delete[] bl_w;
		delete[] bl_grad;
	}
	ConvLayer(int input_w, int input_h, int stride, int filter_w, int filter_h, ACT_TYPE act_type = ACT_TYPE::TANH, std::string name="");

	
	virtual void forward();
	//
	int conv_grad_index(int x, int y);

	int conv_grad_index_x(int x);

	int conv_grad_index_y(int y);

	//For easily computing gradient of pre-layer.
	double get_conv_grad(int input_x, int input_y);

	virtual void update_gradient();

	virtual void update_weight();

	int get_conv_output_w() {
		return conv_output_w
			;
	}

	int get_conv_output_h() {
		return conv_output_h
			;
	}

private:

	int conv_input_w;
	int conv_input_h;	
	int conv_output_w;
	int conv_output_h;
	
	int conv_stride;
	int conv_filter_w;
	int conv_filter_h;
	int conv_filter_size;
	int conv_padding;

	//For deconvolution at backward pass.
	int	conv_pad_w_size;
	int	conv_pad_h_size;
	int conv_grad_w;
	int conv_grad_h;
	int conv_grad_size;	
};
