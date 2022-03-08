#pragma once
#include "LayerHeader.h"


class PoolingLayer : public BaseLayer
{
public:
	enum POOLING_TYPE {
		MAX,
		MEAN
	};
	PoolingLayer(){}
	~PoolingLayer()
	{
		delete[] bl_output;
		delete[] bl_w;
		delete[] bl_grad;
		delete[] pool_flag;
	}

	PoolingLayer(int input_w, int input_h, int filter_w, int filter_h, POOLING_TYPE type = POOLING_TYPE::MAX, std::string name="");

	virtual void forward();

	virtual void update_gradient();

	virtual void update_weight()
	{
		//Pooling layer does not have weight parameter.
	}

	void print_data();

	POOLING_TYPE get_pooling_type() { return pool_type; }
	int*		get_pool_flag() { return pool_flag; }
	int			get_filter_size() { return pool_filter_size; }

	int get_pooling_output_w() {
		return pool_output_w
			;
	}

	int get_pooling_output_h() {
		return pool_output_h
			;
	}

private:

	int pool_input_w;
	int pool_input_h;
	int pool_output_w;
	int pool_output_h;

	int pool_filter_w;
	int pool_filter_h;
	int pool_filter_size;


	//For pooling type	
	POOLING_TYPE		pool_type;
	int*				pool_flag;


};
