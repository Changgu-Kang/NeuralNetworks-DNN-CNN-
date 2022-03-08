#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <chrono>
#include <fstream>
#include <math.h>

enum ACT_TYPE {
	SIGMOID,
	RELU,
	LRELU,
	TANH,
	LOGISTIC
};

enum LAYER_TYPE {
	LINK,
	OUTPUT,
	FC,
	CONV,
	POOLING
};

enum INIT_TYPE {
	XAVIER,
	RANDOM
};


static double learning_rate = 0.1;
static double momentum_rate = 0.9;

inline double act_sigmoid(double v) {
	if (v < -6) return 0;
	else if (v > 6) return 1;
	else
		return 1.0 / (1.0 + exp(-v)); 
}
inline double act_relu(double v) { return 0.0 > v ? 0.0 : v; }
inline double act_lrelu(double v) { return 0.0 > v ? 0.01*v : v; }
inline double act_tanh(double v) {

	double result;

	if (v < -3.0f) return result = -1;
	else if (v > 3.0f) return result = 1;
	else result = (exp(v) - exp(-v)) / (exp(v) + exp(-v));

	if (result > 1) return 1;
	else if(result < -1) return -1;
	else return result;

}
inline double act_logistic(double v) { return 1.0 / (1 + exp(-v)); }

inline double act_grad_sigmoid(double v) { return (1.0 - v) * v; }
inline double act_grad_relu(double v) { if (v > 0.0) return 1.0; else return 0.0; }
inline double act_grad_lrelu(double v) { if (v > 0.0) return 1.0; else return 0.01; }
inline double act_grad_tanh(double v) { return 1 - pow(tanh(v), 2.0); }
inline double act_grad_logistic(double v) { return act_logistic(v)*(1.0 - act_logistic(v)); }

inline double get_act_value(double v, ACT_TYPE type)
{
	switch (type)
	{
	case ACT_TYPE::SIGMOID:
		return act_sigmoid(v);
	case ACT_TYPE::RELU:
		return act_relu(v);
	case ACT_TYPE::LRELU:
		return act_lrelu(v);
	case ACT_TYPE::TANH:
		return act_tanh(v);
	case ACT_TYPE::LOGISTIC:
		return act_logistic(v);
	}
}

inline double get_act_grad(double v, ACT_TYPE type)
{
	switch (type)
	{
	case ACT_TYPE::SIGMOID:
		return act_grad_sigmoid(v);
	case ACT_TYPE::RELU:
		return act_grad_relu(v);
	case ACT_TYPE::LRELU:
		return act_grad_lrelu(v);
	case ACT_TYPE::TANH:
		return act_grad_tanh(v);
	case ACT_TYPE::LOGISTIC:
		return act_grad_logistic(v);
	}
}

void iniWrandom(double* w, int w_size);
void iniWxavier(double* w, int w_size, int n_size);

class BaseLayer {
public:
	BaseLayer() {}
	virtual void forward() = 0;
	virtual void update_gradient() = 0; //Duing backpropagation
	virtual void update_weight() = 0; //Duing backpropagation

	LAYER_TYPE get_layer_type() { return bl_l_type; }
	double*	get_w() { return bl_w; }
	double* get_grad() { return bl_grad; }
	int		get_w_size() { return bl_w_size; }
	int		get_grad_size() { return bl_grad_size; }
	ACT_TYPE get_act_type() { return bl_act_type; }

	int		get_output_size() { return bl_output_size; }
	double* get_output_ptr() { return bl_output; }

	int		get_input_size() { return bl_input_size; }

	void connect_next_layer(BaseLayer* next_layer) {
		bl_next = next_layer;
		next_layer->connect_pre_output_to_my_input(bl_output);
	}



	// Share the storage of input with the output of pre layer.
	void	connect_pre_output_to_my_input(double* pre_output)
	{
		bl_input = pre_output;
	}

	void set_input_ptr(double* x)
	{
		bl_input = x;
	}

	void set_true_y_ptr(double* y)
	{
		bl_true_y = y;
	}

	int get_w_real_size() { return bl_w_real_size; }

	void reset_weight()
	{
		for (int i = 0; i < bl_w_real_size; i++)
		{
			bl_w[i] = 0.0;
			bl_w_delta[i] = 0.0;
		}

		if (bl_w_ini_type == INIT_TYPE::RANDOM)
			iniWrandom(bl_w, bl_w_size);
		else if (bl_w_ini_type == INIT_TYPE::XAVIER)
			iniWxavier(bl_w, bl_w_size, (bl_input_size + bl_output_size) / 2);


	}

	std::string get_name() { return bl_name; }
	std::string set_name(std::string name) { bl_name = name; }

protected:

	std::string bl_name;

	double* bl_grad;
	double* bl_output;
	double* bl_input;
	double* bl_w;
	double* bl_w_delta;
	double	bl_bias;

	double* bl_true_y;


	ACT_TYPE	bl_act_type;
	LAYER_TYPE	bl_l_type;
	INIT_TYPE	bl_w_ini_type;

	int		bl_input_size;
	int		bl_output_size;
	int		bl_w_size;
	int		bl_grad_size;

	int		bl_w_real_size;

	BaseLayer*	bl_next;
};