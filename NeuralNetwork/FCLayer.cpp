#include "FCLayer.h"



FCLayer::FCLayer(int input_size, int output_size, LAYER_TYPE type,ACT_TYPE act_type, std::string name)
{
	bl_name = name;

	bl_input_size = input_size;
	bl_output_size = output_size;
	bl_w_size = bl_input_size * bl_output_size;
	bl_grad_size = bl_output_size;

	bl_w = new double[bl_w_size + bl_output_size];
	bl_w_delta = new double[bl_w_size + bl_output_size];
	bl_grad = new double[bl_output_size];
	bl_output = new double[bl_output_size];

	bl_l_type = type;
	bl_act_type = act_type;
	bl_w_ini_type = INIT_TYPE::XAVIER;

	bl_w_real_size = bl_w_size + bl_output_size;

	reset_weight();

	bl_bias = 1.0;
	bl_next = NULL;
}



void FCLayer::forward()
{

	parallel_for(0, bl_output_size, [&](size_t i)	
	//for (int i = 0; i < bl_output_size; i++)
	{
		bl_output[i] = 0.0;
		for (int j = 0; j < bl_input_size; j++)
		{
			if (
				isnan(bl_input[j]) || isinf(bl_input[j])
				|| bl_input[j] < -10000
				|| bl_input[j] > 10000
				)
			{
				printf("%s: Input Nan break\n", get_name().c_str());
			}

			if (isnan(bl_w[i*bl_input_size + j]) || isinf(bl_w[i*bl_input_size + j]))
			{
				printf("Weight Nan %s\n", get_name().c_str());
			}


			bl_output[i] += bl_w[i*bl_input_size + j] * bl_input[j];

			if (
				isnan(bl_output[i]) || isinf(bl_output[i])
				|| bl_output[i] < -10000
				|| bl_output[i] > 10000
				)
			{
 				printf("Output Nan break %s\n",get_name().c_str());
			}

			if (this->get_name().compare("LE") == 0)
			{
				//printf("%f ", bl_output[i]);
			}

			
		}
		bl_output[i] += bl_w[bl_w_size + i] * bl_bias;

		double pre = bl_output[i];

		
		bl_output[i] = get_act_value(bl_output[i], bl_act_type);

		if (
			bl_act_type == ACT_TYPE::TANH
			&&
			(
				bl_output[i] > 1.0
				|| bl_output[i] < -1.0

				)
			)
		{
			printf("act result error: %f -> %f (%d)\n", pre, bl_output[i], bl_act_type);
		}

		if (
			isnan(bl_output[i]) || isinf(bl_output[i])
			|| bl_output[i] < -10000
			|| bl_output[i] > 10000
			)
		{
			printf("%s bl output break: %f -> %f (%d)\n", this->get_name().c_str(),pre,bl_output[i], bl_act_type);
		}

		if (this->get_name().compare("LE") == 0)
		{
			//printf("%f ", bl_output[i]);
		}

	}
	);


}

void FCLayer::update_gradient()
{
	parallel_for(0, bl_output_size, [&](size_t i)
	//for (int i = 0; i < bl_output_size; i++)
	{
		bl_grad[i] = 0.0f;

		if (bl_l_type == LAYER_TYPE::OUTPUT)
		{
			bl_grad[i] = ((bl_output[i]- bl_true_y[i]) * get_act_grad(bl_output[i], bl_act_type))
							/ (bl_output_size * 2.0);

			if (
				isnan(bl_grad[i]) || isinf(bl_grad[i])
				|| bl_grad[i] < -10000
				|| bl_grad[i] > 10000
				)
			{
				printf("%s bl (update)grad break: %f\n", this->get_name().c_str(), bl_grad);
			}

		}
		else
		{
			switch (bl_next->get_layer_type())
			{
				case LAYER_TYPE::OUTPUT:
				case LAYER_TYPE::FC:
				{
					for (int j = 0; j < bl_next->get_grad_size(); j++)
					{
						bl_grad[i] += (bl_next->get_grad()[j] * bl_next->get_w()[(j * bl_next->get_input_size()) + i]);

					}
					break;
				}
				
				case LAYER_TYPE::CONV://Can't connect?
				{
					ConvLayer * conv = dynamic_cast<ConvLayer*>(bl_next);
					bl_grad[i] = conv->get_conv_grad(i, 0);
					break;
				}
				
				case LAYER_TYPE::LINK:
				{
					LinkLayer * link = dynamic_cast<LinkLayer*>(bl_next);
					for (int j = 0; j < link->get_grad_size(); j++)
					{
						bl_grad[i] += (link->get_grad()[j] * link->get_w(dynamic_cast<BaseLayer*>(this), i, j * link->get_input_size()));

					}
					break;
				}
				
			}
			bl_grad[i] *= get_act_grad(bl_output[i], bl_act_type);
		}
	}
	);

	if (bl_l_type == LAYER_TYPE::OUTPUT)
	{
		double loss = 0.0;

		for (int i = 0; i < bl_output_size; i++)
		{
			loss += bl_grad[i];
		}

		for (int i = 0; i < bl_output_size; i++)
		{
			bl_grad[i] = loss;
		}
	}
}

void FCLayer::update_weight()
{
	parallel_for(0, bl_w_size + bl_output_size, [&](size_t i)
	//for (int i = 0; i < bl_w_size + bl_output_size; i++)
	{
		double delta = 0.0;
		int input_idx = i % bl_input_size;
		int grad_idx = (int)floor(i / bl_input_size);


		
		if (i < bl_w_size)
			delta = bl_grad[grad_idx] * bl_input[input_idx];
		else
			delta = bl_grad[i - bl_w_size] * bl_bias;//update bias weight		

		//bl_w[i] += (-learning_rate * delta);

		//GD + momentum
		double v_d = momentum_rate * bl_w_delta[i] + (1.0 - momentum_rate)*delta;
		bl_w[i] += (-learning_rate*v_d);		
		bl_w_delta[i] = v_d;

		//double pre = bl_w_delta[i];
		//bl_w_delta[i] = momentum_rate * bl_w_delta[i] - learning_rate * delta;
		//bl_w[i] += (-momentum_rate * pre + (1.0 + momentum_rate) * bl_w_delta[i]);



	}
	);
}