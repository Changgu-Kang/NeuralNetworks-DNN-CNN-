#include "ConvLayer.h"

ConvLayer::ConvLayer(int input_w, int input_h, int stride, int filter_w, int filter_h, ACT_TYPE act_type, std::string name) :
	conv_input_w(input_w)
	, conv_input_h(input_h)
	, conv_stride(stride)
	, conv_filter_w(filter_w)
	, conv_filter_h(filter_h)
	, conv_filter_size(filter_w * filter_h)
	, conv_padding(0)
{
	bl_name = name;

	conv_output_w = (conv_input_w + 2 * conv_padding - conv_filter_w) / conv_stride + 1;
	conv_output_h = (conv_input_h + 2 * conv_padding - conv_filter_h) / conv_stride + 1;

	bl_input_size = conv_input_w * conv_input_h;
	bl_output_size = conv_output_w * conv_output_h;
	bl_grad_size = bl_output_size;
	bl_w_size = conv_filter_w * conv_filter_h;

	conv_pad_w_size = conv_filter_w - 1;
	conv_pad_h_size = conv_filter_h - 1;

	conv_grad_w = conv_output_w + conv_pad_w_size * 2 + (conv_stride - 1);
	conv_grad_h = conv_output_h + conv_pad_h_size * 2 + (conv_stride - 1);
	conv_grad_size = conv_grad_w * conv_grad_h;

	//bl_input = new double[bl_input_size];
	bl_output = new double[bl_output_size];
	bl_w = new double[bl_w_size + 1];//Add 1 for the weight of bias
	bl_w_delta = new double[bl_w_size + 1];//For momentum

	bl_w_real_size = bl_w_size + 1;

	bl_l_type = LAYER_TYPE::CONV;
	bl_act_type = act_type;
	bl_w_ini_type = INIT_TYPE::XAVIER;


	bl_grad = new double[conv_grad_size];//Add pad size and stride size for backward pass

	for (int i = 0; i < conv_grad_size; i++)
		bl_grad[i] = 0.0;

	// Parameter initialize(xavier)		
	// w is initialized by using xavier method.


	//

	
	reset_weight();

	/*if (bl_w_ini_type == INIT_TYPE::RANDOM)
		iniWrandom(bl_w, bl_w_size);
	else if (bl_w_ini_type == INIT_TYPE::XAVIER)
		iniWxavier(bl_w, bl_w_size, (bl_input_size + bl_output_size) / 2);*/

	bl_w[bl_w_size] = 0.0;
	bl_bias = 1.0;
	bl_next = NULL;
}



void ConvLayer::forward()
{
	parallel_for(0, conv_output_h, [&](size_t o_h)
	//for (int o_h = 0; o_h < conv_output_h; o_h++)
	{
		for (int o_w = 0; o_w < conv_output_w; o_w++)
		{
			int o_index = o_h * conv_output_w + o_w;
			bl_output[o_index] = 0;

			int i_w_start = o_w * conv_stride;
			int i_h_start = o_h * conv_stride;

			int max_index = -1;
			double max_value = -0xffffff;

			for (int f_h = 0; f_h < conv_filter_h; f_h++)
			{
				for (int f_w = 0; f_w < conv_filter_w; f_w++)
				{
					int f_index = f_h * conv_filter_w + f_w;
					int i_index = (f_h + i_h_start) * conv_input_w + (f_w + i_w_start);

					if (isnan(bl_input[i_index]) || isinf(bl_input[i_index]))
					{
						printf("Input Nan break %s\n",get_name().c_str());
					}

					if (isnan(bl_w[f_index]) || isinf(bl_w[f_index]))
					{
						printf("Weight Nan break %s\n", get_name().c_str());
					}


					bl_output[o_index] += (bl_w[f_index] * bl_input[i_index]);

					if (isnan(bl_output[o_index]) || isinf(bl_output[o_index]))
					{
						printf("Output Nan break %s\n", get_name().c_str());
					}

				}
			}

			if (isnan(bl_output[o_index]) || isinf(bl_output[o_index]))
			{
				printf("output Nan break %s\n", get_name().c_str());
			}

			bl_output[o_index] += (bl_w[bl_w_size] * bl_bias);

			if (isnan(bl_output[o_index]) || isinf(bl_output[o_index]))
			{
				printf("output Nan break %s\n", get_name().c_str());
			}

			double pre_output = bl_output[o_index];

			bl_output[o_index] = get_act_value(bl_output[o_index], this->get_act_type());

			if (isnan(bl_output[o_index]) || isinf(bl_output[o_index]))
			{
				printf("output Nan break %f -->  %f %s", pre_output, bl_output[o_index], get_name().c_str());
				printf("\n");
			}

		}
	}
	);

}

//
int ConvLayer::conv_grad_index(int x, int y)
{
	int g_x = conv_grad_index_x(x);
	int g_y = conv_grad_index_y(y);
	return g_x + g_y * conv_grad_w;
}

int ConvLayer::conv_grad_index_x(int x)
{
	return conv_pad_w_size + x * conv_stride;
}

int ConvLayer::conv_grad_index_y(int y)
{
	return conv_pad_h_size + y * conv_stride;
}
//


//For easily computing gradient of pre-layer.
double ConvLayer::get_conv_grad(int input_x, int input_y)
{
	double grad = 0.0;


	for (int y = 0; y < conv_filter_h; y++)
	{
		for (int x = 0; x < conv_filter_w; x++)
		{
			int w_idx = (conv_filter_h - y - 1) * conv_filter_w + (conv_filter_w - x - 1);//get weight with inverse index.
																						  //int g_idx = (conv_filter_h - y - 1 + input_y) * conv_grad_w + (conv_filter_w - x - 1) + input_x;
			int g_idx = (y + input_y) * conv_grad_w + (input_x + x);

			if (isnan(bl_grad[g_idx]) || isinf(bl_grad[g_idx]))
			{
				printf("my bl grad break: %f %s\n", bl_grad[g_idx], get_name().c_str());
				printf("\n");
			}

			if (isnan(bl_w[w_idx]) || isinf(bl_w[w_idx]))
			{
				printf("my weight break: %f %s\n", bl_grad[g_idx], get_name().c_str());
				printf("\n");
			}

			grad += bl_grad[g_idx] * bl_w[w_idx];

			if (isnan(grad) || isinf(grad))
			{
				printf("bl grad break: %f %s\n", grad, get_name().c_str());
				printf("\n");
			}
		}
	}


	return grad;
}

void ConvLayer::update_gradient()
{
	parallel_for(0, conv_output_h, [&](size_t y)
	//for (int y = 0; y < conv_output_h; y++)
	{
		for (int x = 0; x < conv_output_w; x++)
		{
			int o_idx = y * conv_output_w + y;
			int g_idx = conv_grad_index(x, y);

			bl_grad[g_idx] = 0.0f;
			switch (bl_next->get_layer_type())
			{
			case LAYER_TYPE::FC:
			case LAYER_TYPE::OUTPUT:
				{
					for (int j = 0; j < bl_next->get_grad_size(); j++)
					{
						double next_grad = bl_next->get_grad()[j];
						double next_w = bl_next->get_w()[(j * bl_next->get_input_size()) + (x + y * conv_output_w)];
						if (isnan(next_grad) || isinf(next_grad))
						{
							printf("next bl grad break: %f\n", bl_next->get_grad()[j]);
						}

						if (isnan(next_w) || isinf(next_w))
						{
							printf("next bl grad break: %f\n", bl_next->get_grad()[j]);
						}


						bl_grad[g_idx] += next_grad * next_w;

						if (isnan(bl_grad[g_idx]) || isinf(bl_grad[g_idx]))
						{
							printf("bl grad break: %f\n", bl_grad[g_idx]);
						}
					}
					break;	
				}
				
			case LAYER_TYPE::CONV:
				{
					ConvLayer * conv = dynamic_cast<ConvLayer*>(bl_next);
					bl_grad[g_idx] = conv->get_conv_grad(x, y);
					break;
				}

			case LAYER_TYPE::LINK:
				{
					LinkLayer * link = dynamic_cast<LinkLayer*>(bl_next);
					for (int j = 0; j < link->get_grad_size(); j++)
					{
						bl_grad[g_idx] += (link->get_grad()[j] * link->get_w(dynamic_cast<BaseLayer*>(this), o_idx, j * link->get_input_size()));

					}
					break;
				}
				
				
			case LAYER_TYPE::POOLING:
				{
					PoolingLayer * pool = dynamic_cast<PoolingLayer*>(bl_next);

					int pool_idx = x + y * conv_output_w;

					if (pool->get_pooling_type() == PoolingLayer::POOLING_TYPE::MAX)
					{
						
						//only MAX INPUT
						if (pool->get_pool_flag()[pool_idx] != -1)
							bl_grad[g_idx] = pool->get_grad()[pool->get_pool_flag()[pool_idx]];
					}
					else if (pool->get_pooling_type() == PoolingLayer::POOLING_TYPE::MEAN)
					{
						bl_grad[g_idx] = pool->get_grad()[pool->get_pool_flag()[pool_idx]] / pool->get_filter_size();
					}


					if (isnan(bl_grad[g_idx]) || isinf(bl_grad[g_idx]))
					{
						printf("bl grad break: %f %f %f %s\n", bl_grad[g_idx], pool->get_pool_flag()[pool_idx], pool->get_grad()[pool->get_pool_flag()[pool_idx]], get_name().c_str());
						printf("\n");
					}

					

					break;
				}
				
			}

			double pre_grad = bl_grad[g_idx];

			bl_grad[g_idx] *= get_act_grad(bl_output[o_idx], this->get_act_type());


			if (isnan(bl_grad[g_idx]) || isinf(bl_grad[g_idx]))
			{
				printf("bl grad break: %f --> %f %s\n", pre_grad, bl_grad[g_idx], get_name().c_str());
			}



		}
	}
	);
}

void ConvLayer::update_weight()
{
	for (int i = 0; i < bl_w_size + 1; i++)
	{
		double delta = 0.0;

		int y = (int)floor(i / conv_filter_w);
		int x = i % conv_filter_w;

		int start_idx = y * conv_input_w + x;

		for (int h = 0; h < conv_output_h; h++)
		{
			for (int w = 0; w < conv_output_w; w++)
			{
				int g_idx = conv_grad_index(w, h);
				if (i < bl_w_size)
				{
					int input_idx = start_idx + w * conv_stride + h * conv_stride * conv_input_w;

					if (isnan(bl_input[input_idx]) || isinf(bl_input[input_idx]))
					{
						printf("bl input break: %f %s\n", bl_input[input_idx], get_name().c_str());
					}

					if (isnan(bl_grad[g_idx]) || isinf(bl_grad[g_idx]))
					{
						printf("bl grad break: %f %s\n", bl_grad[g_idx],get_name().c_str());
					}


					delta += bl_input[input_idx] * bl_grad[g_idx];

					if (isnan(delta))
					{
						printf("delta break: %f %f %s\n", delta, bl_grad[g_idx], get_name().c_str());
					}
				}
				else //Compute the weight of bias.
				{
					if (isnan(bl_grad[g_idx]) || isinf(bl_grad[g_idx]))
					{
						printf("bl grad break: %f %s\n", bl_grad[g_idx], get_name().c_str());
					}

					delta += bl_bias * bl_grad[g_idx];
				}
			}
		}


		//GD + momentum
		//double v_d = momentum_rate* bl_w_delta[i] + (1.0 - momentum_rate)*delta;		
		//bl_w[i] += (-learning_rate*v_d);
		//bl_w_delta[i] = v_d;

		double pre = bl_w_delta[i];
		bl_w_delta[i] = momentum_rate * bl_w_delta[i] - learning_rate * delta;
		bl_w[i] += (-momentum_rate * pre + (1.0+ momentum_rate) * bl_w_delta[i]);

		



		if (isnan(bl_w[i]) || isinf(bl_w[i]))
		{
			printf("W update break\n");
		}
	}
}


