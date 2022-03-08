#include "PoolingLayer.h"

PoolingLayer::PoolingLayer(int input_w, int input_h, int filter_w, int filter_h, POOLING_TYPE type, std::string name) :
	pool_input_w(input_w)
	, pool_input_h(input_h)
	, pool_filter_w(filter_w)
	, pool_filter_h(filter_h)
	, pool_filter_size(filter_w * filter_h)
	, pool_type(type)
{
	bl_name = name;

	pool_output_w = (pool_input_w / pool_filter_w);
	pool_output_h = (pool_input_h / pool_filter_h);


	if (pool_input_w%pool_filter_w != 0 || pool_input_h % pool_filter_h != 0)
	{
		//ERROR
		printf("The size is mis-matched\n");
		exit(-1);
	}

	bl_input_size = pool_input_w * pool_input_h;
	bl_output_size = pool_output_w * pool_output_h;

	bl_grad_size = bl_output_size;

	bl_l_type = LAYER_TYPE::POOLING;

	bl_grad = new double[bl_grad_size];


	bl_output = new double[bl_output_size];
	pool_flag = new int[bl_input_size];

	bl_next = NULL;
}


void PoolingLayer::forward()
{
	for (int i = 0; i < bl_input_size; i++)
		pool_flag[i] = -1;


	for (int o_h = 0; o_h < pool_output_h; o_h++)
	{
		for (int o_w = 0; o_w < pool_output_w; o_w++)
		{
			int o_index = o_h * pool_output_w + o_w;

			bl_output[o_index] = 0;

			int i_w_start = o_w * pool_filter_w;
			int i_h_start = o_h * pool_filter_h;

			int max_index = -1;
			double max_value = -0xffffff;

			for (int f_h = 0; f_h < pool_filter_h; f_h++)
			{
				for (int f_w = 0; f_w < pool_filter_w; f_w++)
				{
					int i_index = (f_h + i_h_start) * pool_input_w + (f_w + i_w_start);

					switch (pool_type)
					{
					case POOLING_TYPE::MAX:
						if (max_value < bl_input[i_index])
						{
							max_value = bl_input[i_index];
							max_index = i_index;
						}
						break;
					case POOLING_TYPE::MEAN:
						bl_output[o_index] += bl_input[i_index];
						pool_flag[i_index] = o_index;
						break;
					}
				}
			}

			switch (pool_type)
			{
			case POOLING_TYPE::MAX:
				bl_output[o_index] = max_value;
				pool_flag[max_index] = o_index;
				break;
			case POOLING_TYPE::MEAN:
				bl_output[o_index] /= pool_filter_size;
				break;
			}
		}
	}


	//print_data();

	//printf("\n");
}


void PoolingLayer::update_gradient()
{
	for (int y = 0; y < pool_output_h; y++)
	{
		for (int x = 0; x < pool_output_w; x++)
		{
			int g_idx = x + pool_output_w * y;

			bl_grad[g_idx] = 0.0f;
			switch (bl_next->get_layer_type())
			{
				case LAYER_TYPE::FC:
				case LAYER_TYPE::OUTPUT:
					for (int j = 0; j < bl_next->get_grad_size(); j++)
						bl_grad[g_idx] += bl_next->get_grad()[j] * bl_next->get_w()[(j * bl_output_size) + (x + y * pool_output_w)];
					break;
				case LAYER_TYPE::CONV:
					{
						ConvLayer * conv = (ConvLayer*)(bl_next);
						bl_grad[g_idx] = conv->get_conv_grad(x, y);
						break;
					}
				
				case LAYER_TYPE::POOLING:
					{
						PoolingLayer * pool = dynamic_cast<PoolingLayer*>(bl_next);
						if (pool->get_pooling_type() == POOLING_TYPE::MAX)
						{
							//only MAX INPUT
							if (pool->get_pool_flag()[g_idx] != -1)
								bl_grad[g_idx] = pool->get_grad()[pool->get_pool_flag()[g_idx]];
						}
						else if (pool->get_pooling_type() == POOLING_TYPE::MEAN)
						{
							bl_grad[g_idx] = pool->get_grad()[pool->get_pool_flag()[g_idx]] / pool->get_filter_size();
						}
						break;
					}
				case LAYER_TYPE::LINK:
					LinkLayer * link = dynamic_cast<LinkLayer*>(bl_next);
					for (int j = 0; j < link->get_grad_size(); j++)
					{
						bl_grad[g_idx] += (link->get_grad()[j] * link->get_w(dynamic_cast<BaseLayer*>(this), g_idx, j * link->get_input_size()));

					}
					break;
			}
			if (isnan(bl_grad[g_idx]) || isinf(bl_grad[g_idx]))
			{
				printf("bl grad break: %f %s %s\n", bl_grad[g_idx],  get_name().c_str(), bl_next->get_name().c_str());
				printf("\n");
			}

		}
	}
}

void PoolingLayer::print_data()
{
	printf("INPUT DATA:\n");
	for (int h = 0; h < pool_input_h; h++)
	{
		for (int w = 0; w < pool_input_w; w++)
		{
			printf("%.4f ", bl_input[w + h* pool_input_w]);
		}
		printf("\n");
	}

	printf("POOLING DATA:\n");
	for (int h = 0; h < pool_input_h; h++)
	{
		for (int w = 0; w < pool_input_w; w++)
		{
			printf("%d ", pool_flag[w + h * pool_input_w]);
		}
		printf("\n");
	}
}

