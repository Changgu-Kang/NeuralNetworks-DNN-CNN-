#pragma once
#include "LayerHeader.h"

size_t FindFiles(std::string strPath, std::vector<std::string> &q, bool bIncludeSub);

class NeuralNetwork {

public:
	NeuralNetwork()
	{

	}
	~NeuralNetwork()
	{
		std::vector<BaseLayer*>::iterator it = nn_layers.begin();
		while (it!= nn_layers.end())
		{
			BaseLayer* temp = (*it);
			it = nn_layers.erase(it);
			delete temp;
		}
	}

	void training(int max_loop = 100)
	{
		for (int loop = 0; loop < max_loop; loop++)
		{
			double Error = 0.0;
			//printf("Loop # %d\n", loop);
			for (int d_idx = 0; d_idx < nn_training_x.size(); d_idx++)
	//		for (int rand_idx = 0; rand_idx < nn_training_x.size()/2; rand_idx++)
			{
				//int d_idx = rand() % nn_training_x.size();

				//printf("d_idx %d\n",d_idx);

				copy_training_data(d_idx);

				for (int i = 0; i < nn_layers.size(); i++)
					nn_layers[i]->forward();

				//printf("%d : ",d_idx);
				for (int i = 0; i < nn_layers[nn_layers.size() - 1]->get_output_size(); i++)
				{
					//printf("%f ", nn_layers[nn_layers.size() - 1]->get_output_ptr()[i]);
				}
				//printf("\n");


				for (int i = nn_layers.size() - 1; i >= 0; i--)
					nn_layers[i]->update_gradient();

				for (int i = 0; i < nn_layers.size(); i++)
					nn_layers[i]->update_weight();

				double error_cost = 0.0;

				for (int i = 0; i < nn_training_y_dim; i++)
				{
					error_cost += sqrt(abs(nn_y_data[i] - nn_layers[nn_layers.size() - 1]->get_output_ptr()[i]));
				}

				Error += error_cost / nn_training_y_dim;
				
				
				//if(loop%50==49)
				if (max_loop - 1 == loop)
				{
					printf("%d \t [", d_idx);
					for (int i = 0; i < nn_training_y_dim; i++)
					{						
						printf("%d ", (int)(nn_y_data[i]));
					}
					printf("] -> ");

					printf("[");
					for (int i = 0; i < nn_training_y_dim; i++)
					{
						if (isnan(Error))
						{
							//printf("%f %f\n", nn_y_data[i], nn_layers[nn_layers.size() - 1]->get_output_ptr()[i]);
						}
						
						printf("%.2f ",  nn_layers[nn_layers.size() - 1]->get_output_ptr()[i]);
					}
					printf("] -> ");

					printf("[");
					for (int i = 0; i < nn_training_y_dim; i++)
					{
						printf("%d ", (int)round(nn_layers[nn_layers.size() - 1]->get_output_ptr()[i]));
					}
					printf("]\n");


				}
			}

			/*if (Error / nn_training_x.size() / 10 < 0.001)
			{
				break;
			}*/
				

			//if (loop % 10 == 9)
			{
				printf("Loop # %d - %.6f\n", loop, Error / nn_training_x.size());
			}
		}
	}

	void set_data_all_normalize()
	{
		double max = 0.0;
		double min = 1.0;


		for (int i = 0; i < nn_training_x.size(); i++)
		{

			for (int j = 0; j < nn_training_x[i].size(); j++)
			{
				if (max < nn_training_x[i][j])
					max = nn_training_x[i][j];
				if (min > nn_training_x[i][j])
					min = nn_training_x[i][j];

			}
		}


		for (int i = 0; i < nn_training_x.size(); i++)
		{

			for (int j = 0; j < nn_training_x[i].size(); j++)
			{
				nn_training_x[i][j] = (nn_training_x[i][j] - min) / (max - min);
			}
		}


	}

	void set_data_normalize()
	{
		for (int i = 0; i < nn_training_x[0].size(); i++)
		{
			double max = 0.0;
			double min = 1.0;

			for (int j = 0; j < nn_training_x.size(); j++)
			{
				if (max < nn_training_x[j][i])
					max = nn_training_x[j][i];
				if (min > nn_training_x[j][i])
					min = nn_training_x[j][i];
				
			}

			for (int j = 0; j < nn_training_x.size(); j++)
			{
				nn_training_x[j][i] = (nn_training_x[j][i] - min)/(max-min);
			}
		}
	}

	void reset_weight()
	{
		for (int i = 0; i < nn_layers.size(); i++)
			nn_layers[i]->reset_weight();
	}

	void copy_training_data(int d_idx)
	{
		for (int i = 0; i < nn_training_x_dim; i++)
			nn_x_data[i] = nn_training_x[d_idx][i];

		for (int i = 0; i < nn_training_y_dim; i++)
			nn_y_data[i] = nn_training_y[d_idx][i];
	}

	void design_convolutional_neural_network_TEST()
	{

		nn_training_x_dim = nn_training_x[0].size();
		nn_training_y_dim = nn_training_y[0].size();

		printf("%d %d\n", nn_cnn_2d_x_dim, nn_cnn_2d_y_dim);

		nn_x_data = new double[nn_training_x_dim];
		nn_y_data = new double[nn_training_y_dim];

		ConvLayer*			layer_0_conv = new ConvLayer(nn_cnn_2d_x_dim, nn_cnn_2d_y_dim, 1, 3, 3, ACT_TYPE::TANH,"L0");
		PoolingLayer*		layer_1_pool = new PoolingLayer(layer_0_conv->get_conv_output_w(), layer_0_conv->get_conv_output_h(), 3, 3, PoolingLayer::POOLING_TYPE::MAX, "L0");
		FCLayer*			layer_2_out = new FCLayer(layer_1_pool->get_output_size(), nn_training_y_dim, LAYER_TYPE::OUTPUT, ACT_TYPE::LRELU, "L0");

		layer_0_conv->set_input_ptr(nn_x_data);
		layer_2_out->set_true_y_ptr(nn_y_data);

		layer_0_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_1_pool));
		layer_1_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_2_out));


		//pooling_layer->connect_next_layer(dynamic_cast<BaseLayer*>(hidden_layer));
		//pooling_layer->connect_next_layer(dynamic_cast<BaseLayer*>(end_layer));

		nn_layers.push_back(layer_0_conv);
		nn_layers.push_back(layer_1_pool);
		nn_layers.push_back(layer_2_out);
	}



	void design_convolutional_neural_network_B()
	{

		nn_training_x_dim = nn_training_x[0].size();
		nn_training_y_dim = nn_training_y[0].size();

		printf("%d %d\n", nn_cnn_2d_x_dim, nn_cnn_2d_y_dim);

		nn_x_data = new double[nn_training_x_dim];
		nn_y_data = new double[nn_training_y_dim];

		ConvLayer*		layer_0_conv = new ConvLayer(nn_cnn_2d_x_dim, nn_cnn_2d_y_dim, 1, 9, 9, ACT_TYPE::TANH,"L0");		
		ConvLayer*		layer_1_conv = new ConvLayer(layer_0_conv->get_conv_output_w(), layer_0_conv->get_conv_output_h(), 1, 3, 3, ACT_TYPE::TANH, "L1");
		FCLayer*		layer_2_fc = new FCLayer(layer_1_conv->get_output_size(), layer_1_conv->get_output_size()/2, LAYER_TYPE::FC, ACT_TYPE::RELU, "L2");
		FCLayer*		layer_3_out = new FCLayer(layer_2_fc->get_output_size(), nn_training_y_dim, LAYER_TYPE::OUTPUT, ACT_TYPE::RELU, "L3");

		layer_0_conv->set_input_ptr(nn_x_data);
		layer_3_out->set_true_y_ptr(nn_y_data);

		layer_0_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_1_conv));
		layer_1_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_2_fc));
		layer_2_fc->connect_next_layer(dynamic_cast<BaseLayer*>(layer_3_out));
		

		//pooling_layer->connect_next_layer(dynamic_cast<BaseLayer*>(hidden_layer));
		//pooling_layer->connect_next_layer(dynamic_cast<BaseLayer*>(end_layer));



		nn_layers.push_back(layer_0_conv);
		nn_layers.push_back(layer_1_conv);
		nn_layers.push_back(layer_2_fc);
		nn_layers.push_back(layer_3_out);

	}

	/*
	void design_convolutional_neural_network_E()
	{

		nn_training_x_dim = nn_training_x[0].size();
		nn_training_y_dim = nn_training_y[0].size();

		//printf("%d %d\n", nn_cnn_2d_x_dim, nn_cnn_2d_y_dim);

		nn_x_data = new double[nn_training_x_dim];
		nn_y_data = new double[nn_training_y_dim];

		std::vector<BaseLayer*>	fclayers;

		for (int i = 0; i < 10; i ++ )
		{		

			ConvLayer*		layer_0_conv = new ConvLayer(nn_cnn_2d_x_dim, nn_cnn_2d_y_dim, 1, 3, 3, ACT_TYPE::TANH, "L0");			
			PoolingLayer*	layer_1_pool = new PoolingLayer(layer_0_conv->get_conv_output_w(), layer_0_conv->get_conv_output_h(), 2, 2, PoolingLayer::POOLING_TYPE::MAX, "L1");			
			ConvLayer*		layer_2_conv = new ConvLayer(layer_1_pool->get_pooling_output_w(), layer_1_pool->get_pooling_output_h(), 1, 4, 4, ACT_TYPE::TANH, "L2");			
			PoolingLayer*	layer_3_pool = new PoolingLayer(layer_2_conv->get_conv_output_w(), layer_2_conv->get_conv_output_h(), 4, 4, PoolingLayer::POOLING_TYPE::MAX, "L3");			
			ConvLayer*		layer_4_conv = new ConvLayer(layer_3_pool->get_pooling_output_w(), layer_3_pool->get_pooling_output_h(), 1, 2, 2, ACT_TYPE::TANH, "L4");			
			PoolingLayer*	layer_5_pool = new PoolingLayer(layer_4_conv->get_conv_output_w(), layer_4_conv->get_conv_output_h(), 3, 3, PoolingLayer::POOLING_TYPE::MAX, "L5");
			FCLayer*		layer_6_fc = new FCLayer(layer_5_pool->get_output_size(), layer_5_pool->get_output_size(), LAYER_TYPE::FC, ACT_TYPE::TANH, "L06");
			fclayers.push_back(dynamic_cast<BaseLayer*>(layer_6_fc));

			layer_0_conv->set_input_ptr(nn_x_data);
		}


		LinkLayer*		layer_link = new LinkLayer(fclayers);

		FCLayer*		layer_out = new FCLayer(layer_link->get_output_size(), nn_training_y_dim, LAYER_TYPE::OUTPUT, ACT_TYPE::LRELU, "L7");
		
		layer_out->set_true_y_ptr(nn_y_data);


		layer_0_0_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_0_1_pool));
		layer_0_1_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_0_2_conv));
		layer_0_2_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_0_3_pool));
		layer_0_3_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_0_4_conv));
		layer_0_4_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_0_5_pool));
		layer_0_5_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_0_6_fc));
		layer_0_6_fc->connect_next_layer(dynamic_cast<BaseLayer*>(layer_link));

		layer_1_0_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_1_1_pool));
		layer_1_1_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_1_2_conv));
		layer_1_2_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_1_3_pool));
		layer_1_3_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_1_4_conv));
		layer_1_4_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_1_5_pool));
		layer_1_5_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_1_6_fc));
		layer_1_6_fc->connect_next_layer(dynamic_cast<BaseLayer*>(layer_link));

		layer_2_0_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_2_1_pool));
		layer_2_1_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_2_2_conv));
		layer_2_2_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_2_3_pool));
		layer_2_3_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_2_4_conv));
		layer_2_4_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_2_5_pool));
		layer_2_5_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_2_6_fc));
		layer_2_6_fc->connect_next_layer(dynamic_cast<BaseLayer*>(layer_link));

		layer_link->connect_next_layer(dynamic_cast<BaseLayer*>(layer_out));


		nn_layers.push_back(layer_0_0_conv);
		nn_layers.push_back(layer_0_1_pool);
		nn_layers.push_back(layer_0_2_conv);
		nn_layers.push_back(layer_0_3_pool);
		nn_layers.push_back(layer_0_4_conv);
		nn_layers.push_back(layer_0_5_pool);
		nn_layers.push_back(layer_0_6_fc);

		nn_layers.push_back(layer_1_0_conv);
		nn_layers.push_back(layer_1_1_pool);
		nn_layers.push_back(layer_1_2_conv);
		nn_layers.push_back(layer_1_3_pool);
		nn_layers.push_back(layer_1_4_conv);
		nn_layers.push_back(layer_1_5_pool);
		nn_layers.push_back(layer_1_6_fc);

		nn_layers.push_back(layer_2_0_conv);
		nn_layers.push_back(layer_2_1_pool);
		nn_layers.push_back(layer_2_2_conv);
		nn_layers.push_back(layer_2_3_pool);
		nn_layers.push_back(layer_2_4_conv);
		nn_layers.push_back(layer_2_5_pool);
		nn_layers.push_back(layer_2_6_fc);


		nn_layers.push_back(layer_link);
		nn_layers.push_back(layer_out);
	}

	*/

	void design_convolutional_neural_network_D()
	{

		nn_training_x_dim = nn_training_x[0].size();
		nn_training_y_dim = nn_training_y[0].size();

		//printf("%d %d\n", nn_cnn_2d_x_dim, nn_cnn_2d_y_dim);

		nn_x_data = new double[nn_training_x_dim];
		nn_y_data = new double[nn_training_y_dim];

		ConvLayer*		layer_0_0_conv = new ConvLayer(nn_cnn_2d_x_dim, nn_cnn_2d_y_dim, 1, 3, 3, ACT_TYPE::TANH, "L00");
		PoolingLayer*	layer_0_1_pool = new PoolingLayer(layer_0_0_conv->get_conv_output_w(), layer_0_0_conv->get_conv_output_h(), 2, 2, PoolingLayer::POOLING_TYPE::MAX, "L01");
		ConvLayer*		layer_0_2_conv = new ConvLayer(layer_0_1_pool->get_pooling_output_w(), layer_0_1_pool->get_pooling_output_h(), 1, 4, 4, ACT_TYPE::TANH, "L02");
		PoolingLayer*	layer_0_3_pool = new PoolingLayer(layer_0_2_conv->get_conv_output_w(), layer_0_2_conv->get_conv_output_h(), 4, 4, PoolingLayer::POOLING_TYPE::MAX, "L03");
		ConvLayer*		layer_0_4_conv = new ConvLayer(layer_0_3_pool->get_pooling_output_w(), layer_0_3_pool->get_pooling_output_h(), 1, 2, 2, ACT_TYPE::TANH, "L04");
		PoolingLayer*	layer_0_5_pool = new PoolingLayer(layer_0_4_conv->get_conv_output_w(), layer_0_4_conv->get_conv_output_h(), 3, 3, PoolingLayer::POOLING_TYPE::MAX, "L05");
		FCLayer*		layer_0_6_fc = new FCLayer(layer_0_5_pool->get_output_size(), layer_0_5_pool->get_output_size(), LAYER_TYPE::FC, ACT_TYPE::LRELU, "L06");

		ConvLayer*		layer_1_0_conv = new ConvLayer(nn_cnn_2d_x_dim, nn_cnn_2d_y_dim, 1, 3, 3, ACT_TYPE::TANH, "L10");
		PoolingLayer*	layer_1_1_pool = new PoolingLayer(layer_1_0_conv->get_conv_output_w(), layer_1_0_conv->get_conv_output_h(), 2, 2, PoolingLayer::POOLING_TYPE::MAX, "L11");
		ConvLayer*		layer_1_2_conv = new ConvLayer(layer_1_1_pool->get_pooling_output_w(), layer_1_1_pool->get_pooling_output_h(), 1, 4, 4, ACT_TYPE::TANH, "L12");
		PoolingLayer*	layer_1_3_pool = new PoolingLayer(layer_1_2_conv->get_conv_output_w(), layer_1_2_conv->get_conv_output_h(), 4, 4, PoolingLayer::POOLING_TYPE::MAX, "L13");
		ConvLayer*		layer_1_4_conv = new ConvLayer(layer_1_3_pool->get_pooling_output_w(), layer_1_3_pool->get_pooling_output_h(), 1, 2, 2, ACT_TYPE::TANH, "L14");
		PoolingLayer*	layer_1_5_pool = new PoolingLayer(layer_1_4_conv->get_conv_output_w(), layer_1_4_conv->get_conv_output_h(), 3, 3, PoolingLayer::POOLING_TYPE::MAX, "L15");
		FCLayer*		layer_1_6_fc = new FCLayer(layer_1_5_pool->get_output_size(), layer_1_5_pool->get_output_size(), LAYER_TYPE::FC, ACT_TYPE::LRELU, "L16");

		

		LinkLayer*		layer_link = new LinkLayer({ layer_0_6_fc ,layer_1_6_fc});

		FCLayer*		layer_out = new FCLayer(layer_link->get_output_size(), nn_training_y_dim, LAYER_TYPE::OUTPUT, ACT_TYPE::LRELU, "L7");


		layer_0_0_conv->set_input_ptr(nn_x_data);
		layer_1_0_conv->set_input_ptr(nn_x_data);		
		layer_out->set_true_y_ptr(nn_y_data);
		

		layer_0_0_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_0_1_pool));
		layer_0_1_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_0_2_conv));
		layer_0_2_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_0_3_pool));
		layer_0_3_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_0_4_conv));
		layer_0_4_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_0_5_pool));
		layer_0_5_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_0_6_fc));
		layer_0_6_fc->connect_next_layer(dynamic_cast<BaseLayer*>(layer_link));

		layer_1_0_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_1_1_pool));
		layer_1_1_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_1_2_conv));
		layer_1_2_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_1_3_pool));
		layer_1_3_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_1_4_conv));
		layer_1_4_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_1_5_pool));
		layer_1_5_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_1_6_fc));
		layer_1_6_fc->connect_next_layer(dynamic_cast<BaseLayer*>(layer_link));


		layer_link->connect_next_layer(dynamic_cast<BaseLayer*>(layer_out));

		
		nn_layers.push_back(layer_0_0_conv);
		nn_layers.push_back(layer_0_1_pool);
		nn_layers.push_back(layer_0_2_conv);
		nn_layers.push_back(layer_0_3_pool);
		nn_layers.push_back(layer_0_4_conv);
		nn_layers.push_back(layer_0_5_pool);
		nn_layers.push_back(layer_0_6_fc);

		nn_layers.push_back(layer_1_0_conv);
		nn_layers.push_back(layer_1_1_pool);
		nn_layers.push_back(layer_1_2_conv);
		nn_layers.push_back(layer_1_3_pool);
		nn_layers.push_back(layer_1_4_conv);
		nn_layers.push_back(layer_1_5_pool);
		nn_layers.push_back(layer_1_6_fc);


		nn_layers.push_back(layer_link);
		nn_layers.push_back(layer_out);
	}


	void design_convolutional_neural_network_C()
	{	

		
		nn_training_x_dim = nn_training_x[0].size();
		nn_training_y_dim = nn_training_y[0].size();

		printf("%d %d\n", nn_cnn_2d_x_dim, nn_cnn_2d_y_dim);

		nn_x_data = new double[nn_training_x_dim];
		nn_y_data = new double[nn_training_y_dim];

		ConvLayer*		layer_0_conv = new ConvLayer(nn_cnn_2d_x_dim, nn_cnn_2d_y_dim, 1, 3, 3, ACT_TYPE::TANH,"L0");
		printf("L0(C): %d %d\n", layer_0_conv->get_conv_output_w(), layer_0_conv->get_conv_output_h());
		PoolingLayer*	layer_1_pool = new PoolingLayer(layer_0_conv->get_conv_output_w(), layer_0_conv->get_conv_output_h(), 2, 2, PoolingLayer::POOLING_TYPE::MAX, "L1");
		printf("L1(P): %d %d\n", layer_1_pool->get_pooling_output_w(), layer_1_pool->get_pooling_output_h());
		ConvLayer*		layer_2_conv = new ConvLayer(layer_1_pool->get_pooling_output_w(), layer_1_pool->get_pooling_output_h(), 1, 4, 4, ACT_TYPE::TANH, "L2");
		printf("L2(C): %d %d\n", layer_2_conv->get_conv_output_w(), layer_2_conv->get_conv_output_h());
		PoolingLayer*	layer_3_pool = new PoolingLayer(layer_2_conv->get_conv_output_w(), layer_2_conv->get_conv_output_h(), 4, 4, PoolingLayer::POOLING_TYPE::MAX, "L3");
		printf("L3(P): %d %d\n", layer_3_pool->get_pooling_output_w(), layer_3_pool->get_pooling_output_h());
		ConvLayer*		layer_4_conv = new ConvLayer(layer_3_pool->get_pooling_output_w(), layer_3_pool->get_pooling_output_h(), 1, 2, 2, ACT_TYPE::TANH, "L4");
		printf("L4(C): %d %d\n", layer_4_conv->get_conv_output_w(), layer_4_conv->get_conv_output_h());
		PoolingLayer*	layer_5_pool = new PoolingLayer(layer_4_conv->get_conv_output_w(), layer_4_conv->get_conv_output_h(), 3, 3, PoolingLayer::POOLING_TYPE::MAX, "L5");
		printf("L5(P): %d %d\n", layer_5_pool->get_pooling_output_w(), layer_5_pool->get_pooling_output_h());
		//FCLayer*		layer_6_fc = new FCLayer(layer_5_pool->get_output_size(), layer_5_pool->get_output_size(), LAYER_TYPE::FC, ACT_TYPE::LRELU, "L6");
		FCLayer*		layer_7_out = new FCLayer(layer_5_pool->get_output_size(), nn_training_y_dim, LAYER_TYPE::OUTPUT, ACT_TYPE::LRELU,"L7");

		layer_0_conv->set_input_ptr(nn_x_data);
		layer_7_out->set_true_y_ptr(nn_y_data);

		layer_0_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_1_pool));
		layer_1_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_2_conv));
		layer_2_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_3_pool));
		layer_3_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_4_conv));		
		layer_4_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_5_pool));
		layer_5_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_7_out));
	//	layer_6_fc->connect_next_layer(dynamic_cast<BaseLayer*>(layer_7_out));

		//pooling_layer->connect_next_layer(dynamic_cast<BaseLayer*>(hidden_layer));
		//pooling_layer->connect_next_layer(dynamic_cast<BaseLayer*>(end_layer));



		nn_layers.push_back(layer_0_conv);
		nn_layers.push_back(layer_1_pool);
		nn_layers.push_back(layer_2_conv);
		nn_layers.push_back(layer_3_pool);
		nn_layers.push_back(layer_4_conv);
		nn_layers.push_back(layer_5_pool);
//		nn_layers.push_back(layer_6_fc);
		nn_layers.push_back(layer_7_out);

	}




	void design_convolutional_neural_network_A()
	{

		nn_training_x_dim = nn_training_x[0].size();
		nn_training_y_dim = nn_training_y[0].size();

		printf("%d %d\n", nn_cnn_2d_x_dim, nn_cnn_2d_y_dim);

		nn_x_data = new double[nn_training_x_dim];
		nn_y_data = new double[nn_training_y_dim];

		ConvLayer*		layer_0_conv = new ConvLayer(nn_cnn_2d_x_dim, nn_cnn_2d_y_dim, 1, 9, 9,ACT_TYPE::TANH);
		PoolingLayer*	layer_1_pool = new PoolingLayer(layer_0_conv->get_conv_output_w(), layer_0_conv->get_conv_output_h(), 7, 7, PoolingLayer::POOLING_TYPE::MAX);
		ConvLayer*		layer_2_conv = new ConvLayer(layer_1_pool->get_pooling_output_w(), layer_1_pool->get_pooling_output_h(), 1, 3, 3, ACT_TYPE::TANH);
		PoolingLayer*	layer_3_pool = new PoolingLayer(layer_2_conv->get_conv_output_w(), layer_2_conv->get_conv_output_h(), 2, 2, PoolingLayer::POOLING_TYPE::MAX);
		FCLayer*		layer_4_fc = new FCLayer(layer_3_pool->get_output_size(), layer_3_pool->get_output_size(), LAYER_TYPE::FC,ACT_TYPE::LRELU);
		FCLayer*		layer_5_out = new FCLayer(layer_3_pool->get_output_size(), nn_training_y_dim, LAYER_TYPE::OUTPUT, ACT_TYPE::LRELU);

		layer_0_conv->set_input_ptr(nn_x_data);
		layer_5_out->set_true_y_ptr(nn_y_data);

		layer_0_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_1_pool));
		layer_1_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_2_conv));
		layer_2_conv->connect_next_layer(dynamic_cast<BaseLayer*>(layer_3_pool));
		layer_3_pool->connect_next_layer(dynamic_cast<BaseLayer*>(layer_4_fc));
		layer_4_fc->connect_next_layer(dynamic_cast<BaseLayer*>(layer_5_out));

		//pooling_layer->connect_next_layer(dynamic_cast<BaseLayer*>(hidden_layer));
		//pooling_layer->connect_next_layer(dynamic_cast<BaseLayer*>(end_layer));



		nn_layers.push_back(layer_0_conv);
		nn_layers.push_back(layer_1_pool);
		nn_layers.push_back(layer_2_conv);
		nn_layers.push_back(layer_3_pool);
		nn_layers.push_back(layer_4_fc);
		nn_layers.push_back(layer_5_out);
		
	}


	void design_fully_connected_neural_network(int hidden_size = 0)
	{
		nn_training_x_dim = nn_training_x[0].size();
		nn_training_y_dim = nn_training_y[0].size();

		nn_x_data = new double[nn_training_x_dim];
		nn_y_data = new double[nn_training_y_dim];


		FCLayer* start_layer = new FCLayer(nn_training_x_dim, nn_training_x_dim, LAYER_TYPE::FC,ACT_TYPE::RELU,"LS");
		FCLayer* end_layer = new FCLayer(nn_training_x_dim, nn_training_y_dim, LAYER_TYPE::OUTPUT,ACT_TYPE::RELU, "LE");

		start_layer->set_input_ptr(nn_x_data);
		end_layer->set_true_y_ptr(nn_y_data);

		nn_layers.push_back(start_layer);

		for (int i = 0; i < hidden_size; i++)
		{
			FCLayer* hidden_layer = new FCLayer(nn_training_x_dim, nn_training_x_dim, LAYER_TYPE::FC,ACT_TYPE::RELU);
			nn_layers[nn_layers.size() - 1]->connect_next_layer(dynamic_cast<BaseLayer*>(hidden_layer));
			nn_layers.push_back(hidden_layer);
		}

		nn_layers[nn_layers.size() - 1]->connect_next_layer(dynamic_cast<BaseLayer*>(end_layer));
		nn_layers.push_back(end_layer);
	}

	void load_traning_data_for_cnn(std::string file, int what = 0 ,char delimiter = ',')
	{
		{
			nn_cnn_2d_y_dim = 0;

			std::fstream in;
			in.open(file, std::ios_base::in);
			if (in.is_open())
			{
				std::vector<double> input;

				while (!in.eof())
				{
					char line[1024] = { 0 };
					in.getline(line, 1024);
					std::string strline(line);

					if (strline.length() == 0) continue;

					nn_cnn_2d_x_dim = 1;
					nn_cnn_2d_y_dim++;


					while (strline.find(delimiter) != std::string::npos)
					{
						std::string data = strline.substr(0, strline.find(delimiter));
						input.push_back(atof(data.c_str()));
						strline = strline.substr(strline.find(delimiter) + 1);

						nn_cnn_2d_x_dim++;
					}
					input.push_back(atof(strline.c_str()));
				}
				in.close();
				nn_training_x.push_back(input);
			}

			{
				std::vector<double> temp;
				//temp.push_back(what);
				for (int i = 0; i < 4; i++)
				{
					if (what == i)
						temp.push_back(1);
					else
						temp.push_back(0);
				}
				
				nn_training_y.push_back(temp);
			}

		}
	}



	void print_data(int i)
	{
		printf("data %d\n", i);
		for (int y = 0; y < nn_cnn_2d_y_dim; y++)
		{
			for (int x = 0; x < nn_cnn_2d_x_dim; x++)
			{
				int idx = y * nn_cnn_2d_x_dim + x;
				printf("%f ", nn_training_x[i][idx]);
			}
			printf("\n");
		}
	}

	


	void load_training_data_in_dir(std::string dir, int what , char delimiter = ',')
	{
		
		std::vector<std::string> files;
		FindFiles(dir, files, false);

		printf("Loading training_data(%d):", files.size());

		for (int i = 0; i<files.size(); i++)
		{
			printf(".");
			load_traning_data_for_cnn(files[i], what, delimiter);
		}	
		printf("\n");
	}



	void load_training_data_for_fcnn(std::string x_path, std::string y_path, char delimiter = ',')
	{
		{
			std::fstream in;
			in.open(x_path, std::ios_base::in);
			if (in.is_open())
			{
				while (!in.eof())
				{
					char line[1024] = { 0 };
					in.getline(line, 1024);
					std::string strline(line);

					if (strline.length() == 0) continue;

					std::vector<double> input;

					while (strline.find(delimiter) != std::string::npos)
					{
						std::string data = strline.substr(0, strline.find(delimiter));
						input.push_back(atof(data.c_str()));
						strline = strline.substr(strline.find(delimiter) + 1);
					}
					input.push_back(atof(strline.c_str()));
					nn_training_x.push_back(input);

				}
				in.close();
			}
			nn_training_x_dim = nn_training_x[0].size();
		}

		{
			std::fstream in;
			in.open(y_path, std::ios_base::in);
			if (in.is_open())
			{

				while (!in.eof())
				{
					char line[1024] = { 0 };
					in.getline(line, 1024);
					std::string strline(line);

					if (strline.length() == 0) continue;

					std::vector<double> output;

					while (strline.find(delimiter) != std::string::npos)
					{
						std::string data = strline.substr(0, strline.find(delimiter));
						output.push_back(atof(data.c_str()));
						strline = strline.substr(strline.find(delimiter) + 1);
					}
					output.push_back(atof(strline.c_str()));
					nn_training_y.push_back(output);
				}
			}
			in.close();
		}
		nn_training_y_dim = nn_training_y[0].size();

		nn_x_data = new double[nn_training_x_dim];
		nn_y_data = new double[nn_training_y_dim];
	}

	void save_weight(std::string path)
	{
		std::fstream out;
		out.open(path, std::ios_base::out);

		for (int i = 0; i < nn_layers.size(); i++)
		{

			for (int j = 0; j < nn_layers[i]->get_w_real_size(); j++)
			{
				if (j == 0) out << nn_layers[i]->get_w()[j];
				else out << "," << nn_layers[i]->get_w()[j];
			}
			if (i != nn_layers.size() - 1) out << std::endl;
		}
		out.close();

	}

	void load_weight(std::string path)
	{
		printf("Load weight: %s\n", path.c_str());

		std::fstream in;
		in.open(path, std::ios_base::in);
		if (in.is_open())
		{

			int layer_idx = 0;
			while (!in.eof())
			{
				char line[1024] = { 0 };
				in.getline(line, 1024);
				std::string strline(line);

				if (strline.length() == 0) continue;

				int w_idx = 0;

				while (strline.find(",") != std::string::npos)
				{
					std::string data = strline.substr(0, strline.find(","));
					nn_layers[layer_idx]->get_w()[w_idx] = atof(data.c_str());
					w_idx++;
					strline = strline.substr(strline.find(",") + 1);
				}
				nn_layers[layer_idx]->get_w()[w_idx] = atof(strline.c_str());
				layer_idx++;
			}
			in.close();
		}
	}

	void sort_data()
	{
		for (int i = 0; i < nn_training_x.size(); i++)
		{
			int idx0 = rand() % nn_training_x.size();
			int idx1 = rand() % nn_training_x.size();

			for (int j = 0; j < nn_training_x[idx0].size(); j++)
			{
				nn_x_data[j] = nn_training_x[idx0][j];
				nn_training_x[idx0][j] = nn_training_x[idx1][j];
				nn_training_x[idx1][j] = nn_x_data[j];
			}

			for (int j = 0; j < nn_training_y[idx0].size(); j++)
			{
				nn_y_data[j] = nn_training_y[idx0][j];
				nn_training_y[idx0][j] = nn_training_y[idx1][j];
				nn_training_y[idx1][j] = nn_y_data[j];
			}
		}
	}


private:

	double*							nn_x_data;
	double*							nn_y_data;



	std::vector<BaseLayer*>	nn_layers;
	std::vector<std::vector<double>> nn_training_x;
	std::vector<std::vector<double>> nn_training_y;

	int								nn_cnn_2d_x_dim;
	int								nn_cnn_2d_y_dim;

	int								nn_training_x_dim;
	int								nn_training_y_dim;
};
