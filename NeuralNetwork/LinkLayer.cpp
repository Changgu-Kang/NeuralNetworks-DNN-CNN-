#include "LinkLayer.h"


LinkLayer::LinkLayer(std::initializer_list<BaseLayer*> list)
{
	bl_l_type = LAYER_TYPE::LINK;

	int output_size = 0;
	for (auto elem : list)
	{
		output_size += elem->get_output_size();
		link_input_layers.push_back(elem);

	}


	bl_output_size = output_size;
	bl_grad_size = bl_output_size;

	bl_output = new double[bl_output_size];
	bl_grad = new double[bl_output_size];

}

LinkLayer::LinkLayer(std::initializer_list<BaseLayer*>& list)
{
	bl_l_type = LAYER_TYPE::LINK;

	int output_size = 0;
	for (auto elem : list)
	{
		output_size += elem->get_output_size();
		link_input_layers.push_back(elem);

	}


	bl_output_size = output_size;
	bl_grad_size = bl_output_size;

	bl_output = new double[bl_output_size];
	bl_grad = new double[bl_output_size];
}



LinkLayer::~LinkLayer()
{

}


//BaseLayer * LinkLayer::operator->()
//{
//	return bl_next;
//}

void LinkLayer::forward()
{
	int outIdx = 0;

	for (int i = 0; i < link_input_layers.size(); i++)
	{
		for (int j = 0; j < link_input_layers[i]->get_output_size(); j++)
		{
			this->bl_output[outIdx] = link_input_layers[i]->get_output_ptr()[j];
			outIdx++;
		}
	}	
}

double * LinkLayer::get_grad()
{
	return bl_next->get_grad();
}

double LinkLayer::get_w(BaseLayer * l, int x, int y)
{
	int x_s = 0;
	for (int i = 0; i < link_input_layers.size(); i++)
	{
		if (l == link_input_layers[i])
		{
			return bl_next->get_w()[(i + x_s) + y];
		}
		x_s += link_input_layers[i]->get_output_size();
	}
	printf("Link layer error\n");
	exit(-1);	
}





