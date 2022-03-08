#pragma once
#include "LayerHeader.h"

class LinkLayer : public BaseLayer
{
public:
	LinkLayer() {}
	LinkLayer(std::initializer_list<BaseLayer*> list);
	LinkLayer(std::initializer_list<BaseLayer*> &list);
	~LinkLayer();
	

	//BaseLayer*	operator->();	

	virtual void forward();
	virtual void update_gradient() {};
	virtual void update_weight() {};

	double*		get_grad();
	double		get_w(BaseLayer* l, int x, int y);

		
private:	
	std::vector<BaseLayer*>	link_input_layers;		
};

