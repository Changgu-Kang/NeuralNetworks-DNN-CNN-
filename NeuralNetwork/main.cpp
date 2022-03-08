#include "NeuralNetwork.h"

int main()
{
	NeuralNetwork nn;

	//nn.load_traning_data_for_cnn("cnn_depth_circle.txt",0,',');
	//nn.load_traning_data_for_cnn("cnn_depth_ladder.txt", 1, ',');
	//nn.load_traning_data_for_cnn("cnn_depth_rectangle.txt", 2, ',');
	//nn.load_traning_data_for_cnn("cnn_depth_triangle.txt", 3, ',');

	/*nn.load_traning_data_for_cnn("cnn_circle (1).txt",0,',');
	nn.load_traning_data_for_cnn("cnn_ladder (1).txt", 1, ',');
	nn.load_traning_data_for_cnn("cnn_triangle (1).txt", 2, ',');
	nn.load_traning_data_for_cnn("cnn_rectangle (1).txt", 3, ',');

	nn.load_traning_data_for_cnn("cnn_circle (2).txt", 0, ',');
	nn.load_traning_data_for_cnn("cnn_ladder (2).txt", 1, ',');
	nn.load_traning_data_for_cnn("cnn_triangle (2).txt", 2, ',');
	nn.load_traning_data_for_cnn("cnn_rectangle (2).txt", 3, ',');

	nn.load_traning_data_for_cnn("cnn_circle (3).txt", 0, ',');
	nn.load_traning_data_for_cnn("cnn_ladder (3).txt", 1, ',');
	nn.load_traning_data_for_cnn("cnn_triangle (3).txt", 2, ',');
	nn.load_traning_data_for_cnn("cnn_rectangle (3).txt", 3, ',');

	nn.load_traning_data_for_cnn("cnn_circle (4).txt", 0, ',');
	nn.load_traning_data_for_cnn("cnn_ladder (4).txt", 1, ',');
	nn.load_traning_data_for_cnn("cnn_triangle (4).txt", 2, ',');
	nn.load_traning_data_for_cnn("cnn_rectangle (4).txt", 3, ',');

	nn.load_traning_data_for_cnn("cnn_circle (5).txt", 0, ',');
	nn.load_traning_data_for_cnn("cnn_ladder (5).txt", 1, ',');
	nn.load_traning_data_for_cnn("cnn_triangle (5).txt", 2, ',');
	nn.load_traning_data_for_cnn("cnn_rectangle (5).txt", 3, ',');*/




	

	//nn.load_traning_data_for_cnn("cnn_training_data1.txt", 0,' ');
	//nn.load_traning_data_for_cnn("cnn_training_data2.txt", 1, ' ');
	//nn.load_traning_data_for_cnn("cnn_training_data3.txt", 2, ' ');
	//nn.load_traning_data_for_cnn("cnn_training_data4.txt", 3, ' ');

	//nn.load_training_data_in_dir("training_data/circle", 0);
	//nn.load_training_data_in_dir("training_data/ladder", 1);
	//nn.load_training_data_in_dir("training_data/rectangle", 2);
	//nn.load_training_data_in_dir("training_data/triangle", 3);

	//nn.set_data_all_normalize();

	
	//nn.design_fully_connected_neural_network(0);
	//nn.design_convolutional_neural_network_TEST();
	//nn.design_fully_connected_neural_network(3);
	//nn.design_convolutional_neural_network_D();

	nn.load_training_data_for_fcnn("color_training/1_b_training_data_X.csv","color_training/1_b_training_data_Y.csv",',');
	nn.design_fully_connected_neural_network(1);	

	//learning_rate = 0.02;
	//learning_rate = 0.00003;
	learning_rate = 0.000005;
	momentum_rate = 0.5;
	//[0.5, 0.9, 0.95, 0.99]
	//nn.sort_data();
	nn.training(10000);
	
	
	


	return 0;
}