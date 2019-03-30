#include <iostream>

//Use AVX and TBB for the best performance
#define CNN_USE_AVX
#define CNN_USE_TBB
#include "tiny_dnn/tiny_dnn.h"
#include "tiny_dnn/util/serialization_helper.h"

#include "kwinnder.hpp"
#include <thread>

static void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn,
	std::string type) {
	using fc = tiny_dnn::layers::fc;
	using conv = tiny_dnn::layers::conv;
	using max_pool = tiny_dnn::layers::max_pool;
	using activation = tiny_dnn::activation::tanh;
	using softmax = tiny_dnn::activation::softmax;
	using dropout = tiny_dnn::dropout_layer;
	using batch_normalization = tiny_dnn::batch_normalization_layer;
	using kwinner = tiny_dnn::kwinner_layer;

	nn << conv(32, 32, 5, 1, 6)
		<< max_pool(28, 28, 6, 2)
		<< activation();
	if(type == "dropout") nn << dropout(6*14*14, 0.3);
	if(type == "batchnorm") nn << batch_normalization(14*14, 6);
	nn << conv(14, 14, 5, 6, 16)
		<< max_pool(10, 10, 16, 2)
		<< activation();
	if(type == "dropout") nn << dropout(5*5*16, 0.3);
	if(type == "batchnorm") nn << batch_normalization(5*5, 16);
	nn << conv(5, 5, 5, 16, 120)
		<< activation();
	if(type == "kwinner") nn << kwinner({120}, 0.4);
	if(type == "batchnorm") nn << batch_normalization(120, 1);
	if(type == "dropout") nn << dropout(120, 0.3);
	nn << fc(120, 10) << softmax();
}

static void test_lenet_noise(tiny_dnn::network<tiny_dnn::sequential> &nn
	, const std::vector<tiny_dnn::vec_t>& test_image
	, const std::vector<tiny_dnn::label_t>& test_labels
	, float noise_factor)
{
	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<float> dist(0, 1);
	std::normal_distribution<float> noise_dist;


	//Noiseify the data
	auto noisy_images = test_image;
	for(auto& image : noisy_images) {
		for(auto& d : image) {
			if(dist(rng) < noise_factor)
				d = std::max(std::min(noise_dist(rng), 1.f), 0.f);
		}
	}

	auto result = nn.test(noisy_images, test_labels);
	std::cout << "Noise factor: " << noise_factor << ", acc = " << result.accuracy() << std::endl;
}

static void train_lenet(const std::string &data_dir_path,
		double learning_rate,
		const int n_train_epochs,
		const int n_minibatch,
		std::string type) {
	// specify loss-function and learning strategy
	tiny_dnn::network<tiny_dnn::sequential> nn;
	tiny_dnn::adam optimizer;

	construct_net(nn, type);

	std::cout << "load models..." << std::endl;

	// load MNIST dataset
	std::vector<tiny_dnn::label_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;

	tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
		&train_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
		&train_images, -1.0, 1.0, 2, 2);
	tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
		&test_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
		&test_images, -1.0, 1.0, 2, 2);

	std::cout << "start training" << std::endl;

	tiny_dnn::progress_display disp(train_images.size());
	tiny_dnn::timer t;

	optimizer.alpha *=
		std::min(tiny_dnn::float_t(4),  static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << "\nEpoch " << epoch << "/" << n_train_epochs << " finished. "
							<< t.elapsed() << "s elapsed." << std::endl;
		++epoch;
		tiny_dnn::result res = nn.test(test_images, test_labels);
		std::cout << "Accuracy: " << res.num_success << "/" << res.num_total << std::endl;

		disp.restart(train_images.size());

		if(epoch != n_train_epochs)
			t.restart();
	};

	auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

	// training
	nn.train<tiny_dnn::cross_entropy_multiclass>(optimizer, train_images, train_labels, n_minibatch,
		n_train_epochs, on_enumerate_minibatch,
		on_enumerate_epoch);

	std::cout << "end training." << std::endl;

	// test and show results
	nn.test(test_images, test_labels).print_detail(std::cout);
	//nn.save("LeNet-model"); //Cannot save due to being a custom layer.

	std::mt19937 rng;
	std::normal_distribution<float> dist(0.7, 1);
	tiny_dnn::vec_t v(32*32);
	for(auto& d : v)
		d = dist(rng);
	auto res = nn.predict(v);
	for(auto d : res)
		std::cout << d << " ";
	std::cout << std::endl;
	std::cout << std::endl;
	std::vector<float> noise = {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
	for(auto n : noise)
		test_lenet_noise(nn, test_images, test_labels, n);
}

static tiny_dnn::core::backend_t parse_backend_name(const std::string &name) {
	const std::array<const std::string, 5> names = {{
		"internal", "nnpack", "libdnn", "avx", "opencl",
	}};
	for (size_t i = 0; i < names.size(); ++i) {
		if (name.compare(names[i]) == 0) {
			return static_cast<tiny_dnn::core::backend_t>(i);
		}
	}
	return tiny_dnn::core::default_engine();
}

static void usage(const char *argv0) {
	std::cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder"
		<< " --learning_rate 1"
		<< " --epochs 10"
		<< " --minibatch_size 64"
		<< " --minibatch_size ['raw', 'kwinner', 'dropout', 'batchnorm']"
		<< " --backend_type avx" << std::endl;
}

int main(int argc, char **argv) {
	double learning_rate = 1;
	int epochs = 10;
	std::string data_path = ".";
	int minibatch_size = 64;
	std::string type = "kwinner";
	tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();
	std::vector<std::string> legal_type = {"raw", "kwinner", "dropout", "batchnorm"};
	if (argc == 2) {
		std::string argname(argv[1]);
		if (argname == "--help" || argname == "-h") {
			usage(argv[0]);
			return 0;
		}
	}
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--learning_rate") {
			learning_rate = atof(argv[count + 1]);
		} else if (argname == "--epochs") {
			epochs = atoi(argv[count + 1]);
		} else if (argname == "--minibatch_size") {
			minibatch_size = atoi(argv[count + 1]);
		} else if (argname == "--backend_type") {
			backend_type = parse_backend_name(argv[count + 1]);
		} else if (argname == "--data_path") {
			data_path = std::string(argv[count + 1]);
		} else if (argname == "--type") {
			type = std::string(argv[count + 1]);	
		} else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
								<< std::endl;
			usage(argv[0]);
			return -1;
		}
	}
	if (learning_rate <= 0) {
		std::cerr
			<< "Invalid learning rate. The learning rate must be greater than 0."
			<< std::endl;
		return -1;
	}
	if (epochs <= 0) {
		std::cerr << "Invalid number of epochs. The number of epochs must be "
			"greater than 0."
			<< std::endl;
		return -1;
	}
	if (minibatch_size <= 0 || minibatch_size > 60000) {
		std::cerr
			<< "Invalid minibatch size. The minibatch size must be greater than 0"
				 " and less than dataset size (60000)."
			<< std::endl;
		return -1;
	}
	if(std::find(legal_type.begin(), legal_type.end(), type) == legal_type.end()) {
		std::cerr 
			<< "Invalid regulaization type, must be ['raw', 'kwinner', 'dropout', 'batchnorm']"
			<< std::endl;
			return -1;
	}
	std::cout << "Running with the following parameters:" << std::endl
			<< "Data path: " << data_path << std::endl
			<< "Learning rate: " << learning_rate << std::endl
			<< "Minibatch size: " << minibatch_size << std::endl
			<< "Number of epochs: " << epochs << std::endl
			<< "Backend type: " << backend_type << std::endl
			<< "Reularization method: " << type << std::endl
			<< std::endl;
	try {
		train_lenet(data_path, learning_rate, epochs, minibatch_size, type);
	} catch (tiny_dnn::nn_error &err) {
		std::cerr << "Exception: " << err.what() << std::endl;
	}
	return 0;
}
