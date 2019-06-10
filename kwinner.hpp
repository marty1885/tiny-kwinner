#pragma once

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn
{

template <typename T, typename Compare>
inline std::vector<std::size_t> sort_permutation(
    const T& vec,
    Compare compare)
{
	std::vector<std::size_t> p(vec.size());
	std::iota(p.begin(), p.end(), 0);
	std::sort(p.begin(), p.end(),
		[&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
	return p;
}

class kwinner_layer : public tiny_dnn::layer
{
public:
	kwinner_layer() : layer({vector_type::data}, {vector_type::data}) {}
	kwinner_layer(std::vector<size_t> input_shape, float density, float boost_factor = 0)
		: layer({vector_type::data}, {vector_type::data})
		, num_on_cells_(density*std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>()))
		, input_shape_(input_shape), boost_factor_(boost_factor)
		, count_active_(std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>())) {
	}

	std::string layer_type() const override {
		return "kwinner";
	}
	std::vector<shape3d> in_shape() const override {

		// return input shapes
		// order of shapes must be equal to argument of layer constructor
		return { shape3d(io_shape())};
	}

	std::vector<shape3d> out_shape() const override {
		return { shape3d(io_shape())}; // y
	}

	shape3d io_shape() const {
		auto s = input_shape_;
		for(size_t i=0; i<3;i++)
			s.push_back(1);
		return shape3d(s[0], s[1], s[3]);
	}

	void forward_propagation(const std::vector<tensor_t*>& in_data,
                         std::vector<tensor_t*>& out_data) override {
		const tensor_t &in = *in_data[0];
		tensor_t &out = *out_data[0];
		const size_t sample_count = in.size();
		if (indices_.size() < sample_count)
			indices_.resize(sample_count, std::vector<size_t>(num_on_cells_));

		vec_t boost_factors = vec_t(in[0].size(), 1);
		if(boost_factor_ != 0 && phase_ == net_phase::train) {
			for(size_t i=0;i<boost_factors.size();i++) {
				float target_density = (float)num_on_cells_/in[0].size();
				boost_factors[i] = exp((target_density-count_active_[i]/num_forwards_)*boost_factor_);
			}
		}
		
		for_i(sample_count, [&](size_t sample) {
			vec_t in_vec = in[sample];
			vec_t &out_vec = out[sample];
			
			for(size_t i=0;i<in_vec.size() && phase_ == net_phase::train;i++)
				in_vec[i] *= boost_factors[i];

			auto p = sort_permutation(in_vec, [](auto a, auto b){return a<b;});
			for(size_t i=0;i<out_vec.size();i++)
				out_vec[i] = 0;
			for(size_t i=0;i<num_on_cells_;i++) {
				size_t idx = p[i];
				out_vec[idx] = in_vec[idx];
				
				if(phase_ == net_phase::train)
					count_active_[idx]++;
			}

			std::copy(p.begin(), p.begin()+num_on_cells_, indices_[sample].begin());
		});
		num_forwards_ += in_data.size();
	}

	void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
		tensor_t &prev_delta       = *in_grad[0];
		const tensor_t &curr_delta = *out_grad[0];

		CNN_UNREFERENCED_PARAMETER(in_data);
		CNN_UNREFERENCED_PARAMETER(out_data);

		for_i(prev_delta.size(), [&](size_t sample) {
			auto& s = prev_delta[sample];
			size_t sz = s.size();
			for (size_t i = 0; i < sz; i++)
				s[i] = 0;
			for(size_t i=0;i<num_on_cells_;i++) {
				size_t idx = indices_[sample][i];
				s[idx] = curr_delta[sample][idx];
			}
		});
	}
	
	void set_context(net_phase ctx) override { phase_ = ctx; }

	size_t num_on_cells_;
	std::vector<size_t> input_shape_;
	std::vector<std::vector<size_t>> indices_;
	float boost_factor_;
	net_phase phase_;
	
	size_t num_forwards_ = 1;
	std::vector<float> count_active_;
};

}
