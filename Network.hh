#include <array>
#include <cstddef>

template <typename T, std::size_t width, std::size_t height>
class Array2D
{
public:
  explicit constexpr Array2D(float f) : c{f}
  {
  }

  constexpr T& get(std::size_t x, std::size_t y) noexcept
  {
    return this->c[y * width + x];
  }
  constexpr T const& get(std::size_t x, std::size_t y) const noexcept
  {
    return this->c[y * width + x];
  }
  constexpr auto begin() noexcept
  {
    return this->c.begin();
  }
  constexpr auto end() noexcept
  {
    return this->c.end();
  }

private:
  std::array<T, width * height> c;
};

template <std::size_t NUM_INPUTS,
          std::size_t NUM_NEURONS,
          std::size_t... OTHER_LAYERS>
class Network : public Network<NUM_INPUTS, NUM_NEURONS>
{
public:
  constexpr static auto learning_rate = 0.5f;

  constexpr void compute(std::array<float, NUM_INPUTS> const& inputs) noexcept
  {
    Network<NUM_INPUTS, NUM_NEURONS>::compute(inputs);
    // this->neurons is inherited from Network<NUM_INPUTS, NUM_NEURONS>
    this->subnet.compute(this->neurons);
  }

  constexpr auto const& getSubnet() const noexcept
  {
    return this->subnet;
  }

  constexpr auto const& getOutput() const noexcept
  {
    return this->subnet.getOutput();
  }

  // `expected` should be a std::array<float, N>, with N being the number of
  // neurons on the output layer of the network (last element of OTHER_LAYERS).
  constexpr void backprop(auto const& expected) noexcept
  {
    this->subnet.backprop(expected);
    auto const& sublayer_deltas = this->subnet.getDeltas();
    auto const sublayer_size = sublayer_deltas.size();
    auto const& sublayer_weights = this->subnet.getWeights();

    for (auto neuronidx = std::size_t{0}; neuronidx < NUM_NEURONS; ++neuronidx)
    {
      this->deltas[neuronidx] = 0.f;
      for (auto subneuronidx = std::size_t{0}; subneuronidx < sublayer_size;
           ++subneuronidx)
        this->deltas[neuronidx] +=
            sublayer_deltas[subneuronidx] *
            sublayer_weights.get(subneuronidx, neuronidx);
      for (auto inputidx = std::size_t{0}; inputidx < NUM_INPUTS; ++inputidx)
        this->weights.get(neuronidx, inputidx) +=
            learning_rate * this->deltas[neuronidx] *
            (this->neurons[neuronidx] > 0 ? 1 : 0);
    }
  }

private:
  Network<NUM_NEURONS, OTHER_LAYERS...> subnet;
};

template <std::size_t NUM_INPUTS, std::size_t NUM_NEURONS>
class Network<NUM_INPUTS, NUM_NEURONS>
{
public:
  constexpr static auto learning_rate = 0.5f;

  // For testing purpose only
  constexpr Network() noexcept : neurons{1.f}, deltas{1.f}, weights{1.f}
  {
    for (auto& neuron : this->neurons)
      neuron = 0.5f;
    for (auto& weight : this->weights)
      weight = 0.5f;
  }

  constexpr auto const& getNeurons() const noexcept
  {
    return this->neurons;
  }

  constexpr auto const& getDeltas() const noexcept
  {
    return this->deltas;
  }

  constexpr auto const& getWeights() const noexcept
  {
    return this->weights;
  }

  constexpr void compute(std::array<float, NUM_INPUTS> const& inputs) noexcept
  {
    for (auto neuronidx = std::size_t{0}; neuronidx < NUM_NEURONS; ++neuronidx)
    {
      auto net_input = float{0.f};
      for (auto inputidx = std::size_t{0}; inputidx < NUM_INPUTS; ++inputidx)
        net_input += inputs[inputidx] * this->weights.get(neuronidx, inputidx);
      this->neurons[neuronidx] = activation_function(net_input);
    }
  }

  constexpr void backprop(
      std::array<float, NUM_NEURONS> const& expected) noexcept
  {
    for (auto neuronidx = std::size_t{0}; neuronidx < NUM_NEURONS; ++neuronidx)
    {
      this->deltas[neuronidx] = expected[neuronidx] - this->neurons[neuronidx];
      for (auto inputidx = std::size_t{0}; inputidx < NUM_INPUTS; ++inputidx)
        this->weights.get(neuronidx, inputidx) +=
            learning_rate * this->deltas[neuronidx] * expected[neuronidx] *
            (this->neurons[neuronidx] > 0 ? 1 : 0);
    }
  }

  constexpr auto const& getOutput() const noexcept
  {
    return this->neurons;
  }

protected:
  constexpr static float activation_function(float net_input) noexcept
  {
    return net_input > 0 ? net_input : 0;
  }

  std::array<float, NUM_NEURONS> neurons;
  std::array<float, NUM_NEURONS> deltas;
  Array2D<float, NUM_NEURONS, NUM_INPUTS> weights;
};
