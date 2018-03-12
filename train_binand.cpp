#include <cstdint>
#include <iostream>
#include <random>

#include "Network.hh"

namespace
{
// seed() function and RNG class are adapted from Jason Turner's C++ Weekly -
// Ep 44
constexpr uint64_t seed()
{
  auto shifted = uint64_t{0};

  for (auto const c : __TIME__)
    shifted = (shifted << 8) | c;
  return shifted;
}

class RNG
{
public:
  using result_type = uint32_t;

  constexpr result_type operator()() noexcept
  {
    auto const oldstate = this->state;
    this->state = oldstate * 6364136223846793005ULL + (this->inc | 1);
    auto const xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
    auto const rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
  }

private:
  uint64_t state{0};
  uint64_t inc{seed()};
};

constexpr auto trainNetwork()
{
  auto const CORRECT_IN_A_ROW_THRESHOLD{3};
  auto network = Network<2, 2, 1>{};
  auto rand = RNG{};
  auto correct_in_a_row{0};

  while (correct_in_a_row < CORRECT_IN_A_ROW_THRESHOLD)
  {
    auto const random_number = rand();
    auto const left_input = bool(random_number & 1);
    auto const right_input = bool(random_number & 2);
    auto const expected = left_input & right_input;
    auto const input =
        std::array<float, 2>{{left_input ? 1.f : 0.f, right_input ? 1.f : 0.f}};
    network.compute(input);
    auto const output = network.getOutput()[0] > 0.5f;
    auto const expected_array = std::array<float, 1>{{expected ? 1.f : 0.f}};
    network.backprop(expected_array);
    if (output == expected)
      ++correct_in_a_row;
    else
      correct_in_a_row = 0;
  }
  return network;
}
}

int main(int argc, char const* const* argv)
{
  auto rng = std::mt19937{std::random_device{}()};
  auto rand = std::uniform_int_distribution<>(0, 1);
  constexpr auto trained_network = trainNetwork();
  auto network = trained_network;
  auto correct_guesses = 0u;
  auto wrong_guesses = 0u;
  for (auto i = 0; i < 100000; ++i)
  {
    auto const left_input = bool(rand(rng));
    auto const right_input = bool(rand(rng));
    auto const expected = left_input & right_input;
    auto const input =
        std::array<float, 2>{{left_input ? 1.f : 0.f, right_input ? 1.f : 0.f}};
    network.compute(input);
    auto const output = network.getOutput()[0] > 0.5f;
    std::cout << network.getOutput()[0] << std::endl;
    if (output == expected)
      ++correct_guesses;
    else
      ++wrong_guesses;
  }
  std::cout << "Correct guesses: " << correct_guesses << '/'
            << correct_guesses + wrong_guesses << " --- "
            << (correct_guesses /
                static_cast<float>(correct_guesses + wrong_guesses)) *
                   100.f
            << '%' << std::endl;
  return 0;
}
