#include "torch/nn/functional/activation.h"
#include "torch/script.h"
#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, const char *argv[]) {
  if (argc != 2) {
    std::cerr << " usage: ts-infer <path-to-exported-model>" << std::endl;
    return -1;
  }
  std::cout << "loading model..." << std::endl;

  //反序列化模型
  torch::jit::script::Module module;
  try {
    module = torch::jit::load(argv[1]);
  } catch (const c10::Error &e) {
    std::cerr << "Error loading model\n";
    std::cerr << e.what_without_backtrace();
    return -1;
  }

  // 关掉自动求导
  // 设置eval模式, 关掉dropout
  std::cout << "model loaded successfully\n";
  torch::NoGradGuard no_grad;
  module.eval();

//   module.to(at::kCUDA);

  // dummy image
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::rand({20, 3, 1024, 1024}));

//   inputs.push_back(torch::rand({20, 3, 1024, 1024}).to(at::kCUDA));

  auto t0 = std::chrono::high_resolution_clock::now();
  // 前向推导
  at::Tensor output;
  for (int i = 0; i < 1; ++i) {
    output = module.forward(inputs).toTensor();
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout
      << "cost time:"
      << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
      << "ms" << std::endl;
  std::cout << "output.dim:" << output.dim() << std::endl;
  std::cout << "output.sizes:" << output.sizes() << std::endl;

  // 按照dim=1,进行求softmax,转化为概率值.最后找出概率前5的索引
  namespace F = torch::nn::functional;
  at::Tensor output_sm = F::softmax(output, F::SoftmaxFuncOptions(1));
  std::tuple<at::Tensor, at::Tensor> top5_tensor = output_sm.topk(5);
  at::Tensor top5 = std::get<1>(top5_tensor);

  std::cout << top5[0][0].item() << '\n';
  std::cout << "\nDONE\n";
  std::this_thread::sleep_for(std::chrono::seconds(3));
  return 0;
}