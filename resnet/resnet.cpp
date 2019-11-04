#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

// Where to find the MNIST dataset.
const char* kDataRoot = "/home/testroot/CNN/test_libtorch/resnet/data/";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 10;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kernel_size,
                                      int64_t stride=1, int64_t padding=1, bool with_bias=false){
  torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kernel_size);
  conv_options.stride(stride);
  conv_options.padding(padding);
  conv_options.with_bias(with_bias);
  return conv_options;
}


struct BasicBlock : torch::nn::Module {

  static const int expansion;

  int64_t stride;
  torch::nn::Conv2d conv1;
  torch::nn::BatchNorm bn1;
  torch::nn::Conv2d conv2;
  torch::nn::BatchNorm bn2;
  torch::nn::Sequential downsample;

  BasicBlock(int64_t inplanes, int64_t planes, int64_t stride_=1,
             torch::nn::Sequential downsample_=torch::nn::Sequential())
      : conv1(conv_options(inplanes, planes, 3, stride_, 1)),
        bn1(planes),
        conv2(conv_options(planes, planes, 3, 1, 1)),
        bn2(planes),
        downsample(downsample_)
        {
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    stride = stride_;
    if (!downsample->is_empty()){
      register_module("downsample", downsample);
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    at::Tensor residual(x.clone());
    
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);

    x = conv2->forward(x);
    x = bn2->forward(x);

    if (!downsample->is_empty()){
      residual = downsample->forward(residual);
    }

    x += residual;
    x = torch::relu(x);

    return x;
  }
};

const int BasicBlock::expansion = 1;


struct BottleNeck : torch::nn::Module {

  static const int expansion;

  int64_t stride;
  torch::nn::Conv2d conv1;
  torch::nn::BatchNorm bn1;
  torch::nn::Conv2d conv2;
  torch::nn::BatchNorm bn2;
  torch::nn::Conv2d conv3;
  torch::nn::BatchNorm bn3;
  torch::nn::Sequential downsample;

  BottleNeck(int64_t inplanes, int64_t planes, int64_t stride_=1,
             torch::nn::Sequential downsample_=torch::nn::Sequential())
      : conv1(conv_options(inplanes, planes, 1, 1, 0)),
        bn1(planes),
        conv2(conv_options(planes, planes, 3, stride_, 1)),
        bn2(planes),
        conv3(conv_options(planes, planes * expansion , 1, 1, 0)),
        bn3(planes * expansion),
        downsample(downsample_)
        {
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    register_module("conv3", conv3);
    register_module("bn3", bn3);
    stride = stride_;
    if (!downsample->is_empty()){
      register_module("downsample", downsample);
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    at::Tensor residual(x.clone());

    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);

    x = conv2->forward(x);
    x = bn2->forward(x);
    x = torch::relu(x);

    x = conv3->forward(x);
    x = bn3->forward(x);

    if (!downsample->is_empty()){
      residual = downsample->forward(residual);
    }

    x += residual;
    x = torch::relu(x);

    return x;
  }
};

const int BottleNeck::expansion = 4;


template <class Block> struct ResNet : torch::nn::Module {

  int64_t inplanes = 64;
  torch::nn::Conv2d conv1;
  torch::nn::BatchNorm bn1;
  torch::nn::Sequential layer1;
  torch::nn::Sequential layer2;
  torch::nn::Sequential layer3;
  torch::nn::Sequential layer4;
  torch::nn::Linear fc;

  ResNet(torch::IntList layers, int64_t num_classes=1000)
      : conv1(conv_options(1, 64, 7, 2, 3)),
        bn1(64),
        layer1(_make_layer(64, layers[0])),
        layer2(_make_layer(128, layers[1], 2)),
        layer3(_make_layer(256, layers[2], 2)),
        layer4(_make_layer(512, layers[3], 2)),
        fc(512 * Block::expansion, num_classes)
        {
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("fc", fc);

    /*
    // Initializing weights
    for(auto m: this->modules()){
      if (m.value.name() == "torch::nn::Conv2dImpl"){
        for (auto p: m.value.parameters()){
          torch::nn::init::xavier_normal_(p.value);
        }
      }
      else if (m.value.name() == "torch::nn::BatchNormImpl"){
        for (auto p: m.value.parameters()){
          if (p.key == "weight"){
            torch::nn::init::constant_(p.value, 1);
          }
          else if (p.key == "bias"){
            torch::nn::init::constant_(p.value, 0);
          }
        }
      }
    }
    */
  }

  torch::Tensor forward(torch::Tensor x){

    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    x = torch::max_pool2d(x, 3, 2, 1);

    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);

    x = torch::avg_pool2d(x, 7, 1);
    x = x.view({x.sizes()[0], -1});
    x = fc->forward(x);

    return x;
  }


private:
  torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride=1){
    torch::nn::Sequential downsample;
    if (stride != 1 or inplanes != planes * Block::expansion){
      downsample = torch::nn::Sequential(
          torch::nn::Conv2d(conv_options(inplanes, planes * Block::expansion, 1, stride)),
          torch::nn::BatchNorm(planes * Block::expansion)
      );
    }
    torch::nn::Sequential layers;
    layers->push_back(Block(inplanes, planes, stride, downsample));
    inplanes = planes * Block::expansion;
    for (int64_t i = 0; i < blocks; i++){
      layers->push_back(Block(inplanes, planes));
    }

    return layers;
  }
};


ResNet<BasicBlock> resnet18(){
  ResNet<BasicBlock> model({2, 2, 2, 2});
  return model;
}

ResNet<BasicBlock> resnet34(){
  ResNet<BasicBlock> model({3, 4, 6, 3});
  return model;
}

ResNet<BottleNeck> resnet50(){
  ResNet<BottleNeck> model({3, 4, 6, 3});
  return model;
}

ResNet<BottleNeck> resnet101(){
  ResNet<BottleNeck> model({3, 4, 23, 3});
  return model;
}

ResNet<BottleNeck> resnet152(){
  ResNet<BottleNeck> model({3, 8, 36, 3});
  return model;
}

template <typename DataLoader>
void train(
    size_t epoch,
    ResNet<BottleNeck>& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void test(
    ResNet<BottleNeck>& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
}

auto main() -> int {
  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  ResNet<BottleNeck> model = resnet101();
  model.to(device);

  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset = torch::data::datasets::MNIST(
                          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
  }
}
