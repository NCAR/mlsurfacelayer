#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <vector>
#include <numeric>

// https://pytorch.org/tutorials/advanced/cpp_export.html
// https://github.com/pytorch/pytorch/issues/19786

__global__ void makeOnes(float * arr, int n)
{
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    if (id < n) arr[id] = 1.0f;
}

torch::Tensor ptrToTensor(float * ptr, const std::vector<int64_t>& dims)
{
    return torch::from_blob(ptr, dims, [] (void *) {}, torch::kCUDA);
}

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
        module.to(at::kCUDA);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    //inputs.push_back(torch::ones({1, 3, 224, 224}).to(at::kCUDA));
    float * d_inputs;
    std::vector<int64_t> dims = {100, 3, 224, 224};
    auto total = std::accumulate(std::begin(dims), std::end(dims), 1, std::multiplies<int64_t>());
    cudaMalloc(&d_inputs, total*sizeof(float));
    makeOnes<<<total/128 + 1, 128>>> (d_inputs, total);
    inputs.push_back(ptrToTensor(d_inputs, dims));

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.index({0, torch::indexing::Slice(0,5)}) << '\n';
}