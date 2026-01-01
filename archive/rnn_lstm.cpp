#include "rnn_lstm.hpp"
#include <iostream>

// --- Constructor Implementation ---
LSTMImpl::LSTMImpl(
    int64_t input_dim, 
    int64_t hidden_dim, 
    int64_t embed_dim, 
    int64_t n_layers, 
    int64_t output_dim, 
    int64_t n_heads
) : n_layers(n_layers), hidden_dim(hidden_dim) {
    
    // Initialize LSTM
    lstm = register_module("lstm", torch::nn::LSTM(
        torch::nn::LSTMOptions(input_dim, hidden_dim)
        .num_layers(n_layers)
        .batch_first(true)
    ));

    // Initialize Linear layer
    fc1 = register_module("fc1", torch::nn::Linear(hidden_dim, output_dim));
}

// --- Forward Function Implementation ---
torch::Tensor LSTMImpl::forward(torch::Tensor x) {
    // 1. Initialize hidden and cell states (h0, c0)
    auto device = x.device();
    auto h0 = torch::zeros({n_layers, x.size(0), hidden_dim}, torch::TensorOptions().device(device));
    auto c0 = torch::zeros({n_layers, x.size(0), hidden_dim}, torch::TensorOptions().device(device));

    // 2. Forward pass through LSTM
    auto lstm_out = lstm->forward(x, std::make_tuple(h0, c0));
    auto out = std::get<0>(lstm_out); 

    // 3. Slice the output: out[:, -1, :]
    auto last_time_step = out.select(1, -1);

    // 4. Pass through Linear layer
    auto final_out = fc1->forward(last_time_step);

    // 5. Squeeze
    return final_out.squeeze(1);
}

// Example Usage (in main function)
int main() {
    // Model parameters
    int64_t input_dim = 10;
    int64_t hidden_dim = 20;
    int64_t embed_dim = 0; 
    int64_t n_layers = 2;
    int64_t output_dim = 1;

    // Instantiate model using the TORCH_MODULE wrapper
    LSTM model(input_dim, hidden_dim, embed_dim, n_layers, output_dim);

    // Create a dummy input: (Batch Size=5, Sequence Length=15, Input Dim=10)
    auto x = torch::randn({5, 15, input_dim});

    // Forward pass
    auto output = model->forward(x);

    std::cout << "Model successfully compiled and ran." << std::endl;
    std::cout << "Input shape: " << x.sizes() << std::endl;
    std::cout << "Output shape: " << output.sizes() << std::endl;

    return 0;
}