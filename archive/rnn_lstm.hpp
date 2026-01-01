#pragma once

#include <torch/torch.h>

/**
 * @brief Implementation details for the Long Short-Term Memory (LSTM) network.
 * This struct inherits from torch::nn::Module and defines the layers and 
 * the forward pass logic, mirroring the original Python class.
 */
struct LSTMImpl : torch::nn::Module {
    // --- Member Variables (Submodules) ---
    // Registered submodules will be managed by the parent Module.
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc1{nullptr};
    
    // --- Configuration Parameters ---
    // These must be stored as members to be accessible in the forward method.
    int64_t n_layers;
    int64_t hidden_dim;

    /**
     * @brief Constructor for the LSTM model.
     * @param input_dim The size of the input features.
     * @param hidden_dim The size of the hidden state (h_t) and cell state (c_t).
     * @param embed_dim (Unused in the current logic, included for signature consistency).
     * @param n_layers The number of recurrent layers.
     * @param output_dim The size of the output features after the final linear layer.
     * @param n_heads (Unused in the current logic).
     */
    LSTMImpl(
        int64_t input_dim, 
        int64_t hidden_dim, 
        int64_t embed_dim, 
        int64_t n_layers, 
        int64_t output_dim, 
        int64_t n_heads = 8
    );

    /**
     * @brief Defines the computation performed at every call.
     * @param x The input tensor (Batch, Sequence, Features).
     * @return The output tensor after processing (Batch, Output_Dim).
     */
    torch::Tensor forward(torch::Tensor x);
};

// --- TORCH_MODULE Macro ---
/**
 * @brief Creates a smart-pointer wrapper for LSTMImpl.
 * This is the standard LibTorch way to define a module type that can be 
 * easily passed around and managed.
 */
TORCH_MODULE(LSTM);