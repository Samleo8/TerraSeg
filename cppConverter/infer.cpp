// NOTE: Code generated with Gemini and might be incorrect

#include <c10/cuda/CUDAGuard.h> // Managed CUDA stream synchronization
#include <chrono>
#include <iostream>
#include <memory>
#include <torch/script.h> // Core LibTorch header
#include <torch/torch.h>  // Complete PyTorch functional backend
#include <vector>

int main() {
    // 1. Verify that your system has access to a CUDA-supported GPU device
    if (!torch::cuda::is_available()) {
        std::cerr
            << "Fatal Error: CUDA-enabled GPU device not detected by LibTorch!"
            << std::endl;
        return -1;
    }

    // Explicitly target GPU device 0
    torch::Device device(torch::kCUDA, 0);

    // 2. Load the optimized compiled hybrid model engine
    torch::jit::script::Module module;
    try {
        // Load the flat-tensor wrapper script exported from Python
        module =
            torch::jit::load("models/terraseg_trt_optimized.ts", device);

        // Disable dropout and freeze batch normalization metrics
        module.eval();
        std::cout << "Successfully initialized TerraSeg TensorRT engine."
                  << std::endl;
    } catch (const c10::Error &e) {
        std::cerr << "Error parsing the model file: " << e.msg() << std::endl;
        return -1;
    }

    // 3. Mock incoming streaming data allocation
    // Replace this simulation block with your raw data stream (e.g., ROS
    // PointCloud2 or PCD files)
    static const int INPUT_DIM = 3; // NOTE: input dim should always be 3
    
    const int numPoints = 120000; // Total dynamic points in the current frame
    std::vector<float> simulated_coordinates(
        numPoints * 3, 1.23f); // Flattened continuous XYZ buffer
    std::vector<float> simulated_features(
        numPoints * INPUT_DIM, 0.85f); // Flattened continuous Feature buffer
    std::vector<int32_t> simulated_offsets = {
        numPoints}; // Cumulative point tracking for batch size 1

    /*
    ====================NOTE====================
    Input features must match the 3-dimensional vector described in the paper:
    (constant ones, normalized height z/5.0, normalized horizontal range | |xy |
    |/100.0). Raw(x, y, z) coordinates are reserved for constructing PTv3's
    spatial voxel grid.
    ==========================================
    */

    // 4. Wrap raw vectors into C++ Tensors instantly without memory allocation
    // penalties
    auto float_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto int_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device);

    // Create asynchronous Tensor pointers directly tracking your system vector
    // addresses
    torch::Tensor coord_tensor = torch::from_blob(simulated_coordinates.data(),
                                                  {numPoints, 3}, float_options);
    torch::Tensor feat_tensor = torch::from_blob(
        simulated_features.data(), {numPoints, INPUT_DIM}, float_options);
    torch::Tensor offset_tensor =
        torch::from_blob(simulated_offsets.data(), {1}, int_options);

    // 5. Structure the standard sequential payload list expected by your Python
    // wrapper signature
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(coord_tensor);
    inputs.push_back(feat_tensor);
    inputs.push_back(offset_tensor);

    // 6. Warm-up step to initialize TensorRT execution contexts
    std::cout << "Executing initial pipeline verification pass..." << std::endl;
    try {
        // Run forward pass once to initialize memory routes before entering the
        // timing cycle
        torch::Tensor warm_up_tensor = module.forward(inputs).toTensor();
        c10::cuda::getCurrentCUDAStream(device.index()).synchronize();
    } catch (const std::exception &e) {
        std::cerr << "Runtime failure during graph forward validation: "
                  << e.what() << std::endl;
        return -1;
    }

    // 7. High-resolution latency profiling loop
    std::cout << "Running benchmark step..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Call execution stream
    torch::Tensor output_logits =
        module.forward(inputs).toTensor(); // Returns 1D Matrix: (numPoints,)

    // Block C++ thread progress until the GPU calculations finish execution
    c10::cuda::getCurrentCUDAStream(device.index()).synchronize();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> execution_latency =
        end_time - start_time;
    std::cout << "Inference completed in: " << execution_latency.count()
              << " ms" << std::endl;

    // 8. Extract the predictions back into standard CPU pointer memory blocks
    // NOTE: 0 = ground and 1 = non-ground
    torch::Tensor binary_predictions = output_logits.lt(0.5).to(torch::kCPU);

    // Gain standard pointer access to raw binary boolean array
    bool *ground_mask_raw = binary_predictions.data_ptr<bool>();

    // 9. Inspect samples from the predicted layout
    std::cout << "First 10 Point Segmentations (0=Non-Ground, 1=Ground):"
              << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "Point ID [" << i << "]: "
                  << (ground_mask_raw[i] ? "Ground" : "Non-Ground")
                  << std::endl;
    }

    return 0;
}
