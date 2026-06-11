'''
Convert the TerraSeg PyTorch model to a TorchScript format optimized for C++ deployment using Torch-TensorRT. This script compiles the model with dynamic input shapes and enables FP16 precision for improved inference performance on compatible NVIDIA GPUs. The resulting 'terraseg_trt_optimized.ts' file can be loaded in C++ applications using the TorchScript API, allowing for efficient execution of the TerraSeg model in production environments.
'''

import argparse
import torch
import torch_tensorrt
from terraseg_lib.src.model import build_terraseg, TerraSeg
from terraseg_lib.src.norm import replace_bn1d_with_gn

def parse_args():
    parser = argparse.ArgumentParser(description="Convert TerraSeg model to TorchScript")
    
    parser.add_argument("--variant", type=str, default="S", choices=["B", "S", "b", "s"], 
                        help="TerraSeg variant to build (B or S)")
    parser.add_argument("--output_path", type=str, default="models/terraseg_trt_optimized.ts", 
                        help="Path to save the optimized TorchScript model")

    parser.add_argument("--min_points", type=int, default=1000,
                        help="Minimum number of points for dynamic input shape")
    parser.add_argument("--opt_points", type=int, default=300000,
                        help="Optimal number of points for dynamic input shape")
    parser.add_argument("--max_points", type=int, default=1000000, 
                        help="Maximum number of points for dynamic input shape")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load the model configuration and weights
    model = build_terraseg(args.variant.upper())
    model = replace_bn1d_with_gn(model=model, groups_default=32)  # Replace BatchNorm1d with GroupNorm for stability
    
    model_path = f"models/terraseg_{args.variant.lower()}.pth"
    checkpoint = torch.load(model_path, map_location="cuda", weights_only=False) # TODO: predictor code originally used CPU?
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    
    model.to("cuda")  # Move model to GPU for TensorRT optimization
    model.eval()  # Set model to evaluation mode

    # 2. Define the expected dynamic input specifications
    '''
    Model Inputs:
    - "coord" (torch.Tensor) : (N_tot,3) XYZ coordinates. Unit: meters.
    - "feat" (torch.Tensor): (N_tot, input_dim) per-point features.
    - "offset" (torch.Tensor): (B,) cumulative point counts per scan.
    - "grid_size" (float): Voxel grid size. Unit: meters.
    - "condition" (list[str]): Per-scan dataset name tags.
    
    Note:
        Input features must match the 3-dimensional vector described in the paper:
        (constant ones, normalized height z/5.0, normalized horizontal range | |xy | |/100.0).
        Raw(x, y, z) coordinates are reserved for constructing PTv3's spatial voxel grid.
    '''
    
    inputs = [
        torch_tensorrt.Input(
            min_shape=(args.min_points, 3),
            opt_shape=(args.opt_points, 3),
            max_shape=(args.max_points, 3)
        ), # "coord" input
        torch_tensorrt.Input(
            min_shape=(args.min_points, model.input_dim),
            opt_shape=(args.opt_points, model.input_dim),
            max_shape=(args.max_points, model.input_dim)
        ), # "feat" input
        torch_tensorrt.Input(
            min_shape=(1,), opt_shape=(1,), max_shape=(1,),
            dtype=torch.int32
            # Maps to offset: 
            # "offset" input (for 1 frame batch size, this is always an array of size 1)
        )
    ]

    # Pass 'grid_size' as a kwarg_input to isolate it from the dynamic tensor shapes
    # Match the voxel size used during training (default: 5cm)
    kwarg_inputs = {"grid_size": 0.05}

    # 3. Compile the model via TorchScript backend with structural fallback routing
    # Unsupported custom PTv3 ops will fall back to native PyTorch CUDA seamlessly
    trt_model = torch_tensorrt.compile(
        model,
        ir="torchscript",
        inputs=inputs,
        kwarg_inputs=kwarg_inputs,
        enabled_precisions={torch.float32} # NOTE: FP16 precision is currently unstable for PTv3's sparse-conv path, so we use FP32 for TerraSeg
    )

    # 4. Serialize the optimized pipeline for C++ ingestion
    trt_model.save(args.output_path)
    print(f"Optimization complete. '{args.output_path}' generated successfully.")

if __name__ == "__main__":
    main()