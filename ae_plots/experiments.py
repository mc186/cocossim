import subprocess

# TPU V3 configuration
sim_cmd_base = "./../build/perf_model -sa_sz 128 -vu_sz 1024 -ws 1 -f 0.94"  
input_file = "input.txt"
output_file = "out.txt"
log_file = "results.txt"

# Workloads configuration
workloads = {
    "Matmul": [(4096, 320, 320), (1024, 640, 640), (256, 1280, 1280), (64, 1280, 1280)],
    "DotProduct": [(8, 4096, 40, 4096), (8, 1024, 80, 1024), (8, 256, 160, 256), (8, 64, 160, 64)],
    "Conv": [(4096, 2880, 320), (1024, 5760, 640), (256, 11520, 1280), (64, 11520, 1280)],
    "Softmax": [(8, 4096), (8, 1024), (8, 256), (8, 64)],
    "ResNet": [
        [("LayerNorm", (32, 4096, 320)), ("Conv", (4096, 2880, 320)), ("Activation", (4096, 320)), 
         ("LayerNorm", (32, 4096, 320)), ("Conv", (4096, 2880, 320)), ("Activation", (4096, 320))],
        [("LayerNorm", (32, 1024, 640)), ("Conv", (1024, 5760, 640)), ("Activation", (1024, 640)), 
         ("LayerNorm", (32, 1024, 640)), ("Conv", (1024, 5760, 640)), ("Activation", (1024, 640))],
        [("LayerNorm", (32, 256, 1280)), ("Conv", (256, 11520, 1280)), ("Activation", (256, 1280)), 
         ("LayerNorm", (32, 256, 1280)), ("Conv", (256, 11520, 1280)), ("Activation", (256, 1280))],
        [("LayerNorm", (32, 64, 1280)), ("Conv", (64, 11520, 1280)), ("Activation", (64, 1280)), 
         ("LayerNorm", (32, 64, 1280)), ("Conv", (64, 11520, 1280)), ("Activation", (64, 1280))],

    ],
    "SelfAttention": [(8, 4096, 320, 320), (8, 1024, 640, 640), (8, 256, 1280, 1280), (8, 64, 1280, 1280)],
}

# Operation mapping (optional, e.g., "DotProduct" -> "Matmul")
op_mapping = {
    "DotProduct": "Matmul",
    "Conv": "Conv"
}

def run_simulation(op, dims):
    """Runs the simulation for a single operation and its dimensions."""
    sim_op = op_mapping.get(op, op)
    
    # Write to the input file based on the number of dimensions
    with open(input_file, "w") as f:
        if len(dims) == 2:
            f.write(f"{sim_op} {dims[0]} {dims[1]}\n")
        elif len(dims) == 3:
            f.write(f"{sim_op} {dims[0]} {dims[1]} {dims[2]}\n")
        elif len(dims) == 4:
            f.write(f"{sim_op} {dims[0]} {dims[1]} {dims[2]} {dims[3]}\n")
        else:
            raise ValueError(f"Unexpected number of dimensions: {len(dims)} for operation {op}")

    # Run the simulation command
    full_cmd = f"{sim_cmd_base} -c 1 -i {input_file} -o {output_file}"
    subprocess.run(full_cmd.split(), check=True)

    # Read and return the result from the output file
    with open(output_file, "r") as f:
        result = f.read()

    return result.strip()

def run_resnet_workload(resnet_stage):
    """Runs a composite ResNet workload, combining all ops into a single simulation."""
    
    # Write all operations for the current ResNet stage into input.txt
    with open(input_file, "w") as f:
        for op, dims in resnet_stage:
            sim_op = op_mapping.get(op, op)
            if len(dims) == 2:
                f.write(f"{sim_op} {dims[0]} {dims[1]}\n")
            elif len(dims) == 3:
                f.write(f"{sim_op} {dims[0]} {dims[1]} {dims[2]}\n")
            elif len(dims) == 4:
                f.write(f"{sim_op} {dims[0]} {dims[1]} {dims[2]} {dims[3]}\n")
            else:
                raise ValueError(f"Unexpected number of dimensions: {len(dims)} for operation {op}")

    # Now run the composite simulation for this stage
    full_cmd = f"{sim_cmd_base} -c 4 -i {input_file} -o {output_file}"
    subprocess.run(full_cmd.split(), check=True)

    # Read and return the result from the output file
    with open(output_file, "r") as f:
        result = f.read()

    return result.strip()

def run_workload(workload_name, dims_list):
    """Run through all operations in the given workload."""
    combined_output = []
    
    # Check if the dims_list is composite (like ResNet)
    if isinstance(dims_list[0], list):
        # Iterate through each stage in the ResNet (list of operations)
        for stage in dims_list:
            try:
                # For ResNet, run all ops together as a composite
                output = run_resnet_workload(stage)
                combined_output.append(f"ResNet Stage (0) -> {output}")
            except subprocess.CalledProcessError as e:
                combined_output.append(f"ResNet Stage -> ERROR: {e}")
    else:
        # Iterate through the simple dims_list (non-ResNet)
        for dims in dims_list:
            try:
                output = run_simulation(workload_name, dims)
                combined_output.append(f"{workload_name} {dims} -> {output}")
            except subprocess.CalledProcessError as e:
                combined_output.append(f"{workload_name} {dims} -> ERROR: {e}")

    return combined_output

def main():
    with open(log_file, "w") as log:
        for workload_name, dims_list in workloads.items():
            log.write(f"Running {workload_name} workload:\n")
            
            results = run_workload(workload_name, dims_list)
            for r in results:
                log.write(f"  {r}\n")

if __name__ == "__main__":
    main()
