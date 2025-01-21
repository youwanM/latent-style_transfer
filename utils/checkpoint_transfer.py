import torch

def create_new_pth(file1, file2, output_file):
    # Load the files
    data1 = torch.load(file1, map_location="cpu")
    data2 = torch.load(file2, map_location="cpu")

    # Check if they are dictionaries
    if not isinstance(data1, dict) or not isinstance(data2, dict):
        print("Both .pth files must contain dictionaries.")
        return
    if output_file is not None:
        # Create a new dictionary for the output
        new_data = {}

        for key in data2.keys():
            value2 = data2[key]

            # If the key exists in file1 and is a tensor with the same shape
            if key in data1 and isinstance(data1[key], torch.Tensor) and isinstance(value2, torch.Tensor):
                value1 = data1[key]
                if value1.size() == value2.size():
                    new_data[key] = value1  # Copy the tensor from file1
                    continue

            # If key is not in file1 or shape doesn't match, create a zero tensor with the same shape as file2
            if isinstance(value2, torch.Tensor):
                new_data[key] = torch.zeros_like(value2)  # Create zero tensor with the same shape
            else:
                # For non-tensor keys, retain the value from file2
                new_data[key] = value2

        # Save the new dictionary to the output file
        torch.save(new_data, output_file)
        print(f"New .pth file saved to: {output_file}")

def compare_pth_files(file1, file2):
    # Load the files
    data1 = torch.load(file1, map_location="cpu")
    data2 = torch.load(file2, map_location="cpu")

    # Check if they are dictionaries
    if not isinstance(data1, dict) or not isinstance(data2, dict):
        print("The .pth files do not contain dictionaries. Unable to compare.")
        return

    # Compare keys
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())

    # Print keys only in one file
    only_in_file1 = keys1 - keys2
    only_in_file2 = keys2 - keys1

    if only_in_file1:
        print(f"Keys only in {file1}: {only_in_file1}")
    if only_in_file2:
        print(f"Keys only in {file2}: {only_in_file2}")

    # Compare shared keys
    shared_keys = keys1 & keys2
    for key in shared_keys:
        value1 = data1[key]
        value2 = data2[key]

        # Compare types
        if type(value1) != type(value2):
            print(f"Type mismatch for key '{key}': {type(value1)} vs {type(value2)}")
            continue

        # Compare tensor sizes and values if they are tensors
        if isinstance(value1, torch.Tensor):
            if value1.size() != value2.size():
                print(f"Size mismatch for key '{key}': {value1.size()} vs {value2.size()}")
            elif not torch.equal(value1, value2):
                print(f"Tensor values differ for key '{key}'")
        # Compare other types of values
        elif value1 != value2:
            print(f"Difference in values for key '{key}': {value1} vs {value2}")

if __name__ == "__main__":
    # Replace these with your file paths
    file1 = "./vae_checkpoints/Jan_15_2025_69.pth"
    file2 = "./vae_checkpoints/vae_test.pth"
    output_file = "./vae_checkpoints/vae_transfer.pth"

    create_new_pth(file1, file2, output_file)
    compare_pth_files(file1, output_file)