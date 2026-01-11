import h5py
import os
import hdf5plugin  # Import the plugin to ensure compatibility
import h5py
from pprint import pprint

# Set the HDF5_PLUGIN_PATH environment variable to an empty string to disable plugins
os.environ["HDF5_PLUGIN_PATH"] = os.path.dirname(hdf5plugin.__file__)

import h5py

def inspect_hdf5(file_path, show_data=False, max_elements=10):
    """
    Inspect the structure and contents of an HDF5 file.

    Args:
    - file_path (str): Path to the HDF5 file.
    - show_data (bool): Whether to display the first few elements of datasets. Default is False.
    - max_elements (int): Maximum number of elements to display if `show_data` is True.
    """
    try:
        with h5py.File(file_path, 'r') as h5_file:
            print(f"\nInspecting HDF5 file: {file_path}\n")
            
            def print_group(name, obj):
                print(f"{name}:")
                if isinstance(obj, h5py.Group):
                    print("  Group")
                elif isinstance(obj, h5py.Dataset):
                    print(f"  Dataset - shape: {obj.shape}, dtype: {obj.dtype}")
                    if show_data:
                        # Handle scalar datasets
                        if obj.shape == ():  # Scalar dataset
                            print(f"  Data: {obj[()]}")
                        else:  # Array dataset
                            data = obj[...]
                            print(f"  Data (first {max_elements} elements): {data[:max_elements]}")
                else:
                    print("  Unknown type")
                
                # Print attributes if available
                if obj.attrs:
                    print("  Attributes:")
                    for key, value in obj.attrs.items():
                        print(f"    {key}: {value}")
            
            h5_file.visititems(print_group)

    except Exception as e:
        print(f"Error inspecting HDF5 file: {e}")

if __name__ == "__main__":
    # Example Usage
    inspect_hdf5("events.h5", show_data=True, max_elements=5)