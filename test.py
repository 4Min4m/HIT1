import numpy as np
import pickle

print("--- Inspecting NPZ data arrays ---")
try:
    data = np.load('/home/student/myprojects/HIT1/dataset/test.npz', allow_pickle=True)

    # data.files lists only the data arrays
    if not data.files:
        print("No data arrays found in data.files. Listing all keys instead:")
        print(list(data.keys()))

    for key in data.files:
        try:
            array = data[key]
            print(f"Key: {key}, Shape: {array.shape}, Dtype: {array.dtype}")
        except Exception as e:
            print(f"Error reading array key {key}: {e}")

    print("\n--- Inspecting for pickled files ---")
    # Check all keys for .pkl files
    pkl_keys = [k for k in data.keys() if k.endswith('.pkl')]

    if not pkl_keys:
        print("No .pkl files found in the archive.")

    for key in pkl_keys:
        print(f"Found pickled file: {key}")
        try:
            # data[key] for a pkl file might be bytes
            pkl_data = data[key]

            # If it's bytes, we need to load it with pickle
            if isinstance(pkl_data, bytes):
                content = pickle.loads(pkl_data)
                print(f"  -> Contents of {key}: {content}")
            else:
                # It might be an object that .item() can retrieve
                item = pkl_data.item()
                print(f"  -> Contents of {key}: {item}")

        except Exception as e:
            print(f"  -> Could not read pickled file {key}: {e}")

except Exception as e:
    print(f"Failed to load NPZ file: {e}")

print("--- End of inspection ---")