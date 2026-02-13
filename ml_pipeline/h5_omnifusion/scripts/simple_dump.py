import h5py
import sys

def dump_h5(filepath):
    print(f"DUMPING: {filepath}")
    if not h5py.is_hdf5(filepath):
        print("NOT A VALID HDF5 FILE")
        return

    with h5py.File(filepath, 'r') as f:
        def visitor(name, obj):
            print(f"{name} : {obj}")
        f.visititems(visitor)

if __name__ == "__main__":
    dump_h5(r'c:\Users\thela\OneDrive\Desktop\phase 2\ml_pipeline\h5_omnifusion\output file folder\300.h5')
