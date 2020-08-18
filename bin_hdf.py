import h5py
from data_interface_class import *

def bin_hdf(inp_file, save_path, time_window, chunk_size, ovr):
    run1 = io_object(time_window, chunk_size, inp_file = inp_file, save_path = save_path, overwrite = ovr)
    for batch_num in range(run1.num_batches):
        run1.read_batch(batch_num, 1)
        f = h5py.File(inp_file)
        run1.output_data_batch = f['data'][:, batch_num*chunk_size:(batch_num + 1) * chunk_size ]
        run1.generate_header()
        run1.generate_compressed_data()
        run1.write_bin_file()
