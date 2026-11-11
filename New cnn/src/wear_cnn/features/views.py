"""HDF5 feature subsets without duplicating the underlying feature arrays."""

import h5py


def feature_view(group, source, source_file, columns):
    shape = source.shape
    layout = h5py.VirtualLayout(shape=(*shape[:-1], len(columns)), dtype=source.dtype)
    virtual = h5py.VirtualSource(str(source_file), source.name, shape=shape)
    for destination, column in enumerate(columns):
        layout[..., destination] = virtual[..., int(column)]
    return group.create_virtual_dataset("values", layout)
