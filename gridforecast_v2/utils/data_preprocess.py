import numpy as np


def norm(data, mean=0.5, std=0.25):
    '''
    func: normalize the data to the distribution of norm(mean, std)
    '''
    ori_mean = np.mean(data)
    ori_std = np.std(data)
    y = data - ori_mean
    y = y / ori_std if ori_std != 0 else y
    y = y * std + mean
    return y


def standard_scaler(self, value, mu, scale):
    return (value - mu) / scale


def de_standard_scaler(value, mu, scale):
    return value * scale + mu


def ds_to_float32(ds):
    for var in ds.data_vars.keys():
        ds[var] = ds[var].astype('float32')
    return ds


def zip_ds(ds):
    for var in ds.data_vars:
        ds[var].encoding.update({'zlib': True, 'complevel': 1})
    return ds


def to_chunked_dataset(ds, chunking):
    """
    Create a chunked copy of a Dataset with proper encoding for netCDF export.
    :param ds: xarray.Dataset
    :param chunking: dict: chunking dictionary as passed to
        xarray.Dataset.chunk()
    :return: xarray.Dataset: chunked copy of ds with proper encoding
    """
    chunk_dict = dict(ds.dims)
    chunk_dict.update(chunking)
    for var in ds.data_vars:
        if 'coordinates' in ds[var].encoding:
            del ds[var].encoding['coordinates']
        ds[var].encoding['contiguous'] = False
        ds[var].encoding['original_shape'] = ds[var].shape
        ds[var].encoding['chunksizes'] = \
            tuple([chunk_dict[d] for d in ds[var].dims])
    return ds
