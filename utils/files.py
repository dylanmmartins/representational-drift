# -*- coding: utf-8 -*-
"""
representational_drift/utils/files.py

File operations.

DMM, October 2026
"""

import os
import fnmatch
import h5py
import datetime
import numpy as np


def time2str(time_array):

    fmt = '%Y-%m-%d-%H-%M-%S-%f'

    if type(time_array) == datetime.datetime:
        return time_array.strftime(fmt)


    out = []

    for t in time_array:
        tstr = t.strftime(fmt)
        out.append(tstr)

    return out


def write_h5(filename, dic):

    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """ Recursively write dict or list items into an HDF5 group. """

    if isinstance(dic, dict):
        iterator = dic.items()
    elif isinstance(dic, list):
        iterator = enumerate(dic)
    else:
        ValueError('Cannot save {} type'.format(type(dic)))
    for key, item in iterator:
        key = str(key)
        if isinstance(item, (np.ndarray, np.number, np.bool_, int, float, str, bytes, bool)):
            try:
                h5file[path + key] = item
            except TypeError:
                if isinstance(item, np.ndarray) and (item.dtype == object):
                    recursively_save_dict_contents_to_group(h5file, path + key + '/', item.item())
        elif isinstance(item, dict) or isinstance(item, list):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        elif isinstance(item, datetime.datetime):
            h5file[path + key] = time2str(item)
        else:
            raise ValueError('Cannot save {} type'.format(type(item)))


def recursively_load_dict_contents_from_group(h5file, path):
    """ Recursively read an HDF5 group into a nested dict. """

    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file,
                path + key + '/'
            )

    return ans


def read_h5(filename, aslist=False):
    """ Read an HDF5 file into a nested dict (or list if aslist=True).

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    aslist : bool
        If True, convert the top-level dict to a list using integer keys as indices.

    Returns
    -------
    out : dict or list
    """

    with h5py.File(filename, 'r') as h5file:
        out = recursively_load_dict_contents_from_group(h5file, '/')
        if aslist:
            outl = [None for l in range(len(out.keys()))]
            for key, item in out.items():
                outl[int(key)] = item
            out = outl

        return out


def find(pattern, path, MR=False, retempty=False):
    """ Recursively glob for files under `path` matching `pattern`.

    Parameters
    ----------
    pattern : str
        fnmatch-style pattern, e.g. '*sparsenoise.csv'.
    path : str
        Directory to search, including subdirectories.
    MR : bool
        If True, return only the most recently modified match (as a str).
        Otherwise return a list of all matches.
    retempty : bool
        If True, return None instead of raising when nothing matches.
    """

    result = []
    for root, _, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))

    if len(result) == 0:
        if retempty:
            return None
        raise FileNotFoundError('Found no file(s) matching key {} in directory {}'.format(pattern, path))

    if MR:
        return max(result, key=os.path.getmtime)
    return result
