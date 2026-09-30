"""
Convert a directory of .tif files to a multi-page .tif file

Functions
---------
tif_convert(firstfile=None, savepath=None, delete_singles=False, ret=False, saveas='tif', multicycle=False)
    Convert a sequence of single tif pages to a 3D tif stack.

Example usage
-------------
    $ python -m imgtools.tif_convert --filepath T:/path/to/file.tif
or using dialog boxes:
    $ python -m imgtools.tif_convert

Author: DMM, 2024
"""


import os
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import tifffile
from scipy.io import savemat

from utils.gui_funcs import select_file


def tif_convert(firstfile, savepath=None, delete_singles=False,
                ret=False, saveas='tif', multicycle=False):
    """ Convert a sequence of single tif pages to a 3D tif stack.

    Parameters
    ----------
    firstfile : str
        File path to the first tif file. If None, a dialog box opens in which the file is
        selected. Default is None.
    savepath : str
        Path to the directory in which the new tif stack will be written.
    delete_singles : bool
        If True, the individual tif files be deleted once they are written into the 3D stack.
        Default is False.
    ret : bool
        If True, the stack will be returned as an array.
    saveas : str
        File format to save the stack as. Options are 'tif', 'mat', or 'npy'.
    multicycle : bool
        Allow for more strict file path searches for multicycle imaging. Default is False. This
        option has not been tested much.
    
    Returns
    -------
    savefilename : str
        Full file path that the stack was saved to.
    imgstack : np.ndarray
        The stack returned as an array. This is only returned if `ret` is True.
    """

    firstfile = os.path.normpath(firstfile)

    head, tail = os.path.split(firstfile)
    
    if savepath is None:
        savepath = head
    else:
        savepath = os.path.normpath(savepath)

    base_filename = '_'.join(tail.split('_')[:-1])
    add_ext = '.'.join(firstfile.split('.')[1:])

    file_list = []

    if multicycle is True:
        file_list = glob(os.path.join(head, '*.ome.tif'))
        file_list = [f for f in file_list if 'Reference' not in f]
        file_list = sorted(file_list)

    elif multicycle is False:

        nF = 0
        while True:
            nF += 1
            test_name = '{}_{:06d}.{}'.format(
                base_filename,
                nF,
                add_ext
            )
            testpath = os.path.join(head, test_name)

            if os.path.isfile(testpath):
                file_list.append(testpath)

            elif not os.path.isfile(testpath):
                break

    num_files = len(file_list)

    im = np.array(Image.open(firstfile))
    xPixels = np.size(im, 0)
    yPixels = np.size(im, 1)

    imgstack = np.zeros(
        [num_files, xPixels, yPixels],
        dtype=np.uint16
    )

    print('Reading single-page tif files.')
    for i in tqdm(range(len(file_list))):
        f = file_list[i]
        imgstack[i,:,:] = np.array(Image.open(f), dtype=np.uint16)

    if saveas=='tif':
        print('Writing to multi-page tif file.')
        savefilename = os.path.join(head, base_filename+'.tif')
        with tifffile.TiffWriter(savefilename, bigtiff=True) as savestack:
            savestack.write(
                data=imgstack,
                dtype=np.uint16,
                shape=(num_files, xPixels, yPixels),
                photometric='MINISBLACK'
            )

    #TODO: test npy and mat save methods... haven't had to used these yet

    elif saveas=='npy':
        print('Writing to numpy array.')
        savefilename = os.path.join(head, base_filename+'.npy')
        np.save(savefilename, imgstack)

    elif saveas=='mat':
        print('Writing to Matlab file.')
        savefilename = os.path.join(head, base_filename+'.mat')
        savemat(
            savefilename,
            mdict={'TifStack': imgstack}
        )

    print('New file written to {}'.format(savefilename))

    if delete_singles is True:
        print('Deleting single-page tif files.')
        for f in file_list:
            os.remove(f)

    if ret is True:
        return savefilename, imgstack
    
    elif ret is False:
        return savefilename


def from_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str, default=None)
    parser.add_argument('-s', '--savepath', type=str, default=None)
    parser.add_argument('-as', '--saveas', default='tif')
    parser.add_argument('-del', '--delete', default=True)
    args = parser.parse_args()

    # ret flag will always be false when called from terminal, which
    # is the only access point to this func

    if args.filepath is None:
        filepath = select_file(
            'Select first tif file.',
            filetypes=[('TIF', '*.tif'),('TIF','*.tiff'),]
        )
    else:
        filepath = args.filepath

    tif_convert(filepath, args.savepath,
                delete_singles=args.delete,
                ret=False,
                saveas=args.saveas)


if __name__ == '__main__':

    from_args()