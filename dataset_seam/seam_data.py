"""for SEAM dataset"""
import copy
import random
import numpy as np

from torch.utils.data import Dataset

def load_raw_volume(name, nx, ny, nz):
    """ load image into memory """

    # numpy.ptp(a, axis=None, out=None)
    #     Range of values (maximum - minimum) along an axis

    # # Normalised [0,1]
    # b = (a - np.min(a))/np.ptp(a)

    # # Normalised [0,255] as integer
    # c = (255*(a - np.min(a))/np.ptp(a)).astype(int)

    # # Normalised [-1,1]
    # d = 2*(a - np.min(a))/np.ptp(a)-1

    # # standardization
    # e = (a - np.mean(a)) / np.std(a)

    # load raw volume into memory
    img = np.fromfile(name, dtype=np.float32)

    # Normalized it to [0, 255]
    img = 255*(img - np.min(img))/np.ptp(img)

    # order from (y, x, z) to (z, y, x)
    img = img.astype(int).reshape(ny, nx, nz).transpose(2, 0, 1)

    return img

def generate_tile_indices(nx, ny, nz, tile_size, tile_skip):
    """ generate indices for tiles """

    indices = []
    for iz in range(0, nz-tile_size, tile_skip):
        for iy in range(0, ny-tile_size, tile_skip):
            for ix in range(0, nx-tile_size, tile_skip):
                indices.append((iz, iy, ix))

    # shuffle indices
    random.shuffle(indices)

    return indices

def generate_tile(vol, tile_size, tuple_zyx):
    """ get a tile from given volume """
    iz, iy, ix = tuple_zyx

    return vol[iz, iy:iy+tile_size, ix:ix+tile_size]

class SEAM(Dataset):
    """ SEAM image dataset """

    def __init__(self, img_name, vol_name=None, ny=1001, nx=876, nz=751, tile_size=101, tile_skip=10, transform=None, mask_threshold=0):
        super().__init__()

        self.img_name = img_name
        self.vol_name = vol_name

        self.ny = ny
        self.nx = nx
        self.nz = nz
        #
        self.tile_size = tile_size
        self.tile_skip = tile_skip

        self.transform = transform
        self.mask_threshold = mask_threshold

        # load raw image into memory
        self.img = load_raw_volume(self.img_name, self.nx, self.ny, self.nz)

        # mask volume
        self.mask = None
        if self.vol_name is not None:
            self.mask = load_raw_volume(self.vol_name, self.nx, self.ny, self.nz)

        # indices for loading
        self.indices = generate_tile_indices(self.nx, self.ny, self.nz, self.tile_size, self.tile_skip)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        # convert an index to a tuple
        tuple_zyx = self.indices[index]

        tile = {}
        # build the tile
        tile['image_id'] = '_'.join(map(str, tuple_zyx))
        tile['input'] = generate_tile(self.img, self.tile_size, tuple_zyx)
        if self.mask is not None:
            mask_tile = generate_tile(self.mask, self.tile_size, tuple_zyx)
            mask_tile[mask_tile > 0] = 1
            tile['mask'] = mask_tile
            pixel_count = mask_tile.sum()
            if 0 < pixel_count <= self.mask_threshold:
                return None

        tile = copy.copy(tile)
        if self.transform:
            tile = self.transform(tile)

        return tile

if __name__ == '__main__':

    train_dataset = SEAM(mode='train')
    print(len(train_dataset))
    assert len(train_dataset) == 399  # there is a bug, we ignore the first element from the csv file because of pandas

    test_dataset = SEAM(mode='test')
    assert len(test_dataset) == 18000
