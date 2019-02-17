"""for SEAM dataset"""
import copy
import random
import numpy as np

import cv2

from torch.utils.data import Dataset

def load_volume(name, nx, ny, nz):
    """ load a volume into memory and from yxz to yzx order """

    # load raw volume into memory
    img = np.fromfile(name, dtype=np.float32)
    img = np.reshape(img, (ny, nx, nz))

    return img.transpose(0, 2, 1)

class TileBase:
    """ base class for tile manipulation """
    def __init__(self, tile_size, tile_skip):
        #
        self.tile_size = tile_size
        self.tile_skip = tile_skip

        # indices
        self.indices = None
        self.num_items = None

    def generate_tile_image_xz(self, vol, index):
        """ extract a xz image tile from a given 3D volume 
        """

        # the tile
        iy, ix, iz = self.index_to_tuple(index)
        img = vol[iy, iz:iz+self.tile_size, ix:ix+self.tile_size]

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

        # Normalized it to [0, 255]
        min_x = np.min(img)
        max_x = np.max(img)
        img = 255*(img - min_x)/(max_x - min_x)

        # reshape it to nsigned char
        img = img.astype(np.uint8)

        # make it a rgb
        img = np.stack([img]*3, axis=-1)

        return img

    def generate_tile_mask_xz(self, vol, index):
        """ extract a tile mask. Note a new channel dimension is added
        """

        # the tile
        iy, ix, iz = self.index_to_tuple(index)
        mask = vol[iy, iz:iz+self.tile_size, ix:ix+self.tile_size]

        SALT_VEL = 4480
        SALT_TOL = 10

        mask = mask.astype(np.uint32)

        # find salt
        mask[np.abs(mask-SALT_VEL) < SALT_TOL] = 1
        mask[mask != 1] = 0

        # make it a byte
        mask = mask.astype(np.uint8)

        return mask[..., np.newaxis]

    def index_to_tuple(self, index):
        """ convert 1D index to 3D tuple """
        if index < 0:
            index = self.num_items + index
        assert index >= 0 and index < self.num_items

        return self.indices[index]

    def generate_tile_tuples(self, nx, ny, nz):
        """ generate tile tuples """

        return NotImplementedError("Need to implement in subclass")

class Tile_yxz(TileBase):
    """ manipulate tiles """
    def __init__(self, tile_size=101, tile_skip=10):
        super().__init__(tile_size, tile_skip)

    def generate_tile_tuples(self, nx, ny, nz):
        """ generate tile tuples """

        tuples = []
        for iy in range(0, ny-self.tile_size, self.tile_skip):
            for ix in range(0, nx-self.tile_size, self.tile_skip):
                for iz in range(0, nz-self.tile_size, self.tile_skip):
                    tuples.append((iy, ix, iz))

        self.indices = tuples
        self.num_items = len(tuples)

class SEAM(Dataset):
    """ SEAM image dataset """

    def __init__(self, img_name, vol_name=None, ny=1001, nx=876, nz=751,
                        tile=Tile_yxz(tile_size=101, tile_skip=10),
                        transform=None, mask_threshold=0):
        super().__init__()

        self.img_name = img_name
        self.vol_name = vol_name

        self.ny = ny
        self.nx = nx
        self.nz = nz

        self.transform = transform
        self.mask_threshold = mask_threshold

        # for manipulating tiles
        self.tile = tile
        # need to generate tuples
        self.tile.generate_tile_tuples(self.nx, self.ny, self.nz)

        # load raw image into memory
        self.img = load_volume(self.img_name, self.nx, self.ny, self.nz)

        # mask volume
        self.mask = None
        if self.vol_name is not None:
            self.mask = load_volume(self.vol_name, self.nx, self.ny, self.nz)

    def __len__(self):
        return self.tile.num_items

    def __getitem__(self, index):

        # convert an index to a tuple
        tuple_yxz = self.tile.indices[index]

        tile = {}
        # build the image tile
        tile['image_id'] = '_'.join(map(str, tuple_yxz))

        # extract the tile
        tile['input'] = self.tile.generate_tile_image_xz(self.img, index)

        if self.mask is not None:
            # build mask tile
            mask_tile = self.tile.generate_tile_mask_xz(self.mask, index)
            tile['mask'] = mask_tile

            # pixel_count = mask_tile.sum()
            # if 0 < pixel_count <= self.mask_threshold:
            #     return None

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
