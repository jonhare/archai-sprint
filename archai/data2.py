import collections
import re
from os import listdir
from os.path import isfile, join, exists

import rasterio
import torch
from torch.utils.data import Dataset


def rasterio_loader(path: str) -> torch.Tensor:
    with rasterio.open(path) as r:
        meta = {
            'bounds': r.bounds,
            'transform': tuple(r.transform),
            'crs': r.crs
        }
        x = torch.as_tensor(r.read()).float()

    return x, meta


class MultiSegmentationDataset(Dataset):
    def __init__(self, root, input_types, target_types, name_regex='_\\d+_\\d+', target_prefix='mask'):
        self.root = root
        self.input_types = input_types if isinstance(input_types, collections.Sequence) else tuple(input_types)
        self.target_types = target_types if isinstance(target_types, collections.Sequence) else tuple(target_types)

        searchdir = join(root, input_types[0])
        self.image_names = [f for f in listdir(searchdir) if isfile(join(searchdir, f)) and f.endswith(".tif")
                            and not f.startswith('.')]

        self.name_regex = name_regex
        self.target_prefix = target_prefix

        self._dims = {}
        for t in [*input_types, *target_types]:
            for i in self.image_names:
                fn = join(self.root, t, i)
                if not exists(fn):
                    fn = join(self.root, t, self._rename_target(i))
                if exists(fn):
                    self._dims[t] = rasterio_loader(fn)[0].shape
                    break

    def _rename_target(self, i):
        return self.target_prefix + re.findall(self.name_regex, i)[0] + '.tif'

    def _load(self, name, dirs):
        imgs = []
        metadata = {}
        for d in dirs:
            fn = join(self.root, d, name)
            if not exists(fn):
                fn = join(self.root, d, self._rename_target(name))
            if exists(fn):
                i, metadata = rasterio_loader(fn)
                imgs.append(i)
            else:
                imgs.append(torch.zeros(self._dims[d]))
        ret = torch.cat(imgs, dim=0)

        return ret, metadata

    def __getitem__(self, index):
        name = self.image_names[index]

        input = self._load(name, self.input_types)
        target = self._load(name, self.target_types)

        return input[0], *target  # assuming that all the meta is the same

    def __len__(self):
        return len(self.image_names)


def save_as_geotiff(t, meta, filename):
    npa = t.cpu().detach().as_numpy()
    if meta is not None:
        crs = meta['crs']
        transform = meta['transform']
    else:
        crs = None
        transform = None

    with rasterio.open(
            filename,
            'w',
            driver='GTiff',
            height=npa.shape[1],
            width=npa.shape[2],
            count=npa.shape[0],
            dtype=npa.dtype,
            crs=crs,
            transform=transform,
    ) as dst:
        dst.write(npa)
