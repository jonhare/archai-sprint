import collections
import re
from os import listdir
from os.path import isfile, join, exists

import rasterio
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import transforms


# Use this instead of the normal Compose to keep the metadata attached to the output
class Compose(transforms.Compose):
    def __call__(self, img):
        out = super().__call__(img)
        if hasattr(img, 'metadata'):
            out.metadata = img.metadata

        return out


def rasterio_loader(path: str) -> torch.Tensor:
    with rasterio.open(path) as r:
        meta = {
            'bounds': r.bounds,
            'transform': r.transform,
            'crs': r.crs.as_dict()
        }
        x = torch.as_tensor(r.read()).float()
        x.metadata = meta

    return x


def meta_collate(batch):
    """
    Custom collate function to use with a DataLoader to keep hold of any metadata in the tensors

    :param batch: list of tensors to batch
    :return: the batch
    """
    # print(batch[0][0].metadata)
    newbatch = default_collate(batch)

    for i, x in enumerate(newbatch):
        x.metadata = [batch[j][i].metadata if hasattr(batch[j][i], 'metadata') else {} for j in range(len(batch))]
    return newbatch


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
                    self._dims[t] = rasterio_loader(fn).shape
                    break

    def _rename_target(self, i):
        return self.target_prefix + re.findall(self.name_regex, i)[0] + '.tif'

    def _load(self, name, dirs):
        imgs = []
        for d in dirs:
            fn = join(self.root, d, name)
            if not exists(fn):
                fn = join(self.root, d, self._rename_target(name))
            if exists(fn):
                imgs.append(rasterio_loader(fn))
            else:
                imgs.append(torch.zeros(self._dims[d]))
        ret = torch.cat(imgs, dim=0)
        ret.metadata = imgs[0].metadata
        return ret

    def __getitem__(self, index):
        name = self.image_names[index]

        input = self._load(name, self.input_types)
        target = self._load(name, self.target_types)

        return input, target

    def __len__(self):
        return len(self.image_names)


def get_metadata(t):
    """
    Get metadata associated with a tensor (if any)
    :param t: the tensor
    :return: the metadata
    """
    if hasattr(t, 'metadata'):
        return t.metadata
    return None


def save_as_geotiff(t, filename):
    assert (t.ndim == 3)

    npa = t.cpu().detach().as_numpy()
    if hasattr(t, 'metadata'):
        crs = t.metadata['crs']
        transform = t.metadata['crs']
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
