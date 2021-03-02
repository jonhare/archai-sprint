from archai.data import *
from torch.utils.data import DataLoader


# t1 = rasterio_loader('/Users/jsh2/SU4013.TIF')
# print(t1.metadata)
#
#
# ds = MultiSegmentationDataset('/Users/jsh2/Downloads/southdowns_01-03-2021/', ['southdowns_chips'],
#                               ['southdowns_masks_trackway', 'southdowns_masks_pit', 'southdowns_masks_mound',
#                               'southdowns_masks_ditch', 'southdowns_masks_bank'])
#
# input, target = ds[0]
# print(input.shape, target.shape)
#
#
# dl = DataLoader(ds, collate_fn=meta_collate, batch_size=2)
# batch = iter(dl).__next__()
# print(batch[0].metadata)



from archai.data2 import *


def main():
    t1 = rasterio_loader('/Users/jsh2/SU4013.TIF')
    print(t1)

    ds = MultiSegmentationDataset('/Users/jsh2/Downloads/southdowns_01-03-2021/', ['southdowns_chips'],
                                  ['southdowns_masks_trackway', 'southdowns_masks_pit', 'southdowns_masks_mound',
                                  'southdowns_masks_ditch', 'southdowns_masks_bank'])

    input, target, meta = ds[0]
    print(input)
    print(input.shape, target.shape, meta)

    dl = DataLoader(ds, batch_size=2, num_workers=0)
    batch = iter(dl).__next__()
    print("batch of input images: ", batch[0].shape)
    print("batch of target images: ", batch[1].shape)
    print("batch of metas: ", len(batch[2]))


if __name__ == '__main__':
    main()
