from archai.data import *

t1 = rasterio_loader('/Users/jsh2/SU4013.TIF')
print(t1.metadata)


ds = MultiSegmentationDataset('/Users/jsh2/Downloads/southdowns_01-03-2021/', ['southdowns_chips'],
                              ['southdowns_masks_trackway', 'southdowns_masks_pit', 'southdowns_masks_mound',
                              'southdowns_masks_ditch', 'southdowns_masks_bank'])

input, target = ds[0]
print(input.shape, target.shape)
