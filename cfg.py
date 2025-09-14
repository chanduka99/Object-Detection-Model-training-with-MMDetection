from mmengine.config import Config
# from mmdet.apis import set_random_seed

cfg = Config.fromfile('mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py')

# Modify the dataset type and path.
cfg.dataset_type='VOCDataset'
print(f"Default Config:\n {cfg.pretty_text}")


# Modify dataset type and path.
cfg.dataset_type = 'VOCDataset'
cfg.data_root ='data/'


cfg.data.test.type = 'VOCDataset'
cfg.data.test.data_root = 'data/VOCdevkit/'
cfg.data.test.ann_file='VOC2007/ImageSets/Main/trainval.txt'
cfg.data.train.img_prefix = 'VOC2007/'

cfg.data.train.type = 'VOCDataset'
cfg.data.train.data_root = 'data/VOCdevkit/'
cfg.data.train.ann_file = 'VOC2007/ImageSets/Main/trainval.txt'
cfg.data.train.img_prefix = 'VOC2007/'
#
# test config setup
cfg.data.val.type = 'VOCDataset'
cfg.data.val.data_root = 'data/VOCdevkit/'
cfg.data.val.ann_file = 'VOC2007/ImageSets/Main/test.txt'
cfg.data.val.img_prefix = 'VOC2007/'