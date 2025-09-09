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
# cfg.data.test.ann_file=''