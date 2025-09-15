from mmengine.config import Config


cfg = Config.fromfile('mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py')

# Modify the dataset type and path.
cfg.dataset_type='VOCDataset'
print(f"Default Config:\n {cfg.pretty_text}")


# Modify dataset type and path.
cfg.dataset_type = 'VOCDataset'
cfg.data_root ='data/'

# cfg.data.test.type = 'VOCDataset'
cfg.test_dataloader.dataset.type = 'VOCDataset'
# cfg.data.test.data_root = 'data/VOCtest_06-Nov-2007/VOCdevkit/'
cfg.test_dataloader.dataset.data_root = 'data/VOCtest_06-Nov-2007/VOCdevkit/'
# cfg.data.test.ann_file='VOC2007/ImageSets/Main/test.txt'
cfg.test_dataloader.dataset.ann_file = 'VOC2007/ImageSets/Main/test.txt'
# cfg.data.train.img_prefix = 'VOC2007/'
cfg.test_dataloader.dataset.data_prefix.img = 'VOC2007/'

# cfg.data.train.type = 'VOCDataset'
cfg.train_dataloader.dataset.type = 'VOCDataset'
# cfg.data.train.data_root = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/'
cfg.train_dataloader.dataset.data_root = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/'
# cfg.data.train.ann_file = 'VOC2007/ImageSets/Main/trainval.txt'
cfg.train_dataloader.dataset.ann_file = 'VOC2007/ImageSets/Main/trainval.txt'
# cfg.data.train.img_prefix = 'VOC2007/'
cfg.train_dataloader.dataset.data_prefix.img = 'VOC2007/'

# cfg.data.val.type = 'VOCDataset'
cfg.val_dataloader.dataset.type = 'VOCDataset'
# cfg.data.val.data_root = 'data/VOCtest_06-Nov-2007/VOCdevkit/'
cfg.val_dataloader.dataset.data_root = 'data/VOCtest_06-Nov-2007/VOCdevkit/'
# cfg.data.val.ann_file = 'VOC2007/ImageSets/Main/test.txt'
cfg.val_dataloader.dataset.ann_file = 'VOC2007/ImageSets/Main/test.txt'
# cfg.data.val.img_prefix = 'VOC2007/'
cfg.val_dataloader.dataset.data_prefix.img = 'VOC2007/'

# Batch size (samples for GPU)
# cfg.data.samples_per_gpu = 2
cfg.train_dataloader.batch_size=2


# Modify number of classes as per the model head
cfg.model.roi_head.bbox_head.num_classes = 20

# Comment/Uncomment this to training from scratch/fine-tune according to the
# model checkpoint path.
cfg.load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# output directory for training. As per the model name
cfg.work_dir = 'outputs/faster_rcnn_r50_fpn_1x_coco_fine_tune'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
# cfg.optimizer.lf = 0.02/8
cfg.optim_wrapper.optimizer.lf = 0.02/8
# cfg.lr_config.warmup=None
# cfg.log_config.interval = 5
cfg.default_hooks.logger.interval = 5

# Evaluation Metric
# cfg.evaluation.metric = 'mAP'
# cfg.default_hooks.evaluation.metric = 'mAP'
# Evaluation times.
# cfg.evaluation.interval = 5
# cfg.default_hooks.evaluation.interval = 5
# Checkpoint storage interval.
# cfg.checkpoint_config.interval = 5
# cfg.default_hooks.checkpoint_config.interval = 5


# Set random seed for reproducible results.
# cfg.seed = 0
# cfg.randomness.seed = 0 - no randomness attribute
# set_random_seed(0,deterministic=False)
# cfg.determinsitic =False
# cfg.randomness.determinsitic =False- no randomness attribute

cfg.gpu_ids = range(1)
cfg.device = 'cuda'
# cfg.runner.max_epochs = 10
# cfg.train_cfg.by_epoch = True
cfg.train_cfg.max_epochs = 10

# We can also use tenserboard to log the training process
# cfg.log_config.hooks = [
#     dict(type='TextLoggerHook'),
#     dict(type='TensorboardLoggerHook')]
cfg.visualizer.type ='DetLocalVisualizer'
cfg.visualizer.vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ]

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')