# from mmdet.datasets import build_dataset - only worked in older versions
# from mmdet.models import build_detector - only worked in older versions
#  in the newer versions of mmdetection build dataset and build detector is done inside the Runner
from mmengine.runner import Runner 
# from mmdet.apis import train_detector - only worked in older versions

import os
import mmcv

from cfg import cfg
# # Build dataset
# datasets = [build_dataset(cfg.train.data)]


# # Build detector
# model = build_detector(cfg.model)

runner = Runner.from_cfg(cfg=cfg)

# Add an attribute for visualization convenience
# model.CLASSES = datasets[0].CLASSES

# Create work_dir
# mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
if not os.path.exists(cfg.work_dir):
    os.makedirs(cfg.work_dir,exist_ok=True)
# train_detector(model,datasets,cfg,distributed=False,validate=True)
runner.train()
