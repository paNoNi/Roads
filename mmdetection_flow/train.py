from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel

cfg = Config.fromfile('config.py')

from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
import mmcv
import os
os.environ['WANDB_NOTEBOOK_NAME'] = 'flow.ipynb'
# Build the dataset

if __name__ == '__main__':
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_segmentor(cfg.model)
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    model = MMDistributedDataParallel(model)
    # Create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())
