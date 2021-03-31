from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.coco_split import coco_split
from datasets.imagenet import imagenet
from datasets.vg import vg
from datasets.episode import episode
from datasets.ycb2d import ycb2d

for i in [256, 240, 224, 208, 200, 192, 160, 128, 100, 96, 80, 64, 50, 48, 32, 30, 20, 16, 10]:
  name = f'ycb2d_replace{i}'
  __sets[name] = (lambda split='replace', year=str(i): ycb2d(split, year))

name = 'ycb2d_inference_sparse'
__sets[name] = (lambda split='inference', year='sparse': ycb2d(split, year))
name = 'ycb2d_inferencefs_sparse'
__sets[name] = (lambda split='inferencefs', year='sparse': ycb2d(split, year))
name = 'ycb2d_inference_dense'
__sets[name] = (lambda split='inference', year='dense': ycb2d(split, year))
name = 'ycb2d_inferencefs_dense'
__sets[name] = (lambda split='inferencefs', year='dense': ycb2d(split, year))
name = 'ycb2d_inference'
__sets[name] = (lambda split='inference', year='1234': ycb2d(split, year))

for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, '1cls', '2cls', '3cls', '4cls']:
  name = f'ycb2d_stage{i}'
  __sets[name] = (lambda split='stage', year=str(i): ycb2d(split, year))
for i in [512, 256, 128, 64, 32, 16, 8]:
  name = f'ycb2d_oracle{i}'
  __sets[name] = (lambda split='oracle', year=str(i): ycb2d(split, year))
for i in [64, 32, 16]:
  name = f'ycb2d_oracle_dense{i}'
  __sets[name] = (lambda split='oracledense', year=str(i): ycb2d(split, year))
for i in [20, 10, 5]:
  name = f'ycb2d_fsoracle_dense{i}'
  __sets[name] = (lambda split='fsoracledense', year=str(i): ycb2d(split, year))

name = 'ycb2d_pseudo'
for i in range(1, 10):
  __sets[name+str(i)] = (lambda split='pseudo', year=str(i): ycb2d(split, year))

__sets['coco_ft'] = (lambda split='shot', year='10': coco_split(split, year))

# coco 20 evaluation
for year in ['set1', 'set2']:
  for split in ['3way', '5way']:
    name = 'coco_{}_{}'.format(split, year)
    __sets[name] = (lambda split=split, year=year: coco_split(split, year))

# vis
for year in ['set1', 'set2', 'set3', 'set4']:
  for split in ['vis']:
    name = 'coco_{}_{}'.format(split, year)
    __sets[name] = (lambda split=split, year=year: coco_split(split, year))

# coco 20 evaluation
for year in ['set1', 'set2', 'set3', 'set4']:
  for split in ['20']:
    name = 'coco_{}_{}'.format(split, year)
    __sets[name] = (lambda split=split, year=year: coco_split(split, year))

# coco 60 training
for year in ['set1', 'set2', 'set3', 'set4', 'set1allcat']:
  for split in ['60']:
    name = 'coco_{}_{}'.format(split, year)
    __sets[name] = (lambda split=split, year=year: coco_split(split, year))

# episode
for year in ['novel', 'base', 'val']:
  for n in range(600): 
    split = 'ep' + str(n)
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: episode(split, year))


# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
