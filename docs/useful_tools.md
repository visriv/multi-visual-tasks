# Useful Tools
  This page provides useful tools for data processing.

  <!-- TOC -->

  - [Detection visualization and evaluation](#detection-visualization-and-evaluation)
  - [VSCode debugging](#vscode-debugging)

  <!-- TOC -->


## Detection visualization and evaluation

python3 ./tools/visualization/test_vis_gt_pred.py task_settings/img_det/det_ssd_300_vgg_voc.yaml meta/train_infos/det_ssd_300_vgg_voc/epoch_200.pth --eval 'mAP' --out 'meta/test_infos/det_ssd_300_vgg_voc_eval.pkl' --show-dir 'meta/test_infos/det_ssd_300_vgg_voc_eval'


## VSCode debugging
in the debug console: export PYTHONPATH=$(pwd):$PYTHONPATH
