python3 tools/model_evaluation/eval_with_json_labels.py \
    task_settings/img_det/det_yolov4_retail_one.yaml \
    meta/train_infos/det_yolov4_retail_one/epoch_200.pth \
    --json-path data/test/a_det_annotations.json \
    --out-dir meta/test_a/
