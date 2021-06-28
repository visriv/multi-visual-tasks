model_name=emb_resnet50_mlp_arcmargin_all_retail
python3 ./tools/train.py --work-dir meta/train_infos --no-test \
    --load-from meta/train_infos/emb_resnet50_mlp_arcmargin_retail/epoch_100.pth \
    task_settings/img_emb/${model_name}.yaml
