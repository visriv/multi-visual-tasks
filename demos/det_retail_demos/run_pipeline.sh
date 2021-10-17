#! /usr/bin/env bash
opt=$1

score_thr=0.1
det_epoch=200
det_name=det_yolov4_9a_retail_one

det_cfg=task_settings/img_det/${det_name}.yaml
det_model=meta/train_infos/${det_name}/epoch_${det_epoch}.pth
det_json=data/test/a_det_annotations.json

emb_epoch=50
emb_name=emb_resnet50_mlp_loc_retail

#emb_epoch=36
#emb_name=emb_resnet50_mlp_retail

tag=${det_name}_${det_epoch}_${emb_name}_${emb_epoch}

emb_cfg=task_settings/img_emb/${emb_name}.yaml
emb_model=meta/${emb_name}/epoch_${emb_epoch}.pth
emb_ref=meta/emb_test_b_${tag}.pkl

submit_json=submit/submit_${tag}.json

function step_1 {
    echo -e "Step 1: Running detetor ..."
    echo -e "\t cfg: $det_cfg"
    echo -e "\t model: $det_model"
    echo -e "\t save path: $det_json"

    python3 tools/model_evaluation/eval_with_json_labels.py \
        ${det_cfg} ${det_model} \
        --json-path ${det_json} \
        --out-dir meta/test_a/
}

# Check validation set in $emb_cfg before running following script
# VAL_DATA:
#   DATA_INFO: [["test/b_annotations.json"]]
#   DATA_PREFIX: ["test/b_images/"]
function step_2 {
    echo -e "Step 2: Saving reference embeddings ..."
    echo -e "\t cfg: $emb_cfg"
    echo -e "\t model: $emb_model"
    echo -e "\t save path: $emb_ref"

    python3 tools/model_evaluation/save_embeddings.py \
        ${emb_cfg} ${emb_model} \
        --save-path ${emb_ref}
}


# Check test set in $emb_cfg before running following script
# TEST_DATA:
#   DATA_INFO: ["test/a_det_annotations.json"]
#   DATA_PREFIX: ["test/a_images/"]
function step_3 {
    echo -e "Step 3: Saving result json for submition ..."
    echo -e "\tcfg: $emb_cfg"
    echo -e "\tmodel: $emb_model"

    python3 tools/model_evaluation/pred_embedding_with_json_label.py \
        ${emb_cfg} ${emb_model} ${emb_ref} \
        --json-ori ${det_json} \
        --json-out ${submit_json} \
        --score-thr ${score_thr}
}

case $opt in
    0)
        step_1
        step_2
        step_3
        ;;
    1)
        step_1
        ;;
    2)
        step_2
        ;;
    3)
        step_3
        ;;
    *)
        echo -e "Please input an argument from [0, 1, 2, 3]"
        echo -e "\t1: run detetor"
        echo -e "\t2: save reference embeddings"
        echo -e "\t3: save result json for submition"
        echo -e "\t0: run steps 1,2,3"
        exit
        ;;
esac
