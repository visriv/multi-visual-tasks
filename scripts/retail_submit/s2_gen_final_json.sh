python3 tools/model_evaluation/pred_embedding_with_json_label.py \
    meta/emb_resnet50_mlp_retail/emb_resnet50_mlp_retail.yaml \
    meta/emb_resnet50_mlp_retail/epoch_36.pth \
    meta/reference_test_b_embedding.pkl \
    --json-ori data/test/a_det_annotations.json \
    --json-out submit/out.json
