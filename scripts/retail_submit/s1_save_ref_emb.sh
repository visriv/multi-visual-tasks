# check validation set in task config before running this script
python3 tools/model_evaluation/save_embeddings.py \
    meta/emb_resnet50_mlp_retail/emb_resnet50_mlp_retail.yaml \
    meta/emb_resnet50_mlp_retail/epoch_36.pth \
    --save-path meta/reference_test_b_embedding.pkl
