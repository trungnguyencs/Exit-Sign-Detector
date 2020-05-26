python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ../pre-trained-model/ssd_inception_v2_coco.config \
    --trained_checkpoint_prefix ./ckpt/model.ckpt-5715 \
    --output_directory ./inference_graph
