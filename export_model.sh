echo [$(date)]: "START"
echo [$(date)]: "cd to training_demo dir"
cd workspace/training_demo
echo [$(date)]: "Export model for prediction"
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/my_ssd_resnet50_v1_fpn/pipeline.config --trained_checkpoint_dir ./models/my_ssd_resnet50_v1_fpn/ --output_directory ./exported-models/my_model
echo [$(date)]: "END"