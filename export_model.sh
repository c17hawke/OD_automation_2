__MY_ROOT__=`pwd`
echo [$(date)]: "Started executing: "$BASH_SOURCE >> $__MY_ROOT__/test.log.txt
echo [$(date)]: "START" >> $__MY_ROOT__/test.log.txt
echo [$(date)]: "cd to training_demo dir" >> $__MY_ROOT__/test.log.txt
cd workspace/training_demo
echo [$(date)]: "Export model for prediction" >> $__MY_ROOT__/test.log.txt
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/my_ssd_resnet50_v1_fpn/pipeline.config --trained_checkpoint_dir ./models/my_ssd_resnet50_v1_fpn/ --output_directory ./exported-models/my_model
echo [$(date)]: "Execution complete for: "$BASH_SOURCE >> $__MY_ROOT__/test.log.txt
echo [$(date)]: ">>>>>>>>> END <<<<<<<<<" >> $__MY_ROOT__/test.log.txt