__MY_ROOT__=`pwd`
echo [$(date)]: "Started executing: "$BASH_SOURCE >> $__MY_ROOT__/test.log.txt
echo [$(date)]: "START" >> $__MY_ROOT__/test.log.txt
echo [$(date)]: "add test.log.txt to gitignore" >> $__MY_ROOT__/test.log.txt
echo "test.log.txt" >> .gitignore
echo [$(date)]: "Make sure you have installed protobuff" >> $__MY_ROOT__/test.log.txt
echo [$(date)]: "Create TensorFlow dir and cd to it" >> $__MY_ROOT__/test.log.txt
mkdir TensorFlow && cd TensorFlow
echo [$(date)]: "clone TensorFlow/models repo" >> $__MY_ROOT__/test.log.txt
git clone https://github.com/tensorflow/models.git
echo [$(date)]: "remove .git of models toi avoid conflict" >> $__MY_ROOT__/test.log.txt
rm -rf models/.git
echo [$(date)]: "add TensorFlow dir to .gitignore" >> $__MY_ROOT__/test.log.txt
echo "TensorFlow/models" >> ../.gitignore
echo [$(date)]: "cd to research dir" >> $__MY_ROOT__/test.log.txt
cd ./models/research
echo [$(date)]: "convert protos to protobuff" >> $__MY_ROOT__/test.log.txt
protoc object_detection/protos/*.proto --python_out=.
echo [$(date)]: "copy setup to research dir" >> $__MY_ROOT__/test.log.txt
cp object_detection/packages/tf2/setup.py .
echo [$(date)]: "upgrade pip" >> $__MY_ROOT__/test.log.txt
pip install --upgrade pip --user
echo [$(date)]: "install object_detection api" >> $__MY_ROOT__/test.log.txt
python -m pip install .
echo [$(date)]: "Testing our installation" >> $__MY_ROOT__/test.log.txt
python object_detection/builders/model_builder_tf2_test.py
echo [$(date)]: "Change to ROOT" >> $__MY_ROOT__/test.log.txt
cd $__MY_ROOT__
# -----------------------------------
echo [$(date)]: "START workspace" >> $__MY_ROOT__/test.log.txt
echo [$(date)]: "create workspace training demo" >> $__MY_ROOT__/test.log.txt
mkdir -p workspace/training_demo
echo [$(date)]: "cd to workspace/training_demo" >> $__MY_ROOT__/test.log.txt
cd workspace/training_demo
echo [$(date)]: "create workspace" >> $__MY_ROOT__/test.log.txt
mkdir -p annotations exported-models models/my_ssd_resnet50_v1_fpn pre-trained-models images
echo [$(date)]: "create label map in annotations dir with classes" >> $__MY_ROOT__/test.log.txt
echo "item {
    id: 1
    name: 'helmet'
}
item {
    id: 2
    name: 'head'
}
item {
    id: 3
    name: 'person'
}" > annotations/label_map.pbtxt
echo [$(date)]: "curl tfrecord converter in the root of workspace/training_demo" >> $__MY_ROOT__/test.log.txt
curl https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py > generate_tfrecord.py
echo [$(date)]: "use labelImg to do the annotations" >> $__MY_ROOT__/test.log.txt
echo [$(date)]: "Change to ROOT" >> $__MY_ROOT__/test.log.txt
cd $__MY_ROOT__
# --------------------------------

echo [$(date)]: "START" >> $__MY_ROOT__/test.log.txt
echo [$(date)]: "unzip dataset" >> $__MY_ROOT__/test.log.txt
unzip 'Hard Hat Workers.v2-raw.voc.zip'
echo [$(date)]: "remove unwanted txt files" >> $__MY_ROOT__/test.log.txt
rm README.dataset.txt README.roboflow.txt
echo [$(date)]: "mv train dir to images" >> $__MY_ROOT__/test.log.txt
mv train workspace/training_demo/images
echo [$(date)]: "mv test dir to images" >> $__MY_ROOT__/test.log.txt
mv "test" workspace/training_demo/images
echo [$(date)]: "add train to .gitignore" >> $__MY_ROOT__/test.log.txt
echo "workspace/training_demo/images/train/*" >> .gitignore
echo [$(date)]: "add test to .gitignore" >> $__MY_ROOT__/test.log.txt
echo "workspace/training_demo/images/test/*" >> .gitignore
echo [$(date)]: "cd to training_demo dir" >> $__MY_ROOT__/test.log.txt
cd workspace/training_demo
echo [$(date)]: "generate train tfrecord" >> $__MY_ROOT__/test.log.txt
python generate_tfrecord.py -x images/train -l annotations/label_map.pbtxt -o annotations/train.record
echo [$(date)]: "generate test tfrecord" >> $__MY_ROOT__/test.log.txt
python generate_tfrecord.py -x images/test -l annotations/label_map.pbtxt -o annotations/test.record
echo [$(date)]: "add misc files to .gitignore" >> $__MY_ROOT__/test.log.txt
echo "*.record" >> ../../.gitignore
echo "*.checkpoint" >> ../../.gitignore
echo "*.pb" >> ../../.gitignore
echo "variable*" >> ../../.gitignore
echo [$(date)]: "cd to pre-trained-models" >> $__MY_ROOT__/test.log.txt
cd pre-trained-models/
echo [$(date)]: "curl ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 model" >> $__MY_ROOT__/test.log.txt
curl http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz > ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz 
echo [$(date)]: "add .gz file to .gitignore" >> $__MY_ROOT__/test.log.txt
echo "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz" >> ../../../.gitignore
echo [$(date)]: "untar gz file" >> $__MY_ROOT__/test.log.txt
tar -xzf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
echo [$(date)]: "Change to ROOT" >> $__MY_ROOT__/test.log.txt
cd $__MY_ROOT__

echo [$(date)]: "Change to training_demo" >> $__MY_ROOT__/test.log.txt
cd workspace/training_demo
echo [$(date)]: "Create models/my_ssd_resnet50_v1_fpn" >> $__MY_ROOT__/test.log.txt
mkdir -p models/my_ssd_resnet50_v1_fpn
echo [$(date)]: "echo config file" >> $__MY_ROOT__/test.log.txt
echo 'model {
  ssd {
    num_classes: 3
    image_resizer {
      fixed_shape_resizer {
        height: 640
        width: 640
      }
    }
    feature_extractor {
      type: "ssd_resnet50_v1_fpn_keras"
      depth_multiplier: 1.0
      min_depth: 16
      conv_hyperparams {
        regularizer {
          l2_regularizer {
            weight: 0.00039999998989515007
          }
        }
        initializer {
          truncated_normal_initializer {
            mean: 0.0
            stddev: 0.029999999329447746
          }
        }
        activation: RELU_6
        batch_norm {
          decay: 0.996999979019165
          scale: true
          epsilon: 0.0010000000474974513
        }
      }
      override_base_feature_extractor_hyperparams: true
      fpn {
        min_level: 3
        max_level: 7
      }
    }
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
    matcher {
      argmax_matcher {
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
        use_matmul_gather: true
      }
    }
    similarity_calculator {
      iou_similarity {
      }
    }
    box_predictor {
      weight_shared_convolutional_box_predictor {
        conv_hyperparams {
          regularizer {
            l2_regularizer {
              weight: 0.00039999998989515007
            }
          }
          initializer {
            random_normal_initializer {
              mean: 0.0
              stddev: 0.009999999776482582
            }
          }
          activation: RELU_6
          batch_norm {
            decay: 0.996999979019165
            scale: true
            epsilon: 0.0010000000474974513
          }
        }
        depth: 256
        num_layers_before_predictor: 4
        kernel_size: 3
        class_prediction_bias_init: -4.599999904632568
      }
    }
    anchor_generator {
      multiscale_anchor_generator {
        min_level: 3
        max_level: 7
        anchor_scale: 4.0
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        scales_per_octave: 2
      }
    }
    post_processing {
      batch_non_max_suppression {
        score_threshold: 9.99999993922529e-09
        iou_threshold: 0.6000000238418579
        max_detections_per_class: 100
        max_total_detections: 100
        use_static_shapes: false
      }
      score_converter: SIGMOID
    }
    normalize_loss_by_num_matches: true
    loss {
      localization_loss {
        weighted_smooth_l1 {
        }
      }
      classification_loss {
        weighted_sigmoid_focal {
          gamma: 2.0
          alpha: 0.25
        }
      }
      classification_weight: 1.0
      localization_weight: 1.0
    }
    encode_background_as_zeros: true
    normalize_loc_loss_by_codesize: true
    inplace_batchnorm_update: true
    freeze_batchnorm: false
  }
}
train_config {
  batch_size: 4
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  sync_replicas: true
  optimizer {
    momentum_optimizer {
      learning_rate {
        cosine_decay_learning_rate {
          learning_rate_base: 0.03999999910593033
          total_steps: 25000
          warmup_learning_rate: 0.013333000242710114
          warmup_steps: 2000
        }
      }
      momentum_optimizer_value: 0.8999999761581421
    }
    use_moving_average: false
  }
  fine_tune_checkpoint: "pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0"
  num_steps: 25000
  startup_delay_steps: 0.0
  replicas_to_aggregate: 8
  max_number_of_boxes: 100
  unpad_groundtruth_tensors: false
  fine_tune_checkpoint_type: "detection"
  use_bfloat16: false
  fine_tune_checkpoint_version: V2
}
train_input_reader {
  label_map_path: "annotations/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "annotations/train.record"
  }
}
eval_config {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
}
eval_input_reader {
  label_map_path: "annotations/label_map.pbtxt"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "annotations/test.record"
  }
}' > models/my_ssd_resnet50_v1_fpn/pipeline.config
echo [$(date)]: "cp training file to training_demo dir" >> $__MY_ROOT__/test.log.txt
cp ../../TensorFlow/models/research/object_detection/model_main_tf2.py .
echo [$(date)]: "Start Training.." >> $__MY_ROOT__/test.log.txt
python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config
echo [$(date)]: "copy model exporter to training_demo dir" >> $__MY_ROOT__/test.log.txt
cp ../../TensorFlow/models/research/object_detection/exporter_main_v2.py .
echo [$(date)]: "Execution complete for: "$BASH_SOURCE >> $__MY_ROOT__/test.log.txt
echo [$(date)]: ">>>>>>>>> END <<<<<<<<<" >> $__MY_ROOT__/test.log.txt
