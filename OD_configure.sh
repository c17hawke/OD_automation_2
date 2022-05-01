echo [$(date)]: "START"
__MY_ROOT__=`pwd`
echo $__MY_ROOT__
echo [$(date)]: "Make sure you have installed protobuff"
echo [$(date)]: "Create TensorFlow dir and cd to it"
mkdir TensorFlow && cd TensorFlow
echo [$(date)]: "clone TensorFlow/models repo"
git clone https://github.com/tensorflow/models.git
echo [$(date)]: "remove .git of models toi avoid conflict"
rm -rf models/.git
echo [$(date)]: "add TensorFlow dir to .gitignore"
echo "TensorFlow/models" >> ../.gitignore
echo [$(date)]: "cd to research dir"
cd ./models/research
echo [$(date)]: "convert protos to protobuff"
protoc object_detection/protos/*.proto --python_out=.
echo [$(date)]: "copy setup to research dir"
cp object_detection/packages/tf2/setup.py .
echo [$(date)]: "install object_detection api"
python -m pip install .
echo [$(date)]: "Testing our installation"
python object_detection/builders/model_builder_tf2_test.py
echo [$(date)]: "END"
cd $__MY_ROOT__
# -----------------------------------
echo [$(date)]: "START workspace"
echo [$(date)]: "create workspace training demo"
mkdir -p workspace/training_demo
echo [$(date)]: "cd to workspace/training_demo"
cd workspace/training_demo
echo [$(date)]: "create workspace"
mkdir -p annotations exported-models models/my_ssd_resnet50_v1_fpn pre-trained-models images/test images/train
echo [$(date)]: "create label map in annotations dir with classes"
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
echo [$(date)]: "curl tfrecord converter in the root of workspace/training_demo"
curl https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py > generate_tfrecord.py
echo [$(date)]: "use labelImg to do the annotations"
echo [$(date)]: "create configure_training shell file for next step"
echo [$(date)]: "END"
cd $__MY_ROOT__
# --------------------------------

echo [$(date)]: "START"
echo [$(date)]: "unzip dataset"
unzip 'Hard Hat Workers.v2-raw.voc.zip'
rm README.dataset.txt README.roboflow.txt
echo [$(date)]: "mv train dir to images"
mv train workspace/training_demo/images
echo [$(date)]: "mv test dir to images"
mv test workspace/training_demo/images
echo [$(date)]: "add train to .gitignore"
echo "workspace/training_demo/images/train/*" >> .gitignore
echo [$(date)]: "add test to .gitignore"
echo "workspace/training_demo/images/test/*" >> .gitignore
echo [$(date)]: "cd to training_demo dir"
cd workspace/training_demo
echo [$(date)]: "generate train tfrecord"
python generate_tfrecord.py -x images/train -l annotations/label_map.pbtxt -o annotations/train.record
echo [$(date)]: "generate test tfrecord"
python generate_tfrecord.py -x images/test -l annotations/label_map.pbtxt -o annotations/test.record
echo [$(date)]: "add misc files to .gitignore"
echo "*.record" > ../../.gitignore
echo "*.checkpoint" > ../../.gitignore
echo "*.pb" > ../../.gitignore
echo "variable*" > ../../.gitignore
echo [$(date)]: "cd to pre-trained-models"
cd pre-trained-models/
echo [$(date)]: "curl ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 model"
curl http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz > ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz 
echo [$(date)]: "add .gz file to .gitignore"
echo "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz" >> ../../../.gitignore
echo [$(date)]: "untar gz file"
tar -xzf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
echo [$(date)]: "END"
cd $__MY_ROOT__
echo [$(date)]: "END"
# echo [$(date)]: "START"
# echo [$(date)]: "START"
# echo [$(date)]: "END"
