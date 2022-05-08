# MaskNet
DL network for mask detection.

## The Algorithm

I first collected images on google consisting of 200 images of people with masks and 200 images of people without masks. I split this data set in an 80-10-10 ratio for training, testing, and validation respectively. After retraining resnet-18 with this dataset on the Jetson Nano Developer Kit, I exported the retrained model with onnx. However, I realized the model was innacurate after testing. I tried increasing the number of epochs in training, but the accuracy did not improve. I then collected a new dataset of pictures of myself, consisting of 50 images with a mask and 50 images without a mask. After retraining and exporting resnet-18 with my new dataset, the accuracy had drastically improved. To retrain resnet-18, I followed the steps described in the module "Re-Training Image Classification Models" from NVIDIA's Artifical Intelligence and Machine Learning course. 

## Running this project

1. I first copied the dataset to the Jetson data directory on the Jetson environment
2. To train the network, I used the following command: `python3 train.py --model-dir=models/maskdatav2 data/maskdatav2 --batch-size 4 --workers 1 --epochs 10 --lr 0.01 --momentum 0.1`
3. Next, I exported the retrained model to onnx format using the command `python3 onnx_export.py --model-dir=models/maskdatav2`
4. I used the following command to test the model with my webcam: `imagenet.py --model=models/maskdatav2/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=data/maskdatav2/labels.txt /dev/video0  display://0`
