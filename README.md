# COCO-Text Segmentation (Ongoing)

COCO-Text Segmentation with U-Net.

This project is a demonstration of the text segmentation in the wild with U-Net. The project is still ongoing but due to practical reasons, I cannot commit to continuous engagement. Still, comments and suggestions are very much welcome.

## Setup

1. Download the MS-COCO 2014 dataset from the [MSCOCO website](https://cocodataset.org/#download).

2. Clone the COCO-Text toolbox repository and add it to `$PYTHONPATH`:
```
cd $WORKSPACEPATH
git clone https://github.com/bgshih/coco-text
export PYTHONPATH=${PYTHONPATH}:${WORKSPACEPATH}/coco-text
```

3. Download the [COCO-Text annotation](https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip), and extract the annoatation file `cocotext.v2.json`.

3. Install python package dependencies:
```
pip install -r requirements.txt
```

## Prepare the dataset

A big challenge in text segmentation is a great number of very small text regions compared to the image size. For example, a 640x480 image, when downsampled to typical network input size 224x224, loses many eligible text, making them harder to detect.

To avoid this problem the U-Net is not trained directly on downsampled MS-COCO images, but on a dataset made of patches extracted from the images that contain at least 1 visible text area (although it might be the culprit of many false negatives). The patch size is at least 256x256, bigger if the text region is bigger. 

To generate the dataset for training set, run the following command:

```
python generate_dataset.py cocotext.v2.json path_to_coco_train_set path_to_output_dataset path_to_output_box_csv_file --train_or_val train
```

## Training

Run `train.py` to train the network:

```
python train.py /path_to_output_dataset/train_anno/ /path_to_output_dataset/train/ path_to_output_box_csv_file.csv output_model_dir
```

## Inference

Run `inference.py` to perform inference with the network.  
For example:

```
python inference.py test_image_path.jpg output_model_dir/epoch_xxx.pickle --arch_params output_model_dir/config.json --output out_img.jpg --blur
```

It supports a list of file input or a directory of images. Output is the image of same size with text region either blurred or colored green, controlled by the `--blur` flag. See `python inference.py --help` for help.

## TODO

The project is still ongoing. I intend to investigate the following directions:

1. Add hard negative mining

2. Try network architectures with detection backbone

3. Better blurring technique

## Sample inference Output
### Sample 1
Before:

![Image 1](assets/COCO_val2014_000000000400.jpg)

After:

![Image 1 Blurred](assets/COCO_val2014_000000000400_blurred_07.jpg)

### Sample 2
Before:

![Image 2](assets/COCO_train2014_000000003602.jpg)

After:

![Image 2 Blurred](assets/COCO_train2014_000000003602_blurred.jpg)
