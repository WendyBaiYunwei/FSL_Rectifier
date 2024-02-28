# How to Run FSL-Rectifier and Reproduce Results
This directory contains our major implementations for the work FSL-Rectifier.

## Data and Pretrained Model Preparation
* For the Animals dataset and train-test split txt files, please first download the ImageNet ILSVRC2012 training set, before running `python tools/extract_animalfaces.py datasets/ILSVRC/Data/CLS-LOC/train --output_folder your-data-path --coor_file animalface_coordinates.txt`. The txt files can be obtained from `https://drive.google.com/file/d/1UuJW5Xl5KYI11eIlV34Q6p9j7DRjNVjX/view?usp=sharing` (Google Drive Link).
* For the Traffics dataset, please download the dataset from `https://github.com/mibastro/VPE`, and download the train and test split txt files from the Google Drive Link.
* Replace `animals.yaml` and `traffic.yaml` data-paths in this directory, with the data-paths containing the downloads.

## Get Trained FSL models
* Please run `train_fsl.py` along with your preferred configuration parameters.
* More details can be found in `https://github.com/Sha-Lab/FEAT`.
* Alternatively, you can download the trained models from the Google Drive Link and store them in the current directory ('./').

## Get Image Translator
* Adjust hyperparameters in `animals.yaml` or `traffic.yaml` accordingly
* To train for Animals dataset, run `train_translator.py --config animals.yaml`
* To train for Traffics dataset, run `train_translator.py --config traffic.yaml`
* Alternatively, you can download the trained models from the Google Drive Link and store them in the current directory ('./').

## Get Neighbour Selector
* Adjust hyperparameters in `animals.yaml` or `traffic.yaml` accordingly
* To train for Animals dataset, run `train_picker.py --config animals.yaml`
* To train for Traffics dataset, run `train_picker.py --config traffic.yaml`
* Alternatively, you can download the trained models from the Google Drive Link and store them in the current directory ('./').

## Test FSL model W/O FSL-Rectifier
* Please run `python test_fsl.py --model_class ProtoNet --backbone_class ConvNet --dataset Animals --num_eval_episodes 1000 --model_path {checkpoint.pt path for protonet} --spt_expansion 1 --qry_expansion 0 --add_transform original` to test Conv4 Protonet on Animals dataset.
* You can find more examples in `auto_experiment.py`, which is also a tool for running experiments and saving records in txt files automatically.
* If you find the inference procedure too slow and plan to do multiple experiments, please speed up the testing by running `get_buffer_imgs.py`, which saves the augmented images in your data-paths, before running `test-fsl.py`.

## Get TSNE Projection
* Please run `get_embedding.py`, followed by `TSNE.py` to obtain the TSNE projections.
