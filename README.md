# LabelNoiseInSFDA

Official implementation for our paper **"Unraveling the Mysteries of Label Noise in Source-Free Domain Adaptation: Theory and Practice"**.

---

## Repository Structure

```shell

labelNoiseInSFDA/
├── common/                                 # Datasets
│   ├── vision/
│   │   └── datasets/
├── config/                                 # Configuration files (YAML examples)
├── DATASOURCE/                             # Data storage
│   ├── domainnet/
│   │   ├── image_list/
│   │   └── ...
│   ├── office/
│   ├── office-home/
│   └── VISDA-C/
├── Models/                                 # Source models and checkpoints
├── pointDA-10/                             # Adaptation code for PointDA-10 dataset
│   └── train_tar_merged.py
├── README.md                               # This file
├── train_source_imbalance.py               # Source training script
├── train_target_labelNoise_SFDA.py         # Main adaptation script
├── loggers.py                              # Logging utilities
├── utils_dataloader.py                     # Data loading utilities
├── utils_evaluation.py                     # Evaluation utilities
├── utils_lln_losses.py                     # Label noise loss implementations
├── utils_loss.py                           # General loss functions
├── utils_pseudo_label.py                   # Pseudo-labeling utilities
├── utils_source_train.py                   # Source training utilities
└── vis_sourcefree.py                       # Model architectures and networks 
```

---

## Getting Started

### 1. Data Preparation

#### Office-31 Dataset
Download the dataset [here](https://drive.google.com/file/d/1dmEhhzoP-dnVOqsV_zDUNRKUy4dKN6Bl/view?usp=sharing). Unzip the `office.zip` file into `./DATASOURCE/`. Ensure the `image_list` folder is located in `./DATASOURCE/office/`.

#### Office-Home Dataset
Download the dataset [here](https://drive.google.com/file/d/1gRSeLmBTKWjSiqe6jRWaNsXTqzDdqKPr/view?usp=sharing). Unzip the `office-home.zip` file into `./DATASOURCE/`. Ensure the `image_list_nrc` (for Office-Home) and `image_list_partial` (for partial set) are located in `./DATASOURCE/VISDA-C/`.

#### VisDA-2017 Dataset
Download the original dataset [here](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification). Extract `train.tar` and `validation.tar` into `./DATASOURCE/VISDA-C/`. Ensure `image_list` (for VisDA2017) and `image_list_imb` (for VisDA-RUST) are located in `./DATASOURCE/VISDA-C/`.

#### DomainNet Dataset
Download the dataset [here](https://ai.bu.edu/M3SDA/). Extract the image data of each `DOMAIN` into `./DATASOURCE/domainnet/DOMAIN`. Ensure the `image_list` folder is located in `./DATASOURCE/domainnet/`.

#### PointDA-10 Dataset
Download the dataset [here](https://github.com/canqin001/PointDAN/tree/master?tab=readme-ov-file) from the [PointDAN repository](https://github.com/canqin001/PointDAN). Unzip the data file into `./DATASOURCE/` and retain its original folder name (`PointDA_data`).

### Data Verification
Ensure the `DATASOURCE` folder and the associated `image_list*` folders are correctly downloaded and organized as specified.

---

### 2. Model Preparation

#### Pretrained Models
- For Office-Home and VisDA datasets, we use source models provided by repositories such as [SHOT](https://github.com/tim-learn/SHOT) and [NRC](https://github.com/Albert0147/NRC_SFDA).
- For Office-31 and DomainNet (40 categories classification), we trained source models ourselves following [SHOT](https://github.com/tim-learn/SHOT).
- For PointDA, source model training follows the instructions in [NRC](https://github.com/Albert0147/NRC_SFDA) and [PointDAN](https://github.com/canqin001/PointDAN).

#### Model Checkpoints
Download all model checkpoints [here](https://drive.google.com/drive/folders/1SfPPaTu69ef4TAuaNeW6KSd7bMlqyg6l?usp=sharing) and place them in the `./Models/` directory.

---

### 3. Adaptation Scripts

#### 3.1 Image Classification Tasks

You can configure adaptations using either command-line arguments or YAML configuration files. Below is an example for the Office-Home dataset.

##### Command-Line Example

- **SHOT + ELR (or other LLN losses):**
    ```shell
    python train_target_labelNoise_SFDA.py \
    --dset office-home --source Ar --target Rw --net resnet50 --net_mode fc --list_name image_list_nrc \
    --max_epoch 45 --interval 45 \
    --root ./DATASOURCE \
    --model_root ./Models/officehome \
    --output_dir ./output \
    --log_dir ./logs \
    --expname tpami --key_info officehome_test_shot_elr --seed 2021 \
    --lr_decay True --lr_decay_type shot --lr_F_coef 0.5 --weight_decay 0.0005 \
    --is_shot True \ #SHOT
    --is_lln True --lln_type elr --beta 0.6 --lamb 3.0 --lln_coef 0.3 # params for LLN losses, such as sl, gce, gjs
    ```

- **NVC-LLN:**

    ```shell
    python train_target_labelNoise_SFDA.py \
    --dset office-home --source Ar --target Rw --net resnet50 --net_mode fc --list_name image_list_nrc \
    --max_epoch 45 --interval 45 \
    --root ./DATASOURCE \
    --model_root ./Models/officehome \
    --output_dir ./output \
    --log_dir ./logs \
    --expname tpami --key_info officehome_test_nvc_gce --seed 2021 \
    --lr_decay True --lr_decay_type shot --lr_F_coef 0.5 --weight_decay 0.0005 \
    --is_lln True --lln_type gce --beta 0.2 --lln_coef 0.3 --lln_mask True \ # C-LLN losses, such as sl, gce, gjs, elr
    --is_ca True --K 3 --alpha 1 --alpha_beta 0.75 --alpha_decay False --smooth_ca 0.8 \ # smooth CA
    --is_data_aug True --data_aug_coef 0.1 --data_aug_temp 1.0 \ # data aug alignment
    ```

##### YAML Configuration Example

YAML configurations take precedence over command-line arguments.

```shell
python train_target_labelNoise_SFDA.py --config ./config/oh_nvcgce_example.yml
```

#### 3.2 3D Point Cloud Tasks

For the PointDA-10 dataset, the code is built upon [NRC](https://github.com/Albert0147/NRC_SFDA) and [PointDAN](https://github.com/canqin001/PointDAN). Parameters for ELR and NVC-LLN are the same as those in `train_target_labelNoise_SFDA.py`. Use the script `./pointDA-10/train_tar_merged.py` for adaptation.

<!-- ## Acknowledgement and citation -->

