# 3DR2N2
![Overview](overview.png)

The following project performed as a part of "Machine Learning in 3D" offered by Technical University of Munich. It implements [3D-R2N2](http://3d-r2n2.stanford.edu/) model in Pytorch with [Weights & Biases](https://wandb.ai) integration.

All the model weights and related artifacts can be found in our [project page](https://wandb.ai/ml43d-project/3dr2n2).

# Project Members
1. Cervera, Pascual Tejero
2. Kalkan, Doğukan
3. Korkmaz, Baran Deniz
4. Zverev, Daniil

# Data preparation
All the datasets necessary to run training and validation processes stored in our project [3D-R2N2 (Weights & Biases)](https://wandb.ai/ml43d-project/3dr2n2). All artifacts are publicly available, you can find them in artifact menu. The framework itself can be easily installed by this command:  
```shell 
pip install -qqq wandb
```

Currently, we have following datasets:
- 3D-Future (Entire dataset)
- ShapeNet (Subsample of 1k objects)

In order to download them, run this command in your environment:
```shell
wandb artifact get ml43d-project/3dr2n2/shapenet:v1 --type dataset
wandb artifact get ml43d-project/3dr2n2/3D-FUTURE-model-Sample:v2 --type model
```

### Visualisations and Evaluations
For your convenience we prepared colab notebook where you can investigate how the dataset looks like, as well as download pretrained model weights for quantitative and qualitative analysis.  

<a target="_blank" href="https://colab.research.google.com/drive/1XCO8e1KLF6YLCQf4aTOEU7u2m8iPalgp?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Training
Suppose that you unziped downloaded dataset to local folder "./data" then to start train pipeline, you should run following command:
```shell
python train.py \
--train_split ./data/3D-FUTURE-model-Sample/splits/train.txt \
--val_split ./data/3D-FUTURE-model-Sample/splits/val.txt \
--path_to_dataset ./data/3D-FUTURE-model-Sample \
--logger_type wandb \
--num_renders 20 \
--batch_size 8 \
--learning_rate 0.0001 \
--random_renders True \
--validate_every_n 20
```

In order to get full description of command you can run "--help" command:
```shell
python train.py --help
```

### Change configuration
Model and training configurations can be found in following file `src/configuration/train.ini`. You can either locally modify this file or simply override the configuration by passing necessary parameters as arguments to the training script. Here is an example of overriding:
```shell
python train.py \
--train_split ./data/3D-FUTURE-model-Sample/splits/train.txt \
--val_split ./data/3D-FUTURE-model-Sample/splits/val.txt \
--path_to_dataset ./data/3D-FUTURE-model-Sample \
--logger_type wandb \
--max_epochs 1000 \
--num_renders 20 \
--batch_size 8 \
--num_workers 2 \
--conv_rnn3d_type lstm \
--encoder_decoder_type residual \
--conv_rnn3d_kernel_size 3 \
--learning_rate 0.0001 \
--random_renders True \
--accumulate_grad_batches 8 \
--validate_every_n 20
```


Once again, for your convenience, we prepared colab notebook, that install all necessary dependencies and runs training pipeline easily:  

<a target="_blank" href="https://colab.research.google.com/drive/1behRYwHKO37E6e2R9TPHtHWLEHbotTah?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Project structure 
The structure of project looks like this:
```
- src 
    - callbacks
        # Pytorch-lightning callbacks for evalutation logging, checkpoints 
        # and random view numbers for each batch
        
    - configuration
        # Configurating of training pipeline
        
    - data
        # Dataset modeules and dataloaders
          
    - model
        # Model inplementation 
        
    - scripts
        # Scripts that transform ShapeNet and 3d-Future to
        # our propreitary dataset format 
        
    - utils
```