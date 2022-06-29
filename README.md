# ML43D - 3DR2N2

# Project Members
In alphabetical order by surname:
1. Cervera, Pascual Tejero
2. Kalkan, Doğukan
3. Korkmaz, Baran Deniz
4. Zverev, Daniil

## Data preparation
All the datasets necessary to run training and validation processes stored in [Weights & Biases](https://wandb.ai/ml43d-project/3dr2n2). You can find them in artifact menu. The framework itself can be easily installed by this command:  
```shell 
pip install -qqq wandb
```

Currently, we have following datasets:
- ShapeNet (Sofas)
- ...

In order to download them, run this command in your environment:
```shell
wandb artifact get ml43d-project/3dr2n2/sofas:latest --type model
unzip -qq ./artifacts/sofas:v0/sofas_25_06.zip
```

