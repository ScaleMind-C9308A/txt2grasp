# Language Driven Object Grasping via Feature Augmentation

# Abstract 
This study tackles the problem of robotic object grasping through natural language instructions. First, a multi-modal network is proposed to combine both the visual and linguistic guidance information to localize the oriented object. Second, a feature augmentation is applied to enrich the information, to avoid overfitting. The overall experiments are conducted to evaluate the proposed method.

# Setup
Note that this repo is optimized for running on Google Colab
```
!git clone https://github.com/KhoiDOO/txt2grasp.git
%cd /content/txt2grasp
!pip install alive_progress wandb
```

If you run locally
```
git clone https://github.com/KhoiDOO/txt2grasp.git
cd /path/to/txt2grasp
python3 -m venv .env
source .env/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

# Training
All training parameters can be adjusted using ```main.py```
```
python main.py
```
Example
```
python main.py --bs 64 --epoch 100 --log --aug
```

## To toggle training with feature augmentation
```
python main.py --bs 64 --epoch 100 --log --aug --fa
```

## Wandb Logging
This repo allow logging to wandb. To toggle using argument ```--log```

If you find this repository helpful, please give a star :star2: :star2: :star2: