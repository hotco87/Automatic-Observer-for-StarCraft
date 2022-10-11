# Learning to automatically spectate games for Esports using object detection mechanism (Automatic Observer for StarCraft)

Human game observers, who control in-game cameras and provide viewers with an engaging experience, are a vital part of electronic sports (Esports) which has emerged as a rapidly growing industry in recent years. However, such a professional human observer poses several problems.
For example, they are prone to missing events occurring concurrently across the map. Further, human game observers are difficult to afford when only a small number of spectators are spectating the game.
Consequently, various methods to create automatic observers have been explored, and these methods are based on defining in-game events and focus on defined events. However, these event-based methods necessitate detailed predefined events, demanding high domain knowledge when developing. Additionally, these methods cannot show scenes that contain undefined events.
In this paper, we propose a method to overcome these problems by utilizing multiple human observational data and an object detection method, Mask R-CNN, in a real-time strategy game (e.g., StarCraft).By learning from human observational data, our method can observe scenes that are not defined as events in advance.
The proposed model utilizes an object detection mechanism to find the area where the human spectator is interested. We consider the pattern of the two-dimensional spatial area that the spectator is looking at as the object to find, rather than treating a single unit or building as an object which is conventionally done in object detection.
Consequently, we show that our automatic observer outperforms both current rule-based methods and human observers. The game observation video that compares our method and the rule-based method is available at https://www.youtube.com/watch?v=61JIfSrLHVk.

The paper's title is "Learning to automatically spectate games for Esports using object detection mechanism."
The paper is preprinted at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4173705

## Executing run file (Easy Usage) 
    #!/usr/bin/env bash
    set -ex    
    python3 -u 0_pre.py
    python3 -u 1_main_five_plus_n_data_window_size_add.py --training 36 212 --max-epoch 5
    mkdir -p ../results/models && mv ../results/*.pth ../results/models
    python3 -u 2_inference_five_n_plus_several_data_windowsize.py --testing 1725 --model-epoch 5    
    mkdir -p ../results/vpds && mv ../results/*.vpd ../results/vpds
    mkdir -p ../results/vpds/rcnn_ROCI && mv ../results/vpds/*.vpd ../results/vpds/rcnn_ROCI
    python3 3_evaluation_final.py --replay-names 1725 --model-name rcnn_ROCI --human-names 6 7 8 9 10
    python3 3_evaluation_final.py --replay-names 1725 --model-name sscait --dir-path ../data/vpds/ --human-names 6 7 8 9 10
    python3 3_evaluation_final.py --replay-names 1725 --model-name aiide --dir-path ../data/vpds/ --human-names 6 7 8 9 10
    python3 3_evaluation_final.py --replay-names 1725 --dir-path ../data/vpds/ --human-names 6 7 8 9 10 --human-test



## Training
Training can be done by running the following command

    python3 -u 1_main_five_plus_n_data_window_size_add.py --training 36 212 --max-epoch 5

[parameters]

--training: set the game replay that will be used for training.

--load-dir: set the directory where the training dataset is located.

--save-dir: set the directory that the model is saved.

--batch-size: set the batch size.

--learning-rate: set the learning rate.

--max-epoch: set the maximum epoch you want to train the model.


## 
## Testing (Inference)
After training, testing can be done by running the following command

    python3 -u 2_inference_five_n_plus_several_data_windowsize.py --testing 1725 --model-epoch 5

[parameters]

--testing: set the game replay that will be used for testing.

--load-data-dir: set the directory where the testing dataset is located.

--load-model-dir: set the directory where the trained model is located.

--save-dir: set the directory where the inferenced coordinate (vpd files) is saved.

--model-epoch: set the model epoch that you want to load.


## 
## Evaluation
After testing, evaluate the the performance of the predicted coordinates.
    
    python3 3_evaluation_final.py --replay-names 1725 --model-name rcnn_ROCI --human-names 6 7 8 9 10

    python3 3_evaluation_final.py --replay-names 1725 --model-name sscait --dir-path ../data/vpds/ --human-names 6 7 8 9 10

    python3 3_evaluation_final.py --replay-names 1725 --model-name aiide --dir-path ../data/vpds/ --human-names 6 7 8 9 10
    
    python3 3_evaluation_final.py --replay-names 1725 --dir-path ../data/vpds/ --human-names 6 7 8 9 10 --human-test

[parameters]

--dir-path: set the directory where the inferenced result is located.

--dir-path-human: set the directory where the human observational data is located.

--replay-names: set the game replay names that you want to evaluate.

--model-name: set the model name you want to evaluate.

--human-names: set the human observational dataset that you want to use for the evaluation.

--human-test: if used, evaluate the performance of the human instead of the model.


## Dataset

This code train and test automatic observer using Mask R-CNN model. Since the limitation of storage is 20GB, we uploaded only fraction of dataset for training and testing.
The data includes vpd files, which store coordinates of human watched (1~15) or model predicted(aiide, bc, rcnn_ROCi, sscait). The data also includes result_ROCI, which contains preprocessed input data and target data. Each directory contains data about a single game, and each npy file inside contains data of a single frame.

## Disclaimer
Our implemtation is under Pytorch's object detection baselines (https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html). and the implementation makes minimum changes over the official codebase.  "# Automatic-Observer-for-StarCraft" 
