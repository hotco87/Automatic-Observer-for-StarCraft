
import argparse

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import os
import torch
import torch.nn as nn

import torchvision
import transforms as T

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.backbone.body.conv1 = nn.Conv2d(36, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


class PennFudanDataset(object):
    def __init__(self, path, transforms, window_size):
        self.root = path
        self.transforms = transforms
        pth = os.listdir(path)
        self.label_sequences = []

        self.dir_paths = []
        if os.path.isdir(path):
            self.dir_paths.append(path + '/')
            print(f"Loaded {path}")

        self.window_size = window_size
        self.seq_indexs = []
        self.seq_indexs.append((0, 0, len(pth)))

    def __len__(self):
        return self.seq_indexs[-1][-1]

    def __getitem__(self, idx):
        # load images and masks
        for i, start, end in self.seq_indexs:
            if idx >= start and idx < end:
                real_idx = idx - start

                if real_idx > 3:
                    entire_data1 = np.load(self.dir_paths[i] + '/' + str(int(real_idx) - 3) + ".npy", allow_pickle=True)
                    entire_data2 = np.load(self.dir_paths[i] + '/' + str(int(real_idx) - 2) + ".npy", allow_pickle=True)
                    entire_data3 = np.load(self.dir_paths[i] + '/' + str(int(real_idx) - 1) + ".npy", allow_pickle=True)
                    entire_data4 = np.load(self.dir_paths[i] + '/' + str(int(real_idx) + 0) + ".npy", allow_pickle=True)
                else:
                    entire_data1 = np.load(self.dir_paths[i] + '/' + str(int(real_idx)) + ".npy", allow_pickle=True)
                    entire_data2 = entire_data1
                    entire_data3 = entire_data1
                    entire_data4 = entire_data1

                data1 = entire_data1[0]
                data2 = entire_data2[0]
                data3 = entire_data3[0]
                data4 = entire_data4[0]

                # masks = entire_data4[1]
                break

        input_data = self.preprocessing(data1, data2, data3, data4)
        return input_data

    def preprocessing(self, data1, data2, data3, data4):
        # 0 ground 1 air 2 building 3 spell 4 ground 5 air 6 building 7 spell 8 resource 9 vision 10 terrain
        temp = np.zeros([self.window_size, 9, data1.shape[1], data1.shape[2]])

        temp[0] = data1
        temp[1] = data2
        temp[2] = data3
        temp[3] = data4

        data = temp
        data = data.reshape(self.window_size * data.shape[1], data.shape[2], -1)
        return torch.FloatTensor(data)

def main(args):
    for i in args.testing:
        i = args.load_data_dir + str(i)
        epoch = str(args.model_epoch)
        print("generating replay_name: " , i)
        replay_name = int(i.split('/')[-1])
        dataset = PennFudanDataset(i, get_transform(train=False), window_size=4)

        dataset_len = len(dataset)
        Start, End, Step = 0, len(dataset), 1
        test_img_array = []
        test_img_one_channel_array = []

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("device: ",device)
        num_classes = 2
        model = get_model_instance_segmentation(num_classes)
        model.load_state_dict(torch.load(args.load_model_dir +f"model_{epoch}.pth",map_location=device))
        model.to(device)
        model.eval()

        print("dataset_size:", End)
        for i in range(Start, End, Step):
            img_t = dataset[i]
            test_img_array.append(img_t)
            img_one_channel = img_t.sum(axis=0, keepdim=True)
            test_img_one_channel_array.append(img_one_channel)

        print("input load")
        vpx_array = []
        vpy_array = []
        with torch.no_grad():
            for idx, i in tqdm(enumerate(test_img_array)):
                prediction = model(torch.unsqueeze(i, 0).to(device))
                if prediction[0]["boxes"].shape[0] == 0:
                    prediction = model(torch.unsqueeze(i - 1, 0).to(device))
                    vpx = int(prediction[0]["boxes"][0][0]) * 32
                    vpy = int(prediction[0]["boxes"][0][1]) * 32
                else:
                    vpx = int(prediction[0]["boxes"][0][0]) * 32
                    vpy = int(prediction[0]["boxes"][0][1]) * 32
                if idx % 500 == 0:
                    print("idx:  ", idx)

                vpx_array.append(vpx)
                vpy_array.append(vpy)

        #     dataset_len = 500
        os.makedirs(args.save_dir, exist_ok=True)
        temp = np.zeros((dataset_len, 1))
        for i in range(0, dataset_len):
            temp[i] = int(i * 8)
        temp2 = np.zeros((int(temp.max()), 1))
        for i in range(0, int(temp.max())):
            temp2[i] = i

        dataset_temp = pd.DataFrame({"frame": temp[:, 0], "vpx": vpx_array[:], "vpy": vpy_array[:]})
        dataset_temp2 = pd.DataFrame({"frame": temp2[:, 0]})
        dataset = pd.merge(left=dataset_temp2, right=dataset_temp, how="left", on="frame")
        dataset = dataset.fillna(method="ffill")
        dataset.to_csv(args.save_dir + str(replay_name) + ".rep.vpd", header=True, index=False)

        print("saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--testing", type=int, nargs="+")
    parser.add_argument("--load-data-dir", type=str, default=f"../data/result_ROCI/")
    parser.add_argument("--load-model-dir", type=str, default=f"../results/models/")
    parser.add_argument("--save-dir", type=str, default=f"../results/")
    parser.add_argument("--model-epoch", type=int, default=5)
    args = parser.parse_args()
    main(args)
