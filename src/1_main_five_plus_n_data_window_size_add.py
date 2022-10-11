# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
import argparse

import os
import numpy as np
import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T


class PennFudanDataset(object):
    def __init__(self, path, transforms, window_size, training):
        self.root = path
        self.transforms = transforms
        pth = os.listdir(path)
        self.label_sequences = []
        self.tr_replay = training
        self.dir_paths = []
        for i in pth:
            for training1 in self.tr_replay:
                if os.path.isdir(path + i) and i == str(training1):
                    self.dir_paths.append(path + i + '/')

        self.window_size = window_size

        self.seq_indexs = []
        index_a = 0
        for i in range(len(self.dir_paths)):
            if i == 0 :
                index_a = len(os.listdir(self.dir_paths[i]))-150
                self.seq_indexs.append((i, 0, index_a))
            else:
                nex_index_a = index_a +  len(os.listdir(self.dir_paths[i]))-150
                self.seq_indexs.append((i, index_a, nex_index_a))
                index_a = nex_index_a


    def __len__(self):
        return self.seq_indexs[-1][-1]

    def __getitem__(self, idx):
        # load images and masks
        for i, start, end in self.seq_indexs:
            if idx >= start and idx < end:
                real_idx = idx - start

                entire_data1 = np.load(self.dir_paths[i] + '/' + str(int(real_idx) + 147) + ".npy", allow_pickle=True)
                entire_data2 = np.load(self.dir_paths[i] + '/' + str(int(real_idx) + 148) + ".npy", allow_pickle=True)
                entire_data3 = np.load(self.dir_paths[i] + '/' + str(int(real_idx) + 149) + ".npy", allow_pickle=True)
                entire_data4 = np.load(self.dir_paths[i] + '/' + str(int(real_idx) + 150) + ".npy", allow_pickle=True)

                data1 = entire_data1[0]
                data2 = entire_data2[0]
                data3 = entire_data3[0]
                data4 = entire_data4[0]

                masks = entire_data4[1]

                break

        masks = np.array(masks)
        input_data = self.preprocessing(data1, data2, data3, data4)

        num_objs = len(masks)
        boxes = []

        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return input_data, target


    def preprocessing(self, data1, data2, data3, data4):
        temp = np.zeros([self.window_size, 9, data1.shape[1], data1.shape[2]])

        temp[0] = data1
        temp[1] = data2
        temp[2] = data3
        temp[3] = data4

        data = temp
        data = data.reshape(self.window_size*data.shape[1], data.shape[2], -1)
        return torch.FloatTensor(data) # 36, 128, 128


import torch.nn as nn
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


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main(args):

    torch.cuda.empty_cache()
    data_path = args.load_dir

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    dataset = PennFudanDataset(data_path, get_transform(train=False), window_size=4, training=args.training)
    dataset_test = PennFudanDataset(data_path, get_transform(train=False),window_size=4, training=args.training)

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-200:-150])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    model = get_model_instance_segmentation(num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = args.max_epoch
    data_path_test = args.save_dir
    os.makedirs(data_path_test, exist_ok=True)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)
        if epoch%1 == 0 :
            epoch = epoch+1
            torch.save(model.state_dict(), os.path.join(data_path_test, f"model_{epoch}.pth"))

    print("That's it!")


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--training", type=int, nargs="+")
    parser.add_argument("--load-dir", type=str, default=f"../data/result_ROCI/")
    parser.add_argument("--save-dir", type=str, default=f"../results/")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--max-epoch", type=int, default=6)
    args = parser.parse_args()

    main(args)

