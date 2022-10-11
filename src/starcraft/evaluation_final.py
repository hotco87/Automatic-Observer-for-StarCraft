import numpy as np
import pandas as pd
import argparse
import copy

def load(label_path,replay_names):
    viewport_data_arr = []
    terminal_frame_arr = []

    for i in replay_names:
        viewport_data = pd.read_csv(label_path + i + ".rep.vpd", index_col=None)
        terminal_frame = int(viewport_data['frame'][-1:].item())

        viewport_data = viewport_data.set_index('frame')
        viewport_data = viewport_data.reindex(range(terminal_frame))
        viewport_data = viewport_data.fillna(method='ffill')
        viewport_data = viewport_data.reset_index()

        viewport_data['vpx'] = viewport_data['vpx'].astype(int)
        viewport_data['vpy'] = viewport_data['vpy'].astype(int)

        viewport_data_arr.append(viewport_data)
        terminal_frame_arr.append(terminal_frame)

    return viewport_data_arr, terminal_frame_arr

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def eval(labels_arr, min_length_arr):
    x_len = 20
    y_len = 12
    width = 128
    height = 128
    max_x = 3456
    max_y = 3720

    total_intersection = []
    intersection_multi = []
    is_intersect = []
    is_intersect_30 =[]
    is_intersect_50 = []

    for j in range (0, args.replay_names.__len__()):
        label_arr = [item[j] for item in labels_arr]
        min_length = min_length_arr[j]
        temp_intersect = []
        for i in range (0,min_length):
            total_tiles = np.zeros((width, height))
            #the first element in array is for testing
            for labels in label_arr[1:]:
                labels = labels[:min_length]
                x = int(np.round(labels['vpx'][i]/max_x*(width-x_len)))
                y = int(np.round(labels['vpy'][i]/max_y*(height-y_len)))
                total_tiles[x:x+x_len, y:y+y_len] =total_tiles[x:x+x_len, y:y+y_len]+1
                #total_tiles[labels['vpx'][i]:labels['vpx'][i]+x_len][labels['vpy'][i]:labels['vpy'][i]-y_len] =+1
            pred_x = int(np.round(label_arr[0]['vpx'][i]/max_x*(width-x_len)))
            pred_y = int(np.round(label_arr[0]['vpy'][i]/max_y*(height-y_len)))
            pred_tiles = total_tiles[pred_x:pred_x+x_len, pred_y:pred_y+y_len]

            intersect_multi = np.mean(pred_tiles)
            pred_tiles =np.where(pred_tiles==0, pred_tiles, 1)
            intersection = np.mean(pred_tiles)


            intersection_multi.append(intersect_multi)
            total_intersection.append(intersection)
            temp_intersect.append(intersection)

            if intersection!=0: is_intersect.append(1)
            else: is_intersect.append(0)
            if intersection>=0.3: is_intersect_30.append(1)
            else: is_intersect_30.append(0)
            if intersection>=0.5: is_intersect_50.append(1)
            else: is_intersect_50.append(0)

        temp_intersect = [0 if x != x else x for x in temp_intersect]
        print(str(args.replay_names[j]) + " :" + str(np.mean(temp_intersect)))
    total_intersection = [0 if x != x else x for x in total_intersection]
    return total_intersection, intersection_multi, is_intersect, is_intersect_30, is_intersect_50

def main(args):
    partial_length = args.eval_length
    model_name = args.model_name
    human_name = copy.deepcopy(args.human_names)

    print ("length: ", partial_length)
    if args.human_test:
        print("testing humans")
        print("model_name: ", args.human_names)
    else:
        print("model_name: ", model_name)
    print("----------------------------")

    total_intersection = []
    total_is_intersect = []
    total_is_intersect_30 = []
    total_is_intersect_50 = []
    total_multi_intersection = []
    min_length_arr = np.full((args.replay_names.__len__()),np.inf)
    min_length_arr = min_length_arr.tolist()

    for k in args.human_names:
        labels_arr = []

        if args.human_test:
            label_name = human_name
            label_name.remove(k)
            label_name.insert(0,k)
        else:
            label_name = [model_name] + human_name
            label_name.remove(k)
        print(label_name)
        print("testing")

        for i in range (0,label_name.__len__()):
            label_path = args.dir_path + label_name[i] +'/'
            label_arr, length_arr = load(label_path, args.replay_names)
            labels_arr.append(label_arr)

            for j in range (0,args.replay_names.__len__()):
                min_length_arr[j] = int(min(length_arr[j] * partial_length, min_length_arr[j]))

        print(k)
        intersection, multi_intersection, is_intersect, is_intersect_30, is_intersect_50 = eval(labels_arr, min_length_arr)
        print("intersection percent: ", np.mean(intersection), "total_multi intersection: ",np.mean(multi_intersection))
        print("is_intersect_percent: 0, 30, 50: ", np.mean(is_intersect), np.mean(is_intersect_30), np.mean(is_intersect_50))
        print("---------------------------------")
        total_intersection.append(np.mean(intersection))
        total_is_intersect.append(np.mean(is_intersect))
        total_is_intersect_30.append(np.mean(is_intersect_30))
        total_is_intersect_50.append(np.mean(is_intersect_50))
        total_multi_intersection.append(np.mean(multi_intersection))

    print("final")
    print("total_intersection percent: ", np.mean(total_intersection), "total_multi intersection: ",np.mean(total_multi_intersection))
    print("total_is_intersect_percent: 0, 30, 50: ", np.mean(total_is_intersect), np.mean(total_is_intersect_30), np.mean(total_is_intersect_50))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--dir-path", type=str, default=f"./labels/")
    parser.add_argument("--replay-names", '--list', nargs='+', default=["2351","1628","1559","6219","11251"])
    parser.add_argument("--model-name", type=str, default="rcnn_win_5")
    parser.add_argument("--human-names", '--list2', nargs='+', default=["1","2","3","4","5"])
    parser.add_argument("--human-test", action='store_true') #when testing average value of multiple humans, not the single model
    parser.add_argument("--eval-length", type=float, default=1)

    args = parser.parse_args()

    main(args)
