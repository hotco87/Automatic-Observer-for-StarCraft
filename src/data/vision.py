import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def load(path: str, replay: int):
    return pd.read_csv(os.path.join(path, str(replay)) + ".rep.vision", index_col=None)


def interpolation(dataframe: pd.DataFrame, terminal_frame: int):
    # terminal_frame까지 ffill로 보간
    dataframe = dataframe.set_index('frame')
    dataframe = dataframe.reindex(range(terminal_frame))
    dataframe = dataframe.fillna(method='ffill')
    dataframe = dataframe.reset_index()
    return dataframe


def interval_sample(dataframe: pd.DataFrame, interval: int):
    return dataframe.loc[dataframe['frame'] % interval == 0]


def scatter_values_on_map(dataframe: pd.DataFrame, shape: (int, int)):
    vision = dataframe.iloc[0]['state']
    vision = np.array(list(vision)).astype(int)
    vision = vision.reshape(shape)      # (HEIGHT, WIDTH)
    vision = np.expand_dims(vision, 0)  # (1, HEIGHT, WIDTH)
    return vision


def preprocess(dataframe: pd.DataFrame, interval: int, terrain_shape: (int, int), terminal_frame: int):
    dataframe = interpolation(dataframe, terminal_frame)
    dataframe = interval_sample(dataframe, interval)

    channels = []
    for t in tqdm(list(range(0, terminal_frame, interval)), desc='Processing vision'):
        dataframe_t = dataframe.loc[dataframe['frame'] == t]
        channel_t = scatter_values_on_map(dataframe_t, terrain_shape)
        channels.append(channel_t)

    return np.stack(channels)
