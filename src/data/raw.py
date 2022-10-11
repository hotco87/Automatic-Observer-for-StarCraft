import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import config
from common import channel
from starcraft import UnitType

pd.options.mode.chained_assignment = None  # default='warn'


def load(path: str, replay: int):
    return pd.read_csv(os.path.join(path, str(replay)) + ".rep.raw", index_col=None)


def get_terminal_frame(dataframe: pd.DataFrame):
    players_in_frame = dataframe.loc[dataframe['player'] != 'Neutral'][['frame', 'player']].groupby(['frame', 'player'], as_index=False).agg('count')
    players_in_frame = players_in_frame.groupby('frame', as_index=False).agg('count')
    single_player_only = players_in_frame.loc[players_in_frame['player'] < 2]

    terminal_frame = single_player_only['frame'].min()
    if np.isnan(terminal_frame):
        terminal_frame = dataframe['frame'].max()
    return terminal_frame


def position_to_tileposition(dataframe: pd.DataFrame):
    dataframe.loc[:, 'x_tile'] = dataframe['x'].apply(lambda x: x // config.TILE_SIZE)
    dataframe.loc[:, 'y_tile'] = dataframe['y'].apply(lambda x: x // config.TILE_SIZE)

    dataframe.loc[:, 'left_tile'] = dataframe['left'].apply(lambda x: x / config.TILE_SIZE)
    dataframe.loc[:, 'right_tile'] = dataframe['right'].apply(lambda x: x / config.TILE_SIZE)
    dataframe.loc[:, 'top_tile'] = dataframe['top'].apply(lambda x: x / config.TILE_SIZE)
    dataframe.loc[:, 'bottom_tile'] = dataframe['bottom'].apply(lambda x: x / config.TILE_SIZE)

    dataframe.loc[:, 'left_tile'] = np.round(dataframe['left_tile'])
    dataframe.loc[:, 'right_tile'] = np.round(dataframe['right_tile'])
    dataframe.loc[:, 'top_tile'] = np.round(dataframe['top_tile'])
    dataframe.loc[:, 'bottom_tile'] = np.round(dataframe['bottom_tile'])

    dataframe.loc[:, 'width_tile'] = dataframe['right_tile'] - dataframe['left_tile']
    dataframe.loc[:, 'height_tile'] = dataframe['bottom_tile'] - dataframe['top_tile']
    return dataframe


def fix_neutral_player(dataframe: pd.DataFrame):
    # race is Zerg when player is Neutral
    dataframe.loc[dataframe['player'] == 'Neutral']['race'] = 'None'
    dataframe.loc[dataframe['player'] == 'Neutral']['player_color'] = 'Cyan'
    return dataframe


def interval_sample(dataframe: pd.DataFrame, interval: int):
    return dataframe.loc[dataframe['frame'] % interval == 0]


def scatter_values_on_map(dataframe, terrain_shape):
    channel = dataframe.pivot_table('name', 'y_tile', 'x_tile', aggfunc='count')
    channel = channel.reindex(range(terrain_shape[0]))
    channel = channel.T.reindex(range(terrain_shape[1]))
    channel = channel.fillna(0).astype(int).T
    channel = np.asarray(channel)
    return channel


def distance_2d(x_point, y_point, x, y):
    return np.hypot(x - x_point, y - y_point)


def spread_values_over_map(channel: np.ndarray, decay: float = 0.9):
    if channel.sum() == 0: return channel

    idx_occupied_cells = np.where(channel != 0)
    y, x = idx_occupied_cells[0], idx_occupied_cells[1]

    occupied_cells = []
    for i in range(len(y)):
        _x, _y, _value = x[i], y[i], channel[y[i]][x[i]]
        occupied_cells.append((_x, _y, _value))

    res = []
    for ord, (_x, _y, _value) in enumerate(occupied_cells):
        ys, xs = np.ogrid[0:channel.shape[0], 0:channel.shape[1]]
        distances = distance_2d(_x, _y, xs, ys)
        values_decayed = _value * decay ** distances
        res.append(np.expand_dims(values_decayed, axis=0))

    res = np.vstack(res)
    res = res.sum(axis=0)
    res -= res.min()  # up to [0, )
    # res /= res.max() # down to [0, 1]

    return res


def make_player_channels(dataframe: pd.DataFrame, terrain_shape: (int, int)):
    if len(dataframe) == 0:
        return

    dataframe.loc[:, 'isUnit'] = dataframe['name'].apply(lambda x: UnitType(x).isUnit)
    dataframe.loc[:, 'isWorker'] = dataframe['name'].apply(lambda x: UnitType(x).isWorker)
    dataframe.loc[:, 'isGround'] = dataframe['name'].apply(lambda x: UnitType(x).isGround)
    dataframe.loc[:, 'isAir'] = dataframe['name'].apply(lambda x: UnitType(x).isAir)
    dataframe.loc[:, 'isBuilding'] = dataframe['name'].apply(lambda x: UnitType(x).isBuilding)
    # dataframe.loc[:, 'isAddon'] = dataframe['name'].apply(lambda x: UnitType(x).isAddon)
    dataframe.loc[:, 'isSpell'] = dataframe['name'].apply(lambda x: UnitType(x).isSpell)
    dataframe.loc[:, 'isTrivial'] = dataframe['name'].apply(lambda x: UnitType(x).isTrivial)

    # units = dataframe.loc[dataframe['isUnit']]
    units_worker = dataframe.loc[dataframe['isWorker']]
    units_trivial = dataframe.loc[dataframe['isTrivial']]
    units_ground = dataframe.loc[dataframe['isGround']].loc[~dataframe['isWorker']].loc[~dataframe['isTrivial']]
    units_air = dataframe.loc[dataframe['isAir']]
    buildings = dataframe.loc[dataframe['isBuilding']]
    # addons = dataframe.loc[dataframe['isAddon']]
    spells = dataframe.loc[dataframe['isSpell']]

    channel_worker = scatter_values_on_map(units_worker, terrain_shape)
    channel_trivial = scatter_values_on_map(units_trivial, terrain_shape)
    channel_ground = scatter_values_on_map(units_ground, terrain_shape)
    channel_air = scatter_values_on_map(units_air, terrain_shape)
    channel_building = scatter_values_on_map(buildings, terrain_shape)
    channel_spell = scatter_values_on_map(spells, terrain_shape)

    channel_worker_spread = spread_values_over_map(channel_worker)
    channel_trivial_spread = spread_values_over_map(channel_trivial)
    channel_ground_spread = spread_values_over_map(channel_ground)
    channel_air_spread = spread_values_over_map(channel_air)
    channel_building_spread = spread_values_over_map(channel_building)
    channel_spell_spread = spread_values_over_map(channel_spell)

    return np.stack([channel_worker, channel_trivial, channel_ground,
                     channel_air, channel_building, channel_spell]),\
           np.stack([channel_worker_spread, channel_trivial_spread, channel_ground_spread,
                     channel_air_spread, channel_building_spread, channel_spell_spread])


def make_resource_channels(dataframe: pd.DataFrame, terrain_shape: (int, int)):
    if len(dataframe) == 0:
        return

    dataframe.loc[:, 'isResource'] = dataframe.apply(lambda row: UnitType(row['name']).isResource, axis=1)
    resources = dataframe.loc[dataframe['isResource']]
    channel_resource = scatter_values_on_map(resources, terrain_shape)  # (w, h)
    return np.expand_dims(channel_resource, 0)                          # (1, w, h)


def make_channels(dataframe: pd.DataFrame, terrain_shape: (int, int)):
    if len(dataframe) == 0:
        return

    players = list(dataframe['player'].unique())
    players.remove('Neutral')
    p1, p2 = players[0], players[1]

    dataframe_p1 = dataframe.loc[dataframe['player'] == p1]
    dataframe_p2 = dataframe.loc[dataframe['player'] == p2]
    dataframe_neutral = dataframe.loc[dataframe['player'] == 'Neutral']

    channels_p1, channels_p1_spread = make_player_channels(dataframe_p1, terrain_shape)
    channels_p2, channels_p2_spread = make_player_channels(dataframe_p2, terrain_shape)
    channels_resource = make_resource_channels(dataframe_neutral, terrain_shape)

    return np.vstack([channels_p1, channels_p2, channels_p1_spread, channels_p2_spread, channels_resource])


def make_delta_channels(current_channel: np.ndarray, previous_channel: np.ndarray):
    return current_channel - previous_channel


def preprocess(dataframe: pd.DataFrame, interval: int, terrain_shape: (int, int)):
    terminal_frame = get_terminal_frame(dataframe)

    dataframe = fix_neutral_player(dataframe)
    dataframe = interval_sample(dataframe, interval)
    dataframe = position_to_tileposition(dataframe)

    channels = []
    previous_channel = np.zeros((25,) + terrain_shape)
    for t in tqdm(list(range(0, terminal_frame, interval)), desc="Processing raw"):
        dataframe_t = dataframe.loc[dataframe['frame'] == t]
        channel_t = make_channels(dataframe_t, terrain_shape)
        delta_channel_t = make_delta_channels(channel_t, previous_channel)
        previous_channel = channel_t
        channels.append(channel_t)

    return np.stack(channels)
