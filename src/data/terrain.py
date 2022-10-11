import os
import numpy as np

from common import config


def load(path: str, replay: int):
    return np.genfromtxt(os.path.join(path, str(replay)) + ".rep.terrain", delimiter=',', dtype=int)


def terrain_to_pixel(terrain_tile: np.ndarray):
    terrain = list()
    for h in range(terrain_tile.shape[0]):
        tmp_row = np.hstack([np.asarray([terrain_tile[h][w]] * config.TILE_SIZE) for w in range(terrain_tile.shape[1])])
        tmp_row = np.vstack([tmp_row] * config.TILE_SIZE)
        terrain.append(tmp_row)
    terrain = np.concatenate(terrain)
    return terrain


def preprocess(terrain_tile: np.ndarray):
    terrain_pixel = terrain_to_pixel(terrain_tile)
    return terrain_tile

