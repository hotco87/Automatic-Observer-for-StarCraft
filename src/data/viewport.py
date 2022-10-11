import os
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from itertools import combinations
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure, binary_erosion
from scipy.spatial import distance

from common import config


def store_npy(results: list, dst_path: str, replay: int, method: str):
    dst_path = os.path.join(dst_path, "{}/".format(replay), "{}/".format(method))
    os.makedirs(dst_path, exist_ok=True)
    for t, result in enumerate(results):
        # np.save(dst_path + "{}.vpds.npy".format(t * config.INTERVAL), arr=result)
        np.save(dst_path + "{}.vpds.npy".format(t), arr=result)


def load(path: str, replay: int):
    vpd_paths = glob(os.path.join(path, "*/") + "{}.rep.vpd".format(str(replay)))
    vpds = [pd.read_csv(_, index_col=None) for _ in vpd_paths]
    return vpds


def interpolation(dataframes: list):
    result = []
    for dataframe in dataframes:
        dataframe = dataframe.set_index("frame")
        dataframe = dataframe.reindex(range(dataframe.tail(1).index[0]))
        dataframe = dataframe.fillna(method="ffill")
        dataframe = dataframe.reset_index()
        dataframe = dataframe.astype(int)
        dataframe = dataframe.set_index("frame")
        result.append(dataframe)
    return result


def merge_dataframes(dataframes: list):
    num_dataframes = len(dataframes)
    dataframe = pd.concat(dataframes, axis=1)
    dataframe = dataframe.fillna(method='ffill')
    dataframe = dataframe.astype(int)

    dataframe_columns = []
    for i in range(num_dataframes):
        dataframe_columns.append('vpx_{}'.format(i + 1))
        dataframe_columns.append('vpy_{}'.format(i + 1))
    dataframe.columns = dataframe_columns
    dataframe = (dataframe / config.TILE_SIZE).astype(int)
    dataframe = dataframe.reset_index()

    return dataframe, num_dataframes


def scatter_kernels_on_channel(dataframe: pd.DataFrame, num_dataframes: int,
                               kernel_shape=config.KERNEL_SHAPE, origin_shape=config.ORIGIN_SHAPE):
    channel = np.zeros(origin_shape)
    kernel = np.ones(kernel_shape)
    for i in range(num_dataframes):
        channel[
        dataframe['vpx_{}'.format(i + 1)].item():dataframe['vpx_{}'.format(i + 1)].item() + kernel_shape[0],
        dataframe['vpy_{}'.format(i + 1)].item():dataframe['vpy_{}'.format(i + 1)].item() + kernel_shape[1]
        ] += kernel
    return channel.T


def get_kernel_sum(channel: np.ndarray, origin_shape=config.ORIGIN_SHAPE, kernel_shape=config.KERNEL_SHAPE):
    width_tile = origin_shape[0] - kernel_shape[0]
    height_tile = origin_shape[1] - kernel_shape[1]
    result = np.zeros((width_tile, height_tile))
    for x in range(width_tile):
        for y in range(height_tile):
            result[x][y] = channel[x:x + kernel_shape[0], y:y + kernel_shape[1]].sum()
    return result


def get_viewport_argmax(channel: np.ndarray):
    max_viewport = np.argmax(channel)
    max_viewport_x_tile = (max_viewport % channel.shape[1])
    max_viewport_y_tile = (max_viewport // channel.shape[1])

    # max_viewport_x = max_viewport_x_tile * config.TILE_SIZE
    # max_viewport_y = max_viewport_y_tile * config.TILE_SIZE

    return max_viewport_x_tile, max_viewport_y_tile


def get_local_maximums(channel_kernel_sum: np.ndarray):
    # neighborhood = generate_binary_structure(2, 20)
    struct = generate_binary_structure(2, 2)
    neighborhood = iterate_structure(struct, 12).astype(int)
    local_max = maximum_filter(channel_kernel_sum, footprint=neighborhood) == channel_kernel_sum

    background = (channel_kernel_sum == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    peaks = local_max.astype(bool) ^ eroded_background

    return np.vstack(np.where(peaks)).T, peaks


def get_unique_peaks2(peaks: np.ndarray,
                      horizontal_threshold=config.KERNEL_SHAPE[0] * 0.8,
                      vertical_threshold=config.KERNEL_SHAPE[1] * 0.8,
                      similarity_threshold=0.6):
    def get_distance_axis(arr: np.ndarray, axis: int):
        res = np.zeros((arr[:, axis].shape[0], arr[:, axis].shape[0]))
        for i in range(arr[:, axis].shape[0]):
            for j in range(arr[:, axis].shape[0]):
                res[i][j] = np.abs(arr[:, axis][i] - arr[:, axis][j])
        return res

    def to_similarity_matrix(pairs: list, true_false_matrix: np.ndarray):
        length = true_false_matrix.shape[0]
        counts_equal = np.asarray([np.equal.reduce(true_false_matrix[pair, :]).sum() for pair in pairs])
        similarity = counts_equal / length

        sim_mat = np.zeros((length, length))
        for (i, j), sim in zip(pairs, similarity):
            sim_mat[i][j] = sim_mat[j][i] = sim
        return sim_mat

    def find_similar_things(sim_mat: np.ndarray, similarity_threshold: float):
        def add_result(arr: set):
            if len(res) == 0:
                return res.append(tuple(arr))
            for idx, a_res in enumerate(res):
                if len(arr.union(a_res)) == len(arr) + len(a_res):
                    res.append(tuple(arr))
                    return
                else:
                    if len(arr.difference(a_res)) < len(arr.intersection(a_res)):
                        res[idx] = tuple(arr.union(a_res))
                    else:
                        res.append(tuple(arr))
                    return

        comp_index = set(range(sim_mat.shape[0]))
        res = []
        ith = comp_index.pop()
        while True:
            sim_things = set((ith,)).union(set(np.where(sim_mat[:, ith] >= similarity_threshold)[0]))
            add_result(sim_things)
            comp_index -= sim_things

            if not len(comp_index) > 0:
                break
            ith = comp_index.pop()
        return res

    if peaks.shape[0] == 0:
        return
    result = []
    tmp_peaks = peaks
    while True:
        tf_mat_horizontal = get_distance_axis(tmp_peaks, axis=0) < horizontal_threshold
        tf_mat_vertical = get_distance_axis(tmp_peaks, axis=1) < vertical_threshold
        tf_mat = np.logical_and(tf_mat_horizontal, tf_mat_vertical)

        all_pairs = list(combinations(range(tf_mat.shape[0]), 2))
        similarity_matrix = to_similarity_matrix(all_pairs, tf_mat)
        similar_things = find_similar_things(similarity_matrix, similarity_threshold=0.6)

        tmp_result = np.unique(np.asarray([tmp_peaks[similar_thing, :].mean(axis=0).astype(int) for similar_thing in similar_things]), axis=0)
        tmp_result_tf_mat_horizontal = get_distance_axis(tmp_result, axis=0) < horizontal_threshold
        tmp_result_tf_mat_vertical = get_distance_axis(tmp_result, axis=1) < vertical_threshold
        tmp_tf_mat_horizontal_non_diagonal = tmp_result_tf_mat_horizontal[~np.eye(tmp_result_tf_mat_horizontal.shape[0], dtype=bool)].reshape(tmp_result_tf_mat_horizontal.shape[0], -1)
        tmp_tf_mat_vertical_non_diagonal = tmp_result_tf_mat_vertical[~np.eye(tmp_result_tf_mat_vertical.shape[0], dtype=bool)].reshape(tmp_result_tf_mat_vertical.shape[0], -1)

        if tmp_result.shape[0] == 1 or not (np.logical_and(tmp_tf_mat_horizontal_non_diagonal,
                                                           tmp_tf_mat_vertical_non_diagonal).any()):
            result.append(tmp_result)
            break

        tmp_peaks = tmp_result

    return np.vstack(result)


def get_unique_peaks(peaks: np.ndarray,
                     threshold=distance.euclidean((0, 0), config.KERNEL_SHAPE) / 2,
                     similarity_threshhold=0.9):
    def get_distance_each(arr: np.ndarray):
        tmp = arr.reshape((arr.shape[0], 1, arr.shape[1]))
        return np.sqrt(np.einsum('ijk, ijk->ij', arr - tmp, arr - tmp))

    def to_similarity_matrix(pairs: list, true_false_matrix: np.ndarray):
        length = true_false_matrix.shape[0]
        counts_equal = np.asarray([np.equal.reduce(true_false_matrix[pair, :]).sum() for pair in pairs])
        similarity = counts_equal / length

        sim_mat = np.zeros((length, length))
        for (i, j), sim in zip(pairs, similarity):
            sim_mat[i][j] = sim_mat[j][i] = sim
        return sim_mat

    def find_similar_things(sim_mat: np.ndarray, similarity_threshold: float):
        def add_result(arr):
            if len(res) == 0:
                return res.append(tuple(arr))
            for idx, a_res in enumerate(res):
                if len(arr.union(a_res)) == len(arr) + len(a_res):
                    res.append(tuple(arr))
                    return
                else:
                    if len(arr.difference(a_res)) < len(arr.intersection(a_res)):
                        res[idx] = tuple(arr.union(a_res))
                    else:
                        res.append(tuple(arr))
                    return
        comp_index = set(range(sim_mat.shape[0]))
        res = []
        ith = comp_index.pop()
        while True:
            sim_things = set((ith,)).union(set(np.where(sim_mat[:, ith] >= similarity_threshold)[0]))
            add_result(sim_things)
            comp_index -= sim_things

            if not len(comp_index) > 0:
                break
            ith = comp_index.pop()
        return res

    if peaks.shape[0] == 0:
        return
    result = []

    distance_each = get_distance_each(peaks)
    tf_mat = distance_each > threshold
    # tf_mat_except_diagonal = tf_mat[~np.eye(tf_mat.shape[0], dtype=bool)].reshape(tf_mat.shape[0], -1)

    all_pairs = list(combinations(range(tf_mat.shape[0]), 2))
    similarity_matrix = to_similarity_matrix(all_pairs, tf_mat)
    similar_things = find_similar_things(similarity_matrix, similarity_threshold=0.9)

    for similar_thing in similar_things:
        result.append(peaks[similar_thing, :].mean(axis=0).astype(int))
    result = np.unique(result, axis=0)
    # print()
    # print(result)
    # similar_things = find_similar_things(similarity_matrix, similarity_threshold=0.9)
    return result


# def get_unique_peaks_old2(peaks: np.ndarray, threshold=distance.euclidean((0, 0), config.KERNEL_SHAPE) / 2):
#     def distance_from_zero(arr):
#         return np.sqrt((arr[:, 0] - 0) ** 2 + (arr[:, 1] - 0) ** 2)
#
#     if peaks.shape[0] == 1:
#         return peaks  # return when peaks only have one
#
#     peaks = peaks[np.argsort(distance_from_zero(peaks))]
#     tmp = peaks.reshape((peaks.shape[0], 1, peaks.shape[1]))
#     distance_each = np.sqrt(np.einsum('ijk, ijk->ij', peaks - tmp, peaks - tmp))
#
#     tf_mat = distance_each > threshold
#     tf_mat_except_diagonal = tf_mat[~np.eye(tf_mat.shape[0], dtype=bool)].reshape(tf_mat.shape[0], -1)
#
#     all_true = np.bitwise_and.reduce(tf_mat_except_diagonal, axis=1)
#     isolated_idx = np.argwhere(all_true)
#     not_isolated_idx = np.argwhere(~all_true)
#
#     result = []
#     # isolated은 결과에 들어감
#     isolated = peaks[isolated_idx].reshape((-1, 2)).tolist()
#     if len(isolated) > 0:
#         result.append(isolated)
#     # not_isolated 처리
#     not_isolated = peaks[not_isolated_idx].reshape((-1, 2))
#     not_isolated_patterns = np.unique(tf_mat_except_diagonal[not_isolated_idx], axis=0).squeeze(axis=1)
#     # not_isolated_patterns = not_isolated_patterns.reshape((not_isolated_patterns[0], not_isolated_patterns[2]))
#     if not_isolated_patterns.shape[0] == 1:  # not_isolated의 패턴이 하나면 평균 (다 비슷한곳에 있는 것)
#         result.append(not_isolated.mean(axis=0).astype(int).tolist())
#     else:
#         pairs = list(combinations(range(not_isolated_patterns.shape[0]), 2))
#         length = not_isolated_patterns.shape[1]
#         counts_equal = np.asarray([np.equal.reduce(not_isolated_patterns[pair, :]).sum() for pair in pairs])
#
#         similarity = counts_equal/length
#         print(similarity)
#         infered_similar_patterns = np.asarray(pairs)[np.where(similarity > 0.90)]
#         if infered_similar_patterns.shape[0] > 0:
#             infered_cluster = np.unique(np.stack(infered_similar_patterns))
#             infered_viewports = np.vstack([np.argwhere(clustered_pattern == tf_mat_except_diagonal)
#                                            for clustered_pattern in not_isolated_patterns[infered_cluster]])
#
#
#         # result.append(get_unique_peaks2(not_isolated, threshold * 1.5))
#     # else:
#         # for pattern in not_isolated_patterns:
#         #     idxes = np.bitwise_and.reduce(np.equal(pattern, tf_mat_except_diagonal), axis=1)
#         #     niche = peaks[idxes]
#         #     result.append(get_unique_peaks2(niche, threshold + threshold/2))
#     # for pattern in not_isolated_patterns:
#     #     idxes = np.bitwise_and.reduce(np.equal(pattern, tf_mat_except_diagonal), axis=1)
#     #     niche = peaks[idxes]
#     #     result.append(niche.mean(axis=0).astype(int)[0].tolist())
#     #     result.append(get_unique_peaks2(niche, threshold + threshold/2))
#     return result
#
#
# def get_unique_peaks_old(peaks: np.ndarray, threshold=distance.euclidean((0, 0), config.KERNEL_SHAPE) / 2):
#     if peaks.shape[0] == 0:
#         return
#     tmp = peaks.reshape((peaks.shape[0], 1, peaks.shape[1]))
#     distance_each = np.sqrt(np.einsum('ijk, ijk->ij', peaks - tmp, peaks - tmp))
#
#     tf_mat = distance_each > threshold
#     if (tf_mat == False).all():
#         return peaks.mean(axis=0).astype(int)
#
#     tf_mat_except_diagonal = tf_mat[~np.eye(tf_mat.shape[0], dtype=bool)].reshape(tf_mat.shape[0], -1)
#     all_true = np.bitwise_and.reduce(tf_mat_except_diagonal, axis=1)
#     # 다른 걸로부터 독립적인 것들 (diagonal을 빼고 all true면 isolate)
#     # 서로 독립적이지 않은 것들 (diagonal을 빼고 all true가 아니면 non isolate)
#     isolated = np.where(all_true)
#     non_isolated = np.where(~all_true)
#     # 서로 독립적이지 않은 것들중 Unique한 것을 찾음
#     peaks_non_isolated = peaks[non_isolated]
#     # tf_mat_non_isolated = tf_mat[non_isolated]
#     # not_isolated_result = []
#     # # not_isolated_unique = np.unique(tf_mat_except_diagonal[~all_true], axis=0)
#     # not_isolated_unique = np.unique(tf_mat_non_isolated, axis=0)
#     # if not not_isolated_unique.shape[0] == 0:
#     #     for idx in range(not_isolated_unique.shape[0]):
#     #         niche = np.asarray([(tf_mat_non_isolated[tf_idx] == not_isolated_unique[idx]).all() for tf_idx in
#     #                             range(tf_mat_non_isolated.shape[0])])
#     #         not_isolated_result.append(peaks_non_isolated[niche].mean(axis=0).astype(int))
#
#     isolated_result = peaks[isolated]
#     not_isolated_result = get_unique_peaks(peaks_non_isolated, threshold + 1)
#
#     if isolated_result.shape[0] == 0:
#         unique_peaks = np.vstack([not_isolated_result])
#     elif not_isolated_result.shape[0] == 0:
#         unique_peaks = np.vstack([isolated_result])
#     else:
#         unique_peaks = np.vstack([isolated_result, not_isolated_result])
#     return unique_peaks


def preprocess_argmax_kernel_sum(dataframe: pd.DataFrame, num_vpds: int):
    """
    :param dataframe:
    :param num_vpds:
    :return: argmax kernel sum viewport point
    """
    # channels = []
    result = []
    for t in tqdm(list(range(0, len(dataframe), config.INTERVAL)), desc="Processing viewport(legacy)"):
        dataframe_t = dataframe.loc[dataframe["frame"] == t]
        channel_t = scatter_kernels_on_channel(dataframe_t, num_vpds)

        channel_t_kernel_sum = get_kernel_sum(channel_t)
        viewport_argmax_t = get_viewport_argmax(channel_t_kernel_sum)

        # peaks_t, channel_t_peaks = get_local_maximums(channel_t_kernel_sum)
        # unique_peaks_t = get_unique_peaks(peaks_t)

        # channels.append(channel_t)
        result.append(viewport_argmax_t)
    return result


def preprocess_consider_previous(dataframe: pd.DataFrame, num_vpds: int):
    """
    :param dataframe:
    :param num_vpds:
    :return: argmax kernel sum viewport point considering previous viewport point
    """

    def distance_2d(point1, point2):
        return np.hypot(point1[0] - point2[0], point1[1] - point2[1])

    def compare_with_previous(channel_kernel_sum: np.ndarray, unique_peaks: np.ndarray, viewport_previous: (int, int)):
        if viewport_previous is None:  # first frame
            idx = np.random.choice(unique_peaks.shape[0])
            result = unique_peaks[idx]
        else:
            distance = [distance_2d(peak, viewport_previous) for peak in unique_peaks]
            distance = np.asarray(distance)
            idx = np.argmin(distance)
            result = unique_peaks[idx]
        return result

    result = []
    viewport_previous = None
    for t in tqdm(list(range(0, len(dataframe), config.INTERVAL)), desc="Processing viewport(consider previous)"):
        dataframe_t = dataframe.loc[dataframe["frame"] == t]
        channel_t = scatter_kernels_on_channel(dataframe_t, num_vpds)

        channel_t_kernel_sum = get_kernel_sum(channel_t)
        peaks_t, channel_peaks = get_local_maximums(channel_t_kernel_sum)
        unique_peaks = get_unique_peaks(peaks_t)

        if not len(unique_peaks) == 1:
            viewport_current = compare_with_previous(channel_t_kernel_sum, unique_peaks, viewport_previous)
        else:
            viewport_current = unique_peaks[0]
            pass
        viewport_previous = viewport_current
        result.append(viewport_current)
    return result


def preprocess_unique_local_maximums(dataframe: pd.DataFrame, num_vpds: int):
    """
    :param dataframe:
    :param num_vpds:
    :return: local maximums of argmax kernel sum viewport point
    """
    result = []
    # counts = []
    for t in tqdm(list(range(0, len(dataframe), config.INTERVAL)), desc="Processing viewport(local_maximum)"):
        dataframe_t = dataframe.loc[dataframe["frame"] == t]
        channel_t = scatter_kernels_on_channel(dataframe_t, num_vpds)

        channel_t_kernel_sum = get_kernel_sum(channel_t)
        peaks_t, channel_t_peaks = get_local_maximums(channel_t_kernel_sum)

        unique_peaks = get_unique_peaks2(peaks_t)
        result.append(unique_peaks)
        # print()
        # print(unique_peaks)
        # counts.append(unique_peaks.shape[0])
    # counts = np.asarray(counts)
    # print("# label Number: min: {}, max: {}, average: {:.1f}~{:.2}".format(counts.min(), counts.max(), counts.mean(), counts.std()))
    return result


def preprocess_all_correct(dataframe: pd.DataFrame, num_vpds: int):
    result = []

    for t in tqdm(list(range(0, len(dataframe), config.INTERVAL)), desc="Processing viewport(all correct)"):
        dataframe_t = dataframe.loc[dataframe["frame"] == t]
        labels = np.split(np.asarray(dataframe_t.set_index('frame')).squeeze(), num_vpds)
        result.append(labels)

    return result

def preprocess(dataframes: list, method: str):
    dataframes = interpolation(dataframes)
    dataframe, num_vpds = merge_dataframes(dataframes)

    result = None
    if method == "legacy":
        result = preprocess_argmax_kernel_sum(dataframe, num_vpds)
    elif method == "consider_previous":
        result = preprocess_consider_previous(dataframe, num_vpds)
    elif method == "unique_local_maximums":
        result = preprocess_unique_local_maximums(dataframe, num_vpds)
    elif method == "all_correct":
        result = preprocess_all_correct(dataframe, num_vpds)
    else:
        raise NotImplementedError()
    return result