
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

HUD_WIDTH = 0
HUD_HEIGHT = 96  # 640 x 384
# HUD_HEIGHT = 104  # 640 x 376

ORIGIN_SHAPE = (128, 128)
KERNEL_SHAPE = (20, 12)
TILE_SIZE = 32

FPS = 24
INTERVAL = 8

label_method = [
    # singular label
    "lagacy"
    # "argmax_kernel_sum",
    "consider_previous",
    # plural labels
    "unique_local_maximums",
    "all_correct",
]

output_method = [
    "coord",
    "channel"
]