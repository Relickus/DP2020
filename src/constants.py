import os

# CONSTANTS
SEPARATOR = "="*60
INTENSITY = 250

LEG_LEFT = 1
LEG_RIGHT = 2

ARGS_FILENAME =  os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/argsfile")
IMPLANTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),"resources/implants_all.npy")
TRIMDICT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),"resources/trim_dict_all")

LABEL_NEGATIVE = 0
LABEL_POSITIVE = 1

SAVE_FROM_EPOCH = 10

PATCH_D,PATCH_OVERLAY = 30, 10

PATCH_W, PATCH_H = 63,63

MASK_OUT,MASK_BONE,MASK_MARROW = 0,1,2
MASK_ALL = [0,1,2]

CLIP_MIN, CLIP_MAX = -200,1800

AUGM = {
    "noise":{
        "weak":(0.01,0.04),
        "mild":(0.05,0.09),
        "real":(0.1,0.19),
        "hard":(0.2,0.3),
    },
    "ellipsoid":{
        "weak":(0.02,0.07),
        "mild":(0.08,0.17),
        "real":(0.18,0.24),
        "hard":(0.25,0.35),
    }
    }
