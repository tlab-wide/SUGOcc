import numpy as np

nusc_class_frequencies = np.array(
    [
        9.44004e05,
        1.897170e06,
        1.52386e05,
        2.391677e06,
        1.6957802e07,
        7.24139e05,
        1.89027e05,
        2.074468e06,
        4.13451e05,
        2.38446e06,
        5.916653e06,
        1.75883646e08,
        4.275424e06,
        5.1393615e07,
        6.141162e07,
        1.05975596e08,
        1.16424404e08,
        1.89250063e09,
    ]
)
nusc_class_frequencies_womask = np.array(
    [
        2082349,
        3012970,
        234046,
        5385402,
        34146494,
        2044124,
        325765,
        3330253,
        543815,
        5785079,
        13521112,
        198278651,
        4895895,
        56540471,
        66504617,
        227803562,
        252374615,
        17126390780
    ]
)

nusc_class_names = [
    'others',
    'barrier',
    'bicycle',
    'bus',
    'car',
    'construction_vehicle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'trailer',
    'truck',
    'driveable_surface',
    'other_flat',
    'sidewalk',
    'terrain',
    'manmade',
    'vegetation',
    'free',
]