categories = {
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    61: 'toilet',
    62: 'tv',
    60: 'dining-table',
    69: 'oven',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    41: 'cup',
    39: 'bottle',
}

categories_id_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}

name2index = {
    # Original objectnav categories (HM3D closed set)
    "chair": 0,
    "sofa": 1,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv_monitor": 5,
    "tv": 5,
    "table": 6,
    "dining-table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14,
    # Verified GOAT val_unseen synonyms (from dataset category analysis)
    "fridge": 9,           # refrigerator
    "freezer": 9,          # refrigerator (same appliance)
    "flower vase": 12,     # vase
    "plant": 2,            # potted plant
    "decorative plant": 2, # potted plant
    "flowerpot": 2,        # potted plant
    # All other GOAT categories (mirror, picture, dresser, microwave, rug, carpet,
    # nightstand, pillow, stair, piano, etc.) are NOT in our 15-class detector.
    # They fall through to -1 → text-goal fallback in _init_subtask().
}
