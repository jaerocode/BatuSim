import csv
import random
import re
from pathlib import Path


# ============================================================
# BATUSIM NLP DATASET GENERATOR V2
# ============================================================
#
# V2 goals:
#
# 1) Single-command language
# 2) Multi-command programs
# 3) Casual / colloquial Turkish
# 4) Apostrophe-free forms
#       xte
#       yde
#       zde
#       xyde
# 5) Unit variation
#       mm
#       cm
#       m
# 6) Simultaneous path modifiers
#       MOD_LINEAR
#       MOD_ROTATE
# 7) More shape types
#       SQUARE
#       RECTANGLE
#       TRIANGLE
#       CIRCLE
#
#
# TARGET LANGUAGE
# ============================================================
#
# Single:
#
# MOVE|X|-50
#
# Multi:
#
# MOVE|X|100;ROTATE|Y|90
#
# Shape:
#
# SHAPE|SQUARE|XY|40
#
# Simultaneous:
#
# SHAPE|CIRCLE|XY|40|MOD_LINEAR|Z|50
#
# ============================================================


# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

DEFAULT_DATASET_SIZE = 50000

MAX_GENERATION_ATTEMPTS_FACTOR = 40


BASE_DIR = Path(
    __file__
).resolve().parent


OUTPUT_DIR = (
    BASE_DIR
    /
    "data"
)


OUTPUT_FILE = (
    OUTPUT_DIR
    /
    "batusim_nlp_dataset.csv"
)


random.seed(
    RANDOM_SEED
)


# ============================================================
# BASIC VALUES
# ============================================================

AXES = [
    "X",
    "Y",
    "Z",
]


PLANES = [
    "XY",
    "XZ",
    "YZ",
]


# Daha geniş sayı dağılımı.
#
# Tokenizer V2'de sayıları digit-level hale getireceğiz.

DISTANCES_MM = list(
    range(
        5,
        205,
        5
    )
)


DISTANCES_MM += [
    250,
    300,
    350,
    400,
    450,
    500,
]


ANGLES_DEG = [
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    60,
    75,
    90,
    100,
    120,
    135,
    150,
    180,
]


SHAPE_SIZES_MM = [
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    50,
    60,
    75,
    80,
    100,
    120,
    150,
]


CIRCLE_RADII_MM = [
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    50,
    60,
    75,
    80,
    100,
]


# ============================================================
# LANGUAGE DICTIONARIES
# ============================================================

AXIS_TEXT = {

    "X": [
        "x ekseninde",
        "x yönünde",
        "x dogrultusunda",
        "x doğrultusunda",
        "x ekseni boyunca",
        "x'te",
        "xte",
        "x tarafında",
    ],

    "Y": [
        "y ekseninde",
        "y yönünde",
        "y dogrultusunda",
        "y doğrultusunda",
        "y ekseni boyunca",
        "y'de",
        "yde",
        "y tarafında",
    ],

    "Z": [
        "z ekseninde",
        "z yönünde",
        "z dogrultusunda",
        "z doğrultusunda",
        "z ekseni boyunca",
        "z'de",
        "zde",
        "z tarafında",
    ],

}


ROTATION_AXIS_TEXT = {

    "X": [
        "x ekseninde",
        "x etrafında",
        "x etrafinda",
        "roll",
        "roll ekseninde",
    ],

    "Y": [
        "y ekseninde",
        "y etrafında",
        "y etrafinda",
        "pitch",
        "pitch ekseninde",
    ],

    "Z": [
        "z ekseninde",
        "z etrafında",
        "z etrafinda",
        "yaw",
        "yaw ekseninde",
    ],

}


PLANE_TEXT = {

    "XY": [
        "xy düzleminde",
        "x-y düzleminde",
        "xy duzleminde",
        "xy plane üzerinde",
        "xy üzerinde",
        "xy'de",
        "xyde",
    ],

    "XZ": [
        "xz düzleminde",
        "x-z düzleminde",
        "xz duzleminde",
        "xz plane üzerinde",
        "xz üzerinde",
        "xz'de",
        "xzde",
    ],

    "YZ": [
        "yz düzleminde",
        "y-z düzleminde",
        "yz duzleminde",
        "yz plane üzerinde",
        "yz üzerinde",
        "yz'de",
        "yzde",
    ],

}


MOVE_VERBS = [
    "git",
    "ilerle",
    "hareket et",
    "kay",
    "gitsin",
    "ilerlesin",
]


DRAW_VERBS = [
    "çiz",
    "ciz",
    "oluştur",
    "olustur",
    "çizdir",
    "cizdir",
    "yap",
    "çizsin",
    "cizsin",
]


ROTATE_VERBS = [
    "dön",
    "don",
    "döndür",
    "dondur",
    "çevir",
    "cevir",
    "dönsün",
    "donsun",
]


SEQUENCE_CONNECTORS = [
    " sonra ",
    " ardından ",
    " ardindan ",
    " ve sonra ",
    " daha sonra ",
    " sonrasında ",
    " sonrasinda ",
    ", sonra ",
    ", ardından ",
    "; sonra ",
]


# ============================================================
# TEXT UTILITIES
# ============================================================

def maybe_lowercase(
    text
):

    if random.random() < 0.85:

        return text.lower()


    return text


def maybe_remove_apostrophes(
    text
):

    if random.random() < 0.35:

        text = text.replace(
            "'",
            ""
        )


    return text


def maybe_join_units(
    text
):

    # 50 mm → 50mm
    # 5 cm  → 5cm

    if random.random() < 0.25:

        text = re.sub(
            r"(\d)\s+(mm|cm)\b",
            r"\1\2",
            text
        )


    return text


def maybe_remove_turkish_chars(
    text
):

    if random.random() >= 0.15:

        return text


    replacements = {

        "ç": "c",
        "Ç": "C",

        "ğ": "g",
        "Ğ": "G",

        "ı": "i",
        "İ": "I",

        "ö": "o",
        "Ö": "O",

        "ş": "s",
        "Ş": "S",

        "ü": "u",
        "Ü": "U",

    }


    for source, target in replacements.items():

        text = text.replace(
            source,
            target
        )


    return text


def maybe_add_casual_words(
    text
):

    if random.random() >= 0.15:

        return text


    prefix = random.choice([
        "",
        "robotu ",
        "tcp'yi ",
        "tcp yi ",
        "şimdi ",
        "simdi ",
        "bi ",
    ])


    return (
        prefix
        +
        text
    )


def apply_noise(
    text
):

    text = maybe_remove_apostrophes(
        text
    )


    text = maybe_join_units(
        text
    )


    text = maybe_remove_turkish_chars(
        text
    )


    text = maybe_add_casual_words(
        text
    )


    text = maybe_lowercase(
        text
    )


    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()


# ============================================================
# UNIT FORMATTING
# ============================================================

def format_mm(
    value_mm
):

    options = [
        f"{value_mm} mm"
    ]


    # ========================================================
    # CM
    # ========================================================

    if (
        value_mm
        %
        10
        ==
        0
    ):

        cm = (
            value_mm
            /
            10
        )


        if float(
            cm
        ).is_integer():

            cm = int(
                cm
            )


        options.extend([
            f"{cm} cm",
            f"{cm} santim",
        ])


    # ========================================================
    # DECIMAL CM
    # ========================================================

    if (
        value_mm
        %
        5
        ==
        0
        and
        value_mm
        <
        100
    ):

        cm = (
            value_mm
            /
            10
        )


        if not float(
            cm
        ).is_integer():

            options.append(
                f"{cm:g} cm"
            )


    # ========================================================
    # METERS
    # ========================================================

    if (
        value_mm
        >=
        100
        and
        value_mm
        %
        100
        ==
        0
    ):

        meter = (
            value_mm
            /
            1000
        )


        options.append(
            f"{meter:g} m"
        )


    return random.choice(
        options
    )


# ============================================================
# MOVE
# ============================================================

def generate_move_sample():

    axis = random.choice(
        AXES
    )


    distance = random.choice(
        DISTANCES_MM
    )


    sign = random.choice([
        1,
        -1,
    ])


    signed_distance = (
        sign
        *
        distance
    )


    distance_text = format_mm(
        distance
    )


    axis_text = random.choice(
        AXIS_TEXT[
            axis
        ]
    )


    verb = random.choice(
        MOVE_VERBS
    )


    # ========================================================
    # POSITIVE
    # ========================================================

    if sign > 0:

        templates = [

            f"{axis_text} {distance_text} {verb}",

            f"{distance_text} {axis_text} {verb}",

            (
                f"{axis_text} pozitif yönde "
                f"{distance_text} {verb}"
            ),

            (
                f"{axis_text} artı "
                f"{distance_text} {verb}"
            ),

        ]


        if axis == "Z":

            templates.extend([

                f"{distance_text} yukarı çık",

                f"{distance_text} yukari cik",

                f"zde {distance_text} yüksel",

                f"zde {distance_text} yuksel",

                f"tcp {distance_text} yukarı gelsin",

            ])


    # ========================================================
    # NEGATIVE
    # ========================================================

    else:

        templates = [

            (
                f"{axis_text} -{distance_text} "
                f"{verb}"
            ),

            (
                f"{axis_text} negatif yönde "
                f"{distance_text} {verb}"
            ),

            (
                f"{axis_text} eksi "
                f"{distance_text} {verb}"
            ),

        ]


        if axis == "X":

            templates.extend([

                f"x'te {distance_text} geri git",

                f"xte {distance_text} geri git",

                f"xte {distance_text} geri gel",

                f"xte {distance_text} geri gelsin",

                (
                    f"x yönünde {distance_text} "
                    f"geri çekil"
                ),

            ])


        elif axis == "Y":

            templates.extend([

                f"yde {distance_text} geri git",

                f"yde {distance_text} geri gel",

            ])


        elif axis == "Z":

            templates.extend([

                f"{distance_text} aşağı in",

                f"{distance_text} asagi in",

                f"zde {distance_text} aşağı in",

                f"zde {distance_text} asagi in",

                f"tcp {distance_text} aşağı gelsin",

            ])


    text = random.choice(
        templates
    )


    text = apply_noise(
        text
    )


    target = (
        f"MOVE|{axis}|{signed_distance}"
    )


    return text, target


# ============================================================
# ROTATION
# ============================================================

def generate_rotation_sample():

    axis = random.choice(
        AXES
    )


    angle = random.choice(
        ANGLES_DEG
    )


    sign = random.choice([
        1,
        -1,
    ])


    signed_angle = (
        sign
        *
        angle
    )


    axis_text = random.choice(
        ROTATION_AXIS_TEXT[
            axis
        ]
    )


    verb = random.choice(
        ROTATE_VERBS
    )


    if sign > 0:

        templates = [

            (
                f"{axis_text} "
                f"{angle} derece {verb}"
            ),

            (
                f"{angle} derece "
                f"{axis_text} {verb}"
            ),

            (
                f"{axis_text} pozitif yönde "
                f"{angle} derece {verb}"
            ),

        ]

    else:

        templates = [

            (
                f"{axis_text} -{angle} "
                f"derece {verb}"
            ),

            (
                f"{axis_text} negatif yönde "
                f"{angle} derece {verb}"
            ),

            (
                f"{axis_text} eksi "
                f"{angle} derece {verb}"
            ),

        ]


    text = apply_noise(
        random.choice(
            templates
        )
    )


    target = (
        f"ROTATE|{axis}|{signed_angle}"
    )


    return text, target


# ============================================================
# SQUARE
# ============================================================

def generate_square_sample(
    allow_modifier=True
):

    plane = random.choice(
        PLANES
    )


    size = random.choice(
        SHAPE_SIZES_MM
    )


    plane_text = random.choice(
        PLANE_TEXT[
            plane
        ]
    )


    size_text = format_mm(
        size
    )


    verb = random.choice(
        DRAW_VERBS
    )


    templates = [

        (
            f"{plane_text} {size_text} "
            f"kenarlı kare {verb}"
        ),

        (
            f"{plane_text} kenar uzunluğu "
            f"{size_text} olan kare {verb}"
        ),

        (
            f"{size_text} boyutunda "
            f"{plane_text} kare {verb}"
        ),

        (
            f"{plane_text} {size_text}lik "
            f"bir kare {verb}"
        ),

        (
            f"{plane_text} {size_text} "
            f"kare {verb}"
        ),

    ]


    base_text = random.choice(
        templates
    )


    target = (
        f"SHAPE|SQUARE|{plane}|{size}"
    )


    # ========================================================
    # OPTIONAL MODIFIER
    # ========================================================

    if (
        allow_modifier
        and
        random.random()
        <
        0.30
    ):

        return add_random_shape_modifier(

            base_text,

            target

        )


    return (
        apply_noise(
            base_text
        ),

        target

    )


# ============================================================
# RECTANGLE
# ============================================================

def generate_rectangle_sample(
    allow_modifier=True
):

    plane = random.choice(
        PLANES
    )


    width = random.choice(
        SHAPE_SIZES_MM
    )


    height = random.choice(
        SHAPE_SIZES_MM
    )


    while (
        height
        ==
        width
    ):

        height = random.choice(
            SHAPE_SIZES_MM
        )


    plane_text = random.choice(
        PLANE_TEXT[
            plane
        ]
    )


    width_text = format_mm(
        width
    )


    height_text = format_mm(
        height
    )


    verb = random.choice(
        DRAW_VERBS
    )


    templates = [

        (
            f"{plane_text} {width_text} x "
            f"{height_text} dikdörtgen {verb}"
        ),

        (
            f"{plane_text} genişliği {width_text} "
            f"yüksekliği {height_text} olan "
            f"bir dikdörtgen {verb}"
        ),

        (
            f"{plane_text} {width_text} "
            f"ve {height_text} kenarlı "
            f"dikdörtgen {verb}"
        ),

    ]


    base_text = random.choice(
        templates
    )


    target = (

        f"SHAPE|RECTANGLE|{plane}|"
        f"{width}|{height}"

    )


    if (
        allow_modifier
        and
        random.random()
        <
        0.25
    ):

        return add_random_shape_modifier(

            base_text,

            target

        )


    return (

        apply_noise(
            base_text
        ),

        target

    )


# ============================================================
# TRIANGLE
# ============================================================

def generate_triangle_sample(
    allow_modifier=True
):

    plane = random.choice(
        PLANES
    )


    size = random.choice(
        SHAPE_SIZES_MM
    )


    plane_text = random.choice(
        PLANE_TEXT[
            plane
        ]
    )


    size_text = format_mm(
        size
    )


    verb = random.choice(
        DRAW_VERBS
    )


    templates = [

        (
            f"{plane_text} {size_text} "
            f"kenarlı üçgen {verb}"
        ),

        (
            f"{plane_text} kenarı "
            f"{size_text} olan eşkenar "
            f"üçgen {verb}"
        ),

        (
            f"{size_text} boyutunda "
            f"{plane_text} üçgen {verb}"
        ),

    ]


    base_text = random.choice(
        templates
    )


    target = (
        f"SHAPE|TRIANGLE|{plane}|{size}"
    )


    if (
        allow_modifier
        and
        random.random()
        <
        0.25
    ):

        return add_random_shape_modifier(

            base_text,

            target

        )


    return (

        apply_noise(
            base_text
        ),

        target

    )


# ============================================================
# CIRCLE
# ============================================================

def generate_circle_sample(
    allow_modifier=True
):

    plane = random.choice(
        PLANES
    )


    radius = random.choice(
        CIRCLE_RADII_MM
    )


    plane_text = random.choice(
        PLANE_TEXT[
            plane
        ]
    )


    radius_text = format_mm(
        radius
    )


    verb = random.choice(
        DRAW_VERBS
    )


    templates = [

        (
            f"{plane_text} yarıçapı "
            f"{radius_text} olan daire {verb}"
        ),

        (
            f"{plane_text} {radius_text} "
            f"yarıçaplı daire {verb}"
        ),

        (
            f"{radius_text} radius ile "
            f"{plane_text} daire {verb}"
        ),

        (
            f"{plane_text} radius {radius_text} "
            f"daire {verb}"
        ),

    ]


    base_text = random.choice(
        templates
    )


    target = (
        f"SHAPE|CIRCLE|{plane}|{radius}"
    )


    if (
        allow_modifier
        and
        random.random()
        <
        0.45
    ):

        return add_random_shape_modifier(

            base_text,

            target

        )


    return (

        apply_noise(
            base_text
        ),

        target

    )


# ============================================================
# SIMULTANEOUS SHAPE MODIFIER
# ============================================================

def add_random_shape_modifier(
    base_text,
    base_target
):

    modifier_type = random.choice([
        "LINEAR",
        "ROTATE",
    ])


    # ========================================================
    # LINEAR PROGRESS
    # ========================================================

    if modifier_type == "LINEAR":

        axis = random.choice(
            AXES
        )


        distance = random.choice(
            DISTANCES_MM
        )


        sign = random.choice([
            1,
            -1,
        ])


        signed_distance = (
            sign
            *
            distance
        )


        distance_text = format_mm(
            distance
        )


        if sign > 0:

            if axis == "Z":

                modifier_text = random.choice([

                    (
                        f"zde {distance_text} "
                        f"yukarı çık"
                    ),

                    (
                        f"z ekseninde "
                        f"{distance_text} yüksel"
                    ),

                    (
                        f"z ekseninde "
                        f"{distance_text} ilerle"
                    ),

                ])

            else:

                modifier_text = (

                    f"{axis.lower()} ekseninde "
                    f"{distance_text} ilerle"

                )

        else:

            if axis == "Z":

                modifier_text = random.choice([

                    (
                        f"zde {distance_text} "
                        f"aşağı in"
                    ),

                    (
                        f"z ekseninde -"
                        f"{distance_text} ilerle"
                    ),

                ])

            else:

                modifier_text = (

                    f"{axis.lower()} ekseninde "
                    f"-{distance_text} ilerle"

                )


        combined_text = random.choice([

            (
                f"{base_text} çizerken "
                f"aynı anda {modifier_text}"
            ),

            (
                f"{base_text}, aynı anda "
                f"{modifier_text}"
            ),

            (
                f"{base_text} ve bunu yaparken "
                f"{modifier_text}"
            ),

            (
                f"{base_text}; eş zamanlı olarak "
                f"{modifier_text}"
            ),

        ])


        combined_target = (

            f"{base_target}"
            f"|MOD_LINEAR|{axis}|"
            f"{signed_distance}"

        )


    # ========================================================
    # ROTATION PROGRESS
    # ========================================================

    else:

        axis = random.choice(
            AXES
        )


        angle = random.choice(
            ANGLES_DEG
        )


        sign = random.choice([
            1,
            -1,
        ])


        signed_angle = (
            sign
            *
            angle
        )


        alias = {

            "X":
                random.choice([
                    "roll",
                    "x ekseninde",
                ]),

            "Y":
                random.choice([
                    "pitch",
                    "y ekseninde",
                ]),

            "Z":
                random.choice([
                    "yaw",
                    "z ekseninde",
                ]),

        }[
            axis
        ]


        if sign > 0:

            modifier_text = (
                f"{alias} {angle} derece dön"
            )

        else:

            modifier_text = (
                f"{alias} -{angle} derece dön"
            )


        combined_text = random.choice([

            (
                f"{base_text} çizerken "
                f"{modifier_text}"
            ),

            (
                f"{base_text}, aynı anda "
                f"{modifier_text}"
            ),

            (
                f"{base_text} ve eş zamanlı "
                f"{modifier_text}"
            ),

        ])


        combined_target = (

            f"{base_target}"
            f"|MOD_ROTATE|{axis}|"
            f"{signed_angle}"

        )


    return (

        apply_noise(
            combined_text
        ),

        combined_target

    )


# ============================================================
# RETURN
# ============================================================

def generate_return_sample():

    mode = random.choice([
        "TCP",
        "JOINTS",
    ])


    if mode == "TCP":

        templates = [

            "başlangıç konumuna geri dön",

            "baslangic konumuna geri don",

            "başladığın noktaya geri dön",

            "basladigin yere geri don",

            "ilk tcp konumuna dön",

            "başlangıç noktasına dön",

            "ilk noktaya geri gel",

        ]

    else:

        templates = [

            "başlangıç joint pozuna dön",

            "baslangic joint pozuna don",

            "ilk eklem konfigürasyonuna geri dön",

            "robotu başlangıç joint durumuna getir",

            "başlangıç eklem pozisyonuna dön",

            "ilk joint pozuna geri gel",

        ]


    return (

        apply_noise(
            random.choice(
                templates
            )
        ),

        f"RETURN|{mode}"

    )


# ============================================================
# SINGLE COMMAND GENERATORS
# ============================================================

SINGLE_GENERATORS = [

    generate_move_sample,

    generate_rotation_sample,

    generate_square_sample,

    generate_rectangle_sample,

    generate_triangle_sample,

    generate_circle_sample,

    generate_return_sample,

]


# ============================================================
# GENERATE SINGLE COMMAND
# ============================================================

def generate_single_command(
    allow_return=True
):

    generators = list(
        SINGLE_GENERATORS
    )


    if not allow_return:

        generators = [

            generator

            for generator
            in generators

            if generator
            !=
            generate_return_sample

        ]


    generator = random.choice(
        generators
    )


    return generator()


# ============================================================
# MULTI-COMMAND
#
# Example:
#
# INPUT:
#
# xte 100 git sonra yde 90 derece dön
#
# TARGET:
#
# MOVE|X|100;ROTATE|Y|90
# ============================================================

def generate_multi_command_sample():

    # 2 command most common.
    #
    # 3 command often.
    #
    # 4 command occasionally.

    command_count = random.choices(

        population=[
            2,
            3,
            4,
        ],

        weights=[
            0.60,
            0.30,
            0.10,
        ],

        k=1

    )[0]


    texts = []

    targets = []


    for index in range(
        command_count
    ):

        # RETURN yalnızca son komutta olsun.
        #
        # Böylece:
        #
        # return → sonra hareket
        #
        # gibi anlamsız training sample üretmeyelim.

        allow_return = (
            index
            ==
            command_count - 1
        )


        text, target = (
            generate_single_command(
                allow_return=
                    allow_return
            )
        )


        texts.append(
            text
        )


        targets.append(
            target
        )


    # ========================================================
    # JOIN NATURAL LANGUAGE
    # ========================================================

    combined_text = texts[
        0
    ]


    for text in texts[
        1:
    ]:

        connector = random.choice(
            SEQUENCE_CONNECTORS
        )


        combined_text += (
            connector
            +
            text
        )


    # Some users give no connector:
    #
    # "xte 100 git yde 90 derece dön"

    if (
        command_count
        ==
        2
        and
        random.random()
        <
        0.12
    ):

        combined_text = (

            texts[
                0
            ]

            +
            " "

            +
            texts[
                1
            ]

        )


    # ========================================================
    # OPTIONAL "ÖNCE"
    # ========================================================

    if random.random() < 0.25:

        prefix = random.choice([
            "önce ",
            "once ",
            "ilk olarak ",
        ])


        combined_text = (
            prefix
            +
            combined_text
        )


    combined_text = apply_noise(
        combined_text
    )


    # ========================================================
    # TARGET PROGRAM
    # ========================================================

    combined_target = ";".join(
        targets
    )


    return (
        combined_text,
        combined_target
    )


# ============================================================
# HAND-CURATED ROBUSTNESS SAMPLES
#
# Bunlar özellikle V1'de problem yaşadığımız
# gerçek kullanıcı tarzlarını dataset'e sokuyor.
# ============================================================

def generate_curated_samples():

    samples = [

        (
            "xte 50 geri gel",
            "MOVE|X|-50"
        ),

        (
            "xte 50 geri gelsin",
            "MOVE|X|-50"
        ),

        (
            "xte 100 git",
            "MOVE|X|100"
        ),

        (
            "yde 20 geri gel",
            "MOVE|Y|-20"
        ),

        (
            "zde 50 yukarı çık",
            "MOVE|Z|50"
        ),

        (
            "zde 50 aşağı in",
            "MOVE|Z|-50"
        ),

        (
            "yde 90 derece dön",
            "ROTATE|Y|90"
        ),

        (
            "yde 90 don",
            "ROTATE|Y|90"
        ),

        (
            "xte 100 git sonra yde 90 dön",
            "MOVE|X|100;ROTATE|Y|90"
        ),

        (
            "xte 100 git sonra yde 90 derece dön",
            "MOVE|X|100;ROTATE|Y|90"
        ),

        (
            "xte 50 geri gel sonra zde 20 yukarı çık",
            "MOVE|X|-50;MOVE|Z|20"
        ),

        (
            "xte 50 git yde 20 git",
            "MOVE|X|50;MOVE|Y|20"
        ),

        (
            "xyde 30luk kare çiz",
            "SHAPE|SQUARE|XY|30"
        ),

        (
            "xzde 20lik kare yap",
            "SHAPE|SQUARE|XZ|20"
        ),

        (
            "yzde 50lik üçgen çiz",
            "SHAPE|TRIANGLE|YZ|50"
        ),

        (
            "xyde 40 yarıçaplı daire çiz",
            "SHAPE|CIRCLE|XY|40"
        ),

        (
            "xyde 40lık daire çiz",
            "SHAPE|CIRCLE|XY|40"
        ),

        (
            "xyde 40 yarıçap daire çizerken zde 50 yüksel",
            (
                "SHAPE|CIRCLE|XY|40"
                "|MOD_LINEAR|Z|50"
            )
        ),

        (
            "xyde 40lık daire çizerken zde 50 yukarı çık",
            (
                "SHAPE|CIRCLE|XY|40"
                "|MOD_LINEAR|Z|50"
            )
        ),

        (
            "xyde 40 daire çiz aynı anda yaw 90 dön",
            (
                "SHAPE|CIRCLE|XY|40"
                "|MOD_ROTATE|Z|90"
            )
        ),

        (
            "xte 100 git sonra xyde 30luk kare çiz",
            (
                "MOVE|X|100;"
                "SHAPE|SQUARE|XY|30"
            )
        ),

        (
            "xte 100 git sonra xyde 30luk kare çiz sonra başlangıca dön",
            (
                "MOVE|X|100;"
                "SHAPE|SQUARE|XY|30;"
                "RETURN|TCP"
            )
        ),

        (
            "zde 50 yüksel ardından yde 90 dön sonra başlangıç noktasına dön",
            (
                "MOVE|Z|50;"
                "ROTATE|Y|90;"
                "RETURN|TCP"
            )
        ),

    ]


    return [

        {
            "text":
                text,

            "target":
                target,

        }

        for text, target
        in samples

    ]


# ============================================================
# SAMPLE TYPE SELECTOR
# ============================================================

def generate_random_sample():

    # ~55% single
    # ~45% multi

    if random.random() < 0.45:

        return generate_multi_command_sample()


    return generate_single_command()


# ============================================================
# GENERATE DATASET
# ============================================================

def generate_dataset(
    size=DEFAULT_DATASET_SIZE
):

    samples = []

    seen = set()


    # ========================================================
    # ADD CURATED FIRST
    # ========================================================

    for sample in generate_curated_samples():

        key = (

            sample[
                "text"
            ].strip().lower(),

            sample[
                "target"
            ]

        )


        if key in seen:

            continue


        seen.add(
            key
        )


        samples.append(
            sample
        )


    # ========================================================
    # RANDOM GENERATION
    # ========================================================

    attempts = 0


    max_attempts = (

        size

        *
        MAX_GENERATION_ATTEMPTS_FACTOR

    )


    while (
        len(
            samples
        )
        <
        size
        and
        attempts
        <
        max_attempts
    ):

        attempts += 1


        text, target = (
            generate_random_sample()
        )


        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()


        key = (
            text.lower(),
            target
        )


        if key in seen:

            continue


        seen.add(
            key
        )


        samples.append({

            "text":
                text,

            "target":
                target,

        })


    random.shuffle(
        samples
    )


    return samples


# ============================================================
# SPLIT
# ============================================================

def split_dataset(
    samples,
    train_ratio=0.80,
    validation_ratio=0.10
):

    total = len(
        samples
    )


    train_end = int(

        total

        *
        train_ratio

    )


    validation_end = int(

        total

        *
        (
            train_ratio
            +
            validation_ratio
        )

    )


    train = samples[
        :train_end
    ]


    validation = samples[
        train_end:validation_end
    ]


    test = samples[
        validation_end:
    ]


    return (
        train,
        validation,
        test
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    path,
    rows
):

    path = Path(
        path
    )


    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=[
                "text",
                "target",
            ]

        )


        writer.writeheader()


        writer.writerows(
            rows
        )


# ============================================================
# DATASET STATISTICS
# ============================================================

def calculate_statistics(
    samples
):

    statistics = {

        "single":
            0,

        "multi":
            0,

        "move":
            0,

        "rotate":
            0,

        "shape":
            0,

        "return":
            0,

        "modifier_linear":
            0,

        "modifier_rotate":
            0,

    }


    for sample in samples:

        target = sample[
            "target"
        ]


        if ";" in target:

            statistics[
                "multi"
            ] += 1

        else:

            statistics[
                "single"
            ] += 1


        if "MOVE|" in target:

            statistics[
                "move"
            ] += 1


        if "ROTATE|" in target:

            statistics[
                "rotate"
            ] += 1


        if "SHAPE|" in target:

            statistics[
                "shape"
            ] += 1


        if "RETURN|" in target:

            statistics[
                "return"
            ] += 1


        if "MOD_LINEAR" in target:

            statistics[
                "modifier_linear"
            ] += 1


        if "MOD_ROTATE" in target:

            statistics[
                "modifier_rotate"
            ] += 1


    return statistics


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "BATUSIM NLP DATASET GENERATOR V2"
    )

    print(
        "=" * 78
    )


    # ========================================================
    # GENERATE
    # ========================================================

    samples = generate_dataset(
        DEFAULT_DATASET_SIZE
    )


    # ========================================================
    # SPLIT
    # ========================================================

    (
        train,
        validation,
        test

    ) = split_dataset(
        samples
    )


    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    save_csv(

        OUTPUT_FILE,

        samples

    )


    save_csv(

        OUTPUT_DIR
        /
        "train.csv",

        train

    )


    save_csv(

        OUTPUT_DIR
        /
        "validation.csv",

        validation

    )


    save_csv(

        OUTPUT_DIR
        /
        "test.csv",

        test

    )


    # ========================================================
    # STATISTICS
    # ========================================================

    stats = calculate_statistics(
        samples
    )


    print()

    print(
        f"Total       : {len(samples)}"
    )


    print(
        f"Train       : {len(train)}"
    )


    print(
        f"Validation  : {len(validation)}"
    )


    print(
        f"Test        : {len(test)}"
    )


    print()

    print(
        "COMMAND DISTRIBUTION"
    )

    print(
        "-" * 40
    )


    for key, value in stats.items():

        percentage = (

            value

            /
            max(
                len(
                    samples
                ),
                1
            )

            *
            100.0

        )


        print(

            f"{key:<20}: "
            f"{value:>6} "
            f"({percentage:5.1f}%)"

        )


    # ========================================================
    # SAMPLE OUTPUTS
    # ========================================================

    print()

    print(
        "=" * 78
    )

    print(
        "SAMPLE OUTPUTS"
    )

    print(
        "=" * 78
    )


    # Multi-command örneklerini özellikle göster.

    multi_samples = [

        sample

        for sample
        in samples

        if ";"
        in sample[
            "target"
        ]

    ]


    random.shuffle(
        multi_samples
    )


    display_samples = (

        multi_samples[
            :10
        ]

        +

        samples[
            :10
        ]

    )


    for sample in display_samples:

        print()

        print(
            "INPUT :",
            sample[
                "text"
            ]
        )


        print(
            "TARGET:",
            sample[
                "target"
            ]
        )


    print()

    print(
        "=" * 78
    )

    print(
        "Dataset directory:"
    )

    print(
        OUTPUT_DIR
    )


    print()

    print(
        "IMPORTANT:"
    )

    print(
        (
            "V2 targets contain ';' for multi-command programs. "
            "Run tokenizer V2 before training."
        )
    )


if __name__ == "__main__":

    main()