from pathlib import Path

import re

import torch


from backend.ai.model import (
    BatuSimTransformer,
)

from backend.ai.tokenizer import (
    Vocabulary,
    tokenize_input,
    rebuild_target_command,
    INPUT_VOCAB_FILE,
    TARGET_VOCAB_FILE,
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
)


# ============================================================
# BATUSIM NLP INFERENCE V3
# ============================================================
#
# MODEL:
#
#     Natural language
#           ↓
#     semantic command
#
#
# PYTHON:
#
#     Original numeric values
#           ↓
#     deterministic repair
#
#
# EXAMPLE
# ------------------------------------------------------------
#
# USER:
#
# xte 137 git sonra yde 83 derece dön
#
#
# RAW MODEL:
#
# MOVE|X|130;ROTATE|Y|8
#
#
# NUMERIC REPAIR:
#
# MOVE|X|137;ROTATE|Y|83
#
#
# TASK IR:
#
# {
#     "intent": "TASK_SEQUENCE",
#     "steps": [...]
# }
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


CHECKPOINT_PATH = (
    BASE_DIR
    /
    "checkpoints"
    /
    "batusim_transformer_best.pt"
)


# ============================================================
# CONFIG
# ============================================================

MAX_OUTPUT_LENGTH = 128


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )


    return torch.device(
        "cpu"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    device = get_device()


    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(

            (
                "Checkpoint bulunamadı:\n"
                f"{CHECKPOINT_PATH}"
            )

        )


    # ========================================================
    # VOCABS
    # ========================================================

    input_vocab = Vocabulary.load(
        INPUT_VOCAB_FILE
    )


    target_vocab = Vocabulary.load(
        TARGET_VOCAB_FILE
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint = torch.load(

        CHECKPOINT_PATH,

        map_location=
            device,

        weights_only=False

    )


    config = checkpoint[
        "model_config"
    ]


    # ========================================================
    # COMPATIBILITY
    # ========================================================

    if (
        config[
            "input_vocab_size"
        ]
        !=
        len(
            input_vocab
        )
    ):

        raise RuntimeError(

            (
                "Input vocabulary checkpoint ile uyuşmuyor. "
                "Tokenizer veya checkpoint yanlış olabilir."
            )

        )


    if (
        config[
            "target_vocab_size"
        ]
        !=
        len(
            target_vocab
        )
    ):

        raise RuntimeError(

            (
                "Target vocabulary checkpoint ile uyuşmuyor."
            )

        )


    # ========================================================
    # MODEL
    # ========================================================

    model = BatuSimTransformer(

        input_vocab_size=
            config[
                "input_vocab_size"
            ],

        target_vocab_size=
            config[
                "target_vocab_size"
            ],

        d_model=
            config[
                "d_model"
            ],

        nhead=
            config[
                "nhead"
            ],

        num_encoder_layers=
            config[
                "num_encoder_layers"
            ],

        num_decoder_layers=
            config[
                "num_decoder_layers"
            ],

        dim_feedforward=
            config[
                "dim_feedforward"
            ],

        dropout=
            config[
                "dropout"
            ],

        input_pad_id=
            config[
                "input_pad_id"
            ],

        target_pad_id=
            config[
                "target_pad_id"
            ]

    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )


    model = model.to(
        device
    )


    model.eval()


    return (

        model,

        input_vocab,

        target_vocab,

        device,

        checkpoint

    )


# ============================================================
# MODEL CACHE
# ============================================================
#
# FastAPI her request'te checkpoint'i tekrar diskten
# yüklemesin.
# ============================================================

_MODEL_CACHE = None


def get_runtime_model():

    global _MODEL_CACHE


    if (
        _MODEL_CACHE
        is None
    ):

        _MODEL_CACHE = load_model()


    return _MODEL_CACHE


# ============================================================
# AUTOREGRESSIVE MODEL GENERATION
# ============================================================

@torch.no_grad()
def generate_raw_command(
    text,
    model,
    input_vocab,
    target_vocab,
    device,
    max_output_length=MAX_OUTPUT_LENGTH
):

    text = str(
        text
        or
        ""
    ).strip()


    if not text:

        raise ValueError(
            "Input boş olamaz."
        )


    # ========================================================
    # INPUT
    # ========================================================

    input_tokens = tokenize_input(
        text
    )


    input_ids = input_vocab.encode(

        input_tokens,

        add_sos=True,

        add_eos=True

    )


    input_tensor = torch.tensor(

        [
            input_ids
        ],

        dtype=torch.long,

        device=device

    )


    # ========================================================
    # TARGET SPECIAL TOKENS
    # ========================================================

    sos_id = (
        target_vocab.token_to_id[
            SOS_TOKEN
        ]
    )


    eos_id = (
        target_vocab.token_to_id[
            EOS_TOKEN
        ]
    )


    pad_id = (
        target_vocab.token_to_id[
            PAD_TOKEN
        ]
    )


    generated_ids = [
        sos_id
    ]


    finished = False


    # ========================================================
    # AUTOREGRESSIVE LOOP
    # ========================================================

    for _ in range(
        max_output_length
    ):

        decoder_input = torch.tensor(

            [
                generated_ids
            ],

            dtype=torch.long,

            device=device

        )


        logits = model(

            input_tensor,

            decoder_input

        )


        next_token_logits = (

            logits[
                0,
                -1,
                :
            ]
            .clone()

        )


        # PAD ve SOS üretme.

        next_token_logits[
            pad_id
        ] = float(
            "-inf"
        )


        next_token_logits[
            sos_id
        ] = float(
            "-inf"
        )


        next_token_id = int(

            torch.argmax(
                next_token_logits
            ).item()

        )


        generated_ids.append(
            next_token_id
        )


        if (
            next_token_id
            ==
            eos_id
        ):

            finished = True

            break


    decoded_tokens = target_vocab.decode(

        generated_ids,

        remove_special=True

    )


    command = rebuild_target_command(
        decoded_tokens
    )


    return {

        "input_tokens":
            input_tokens,

        "output_tokens":
            decoded_tokens,

        "raw_command":
            command,

        "finished":
            finished,

    }


# ============================================================
# TEXT NORMALIZATION FOR NUMBER EXTRACTION
# ============================================================

def normalize_numeric_text(
    text
):

    text = str(
        text
        or
        ""
    ).lower()


    text = (
        text
        .replace(
            ",",
            "."
        )
        .replace(
            "°",
            " derece "
        )
    )


    text = re.sub(

        r"\s+",

        " ",

        text

    )


    return text.strip()


# ============================================================
# NUMBER EXTRACTION
# ============================================================
#
# Returns:
#
# [
#   {
#       "raw": 5,
#       "unit": "cm",
#       "distance_mm": 50
#   }
# ]
#
# ============================================================

def extract_numeric_values(
    text
):

    text = normalize_numeric_text(
        text
    )


    pattern = re.compile(

        r"""
        (?P<sign>[+-]?)
        \s*
        (?P<number>\d+(?:\.\d+)?)
        \s*
        (?P<unit>
            mm
            |
            cm
            |
            m
            |
            santim
            |
            santimetre
            |
            derece
        )?
        """,

        re.VERBOSE

    )


    values = []


    for match in pattern.finditer(
        text
    ):

        sign_text = (
            match.group(
                "sign"
            )
            or
            ""
        )


        number = float(

            match.group(
                "number"
            )

        )


        if sign_text == "-":

            number *= -1.0


        unit = (
            match.group(
                "unit"
            )
            or
            ""
        ).strip()


        unit = unit.lower()


        # ====================================================
        # DISTANCE MM CONVERSION
        # ====================================================

        if unit in (
            "cm",
            "santim",
            "santimetre"
        ):

            converted = (
                number
                *
                10.0
            )


        elif unit == "m":

            converted = (
                number
                *
                1000.0
            )


        else:

            # mm
            # degree
            # no unit

            converted = number


        values.append({

            "raw_number":
                number,

            "unit":
                unit,

            "converted":
                converted,

            "start":
                match.start(),

            "end":
                match.end(),

        })


    return values


# ============================================================
# INTEGER / FLOAT FORMAT
# ============================================================

def format_numeric_value(
    value
):

    value = float(
        value
    )


    if value.is_integer():

        return str(
            int(
                value
            )
        )


    return (
        f"{value:.6f}"
        .rstrip(
            "0"
        )
        .rstrip(
            "."
        )
    )


# ============================================================
# EXPECTED NUMERIC SLOT COUNT
# ============================================================

def command_numeric_slot_count(
    command_part
):

    parts = command_part.split(
        "|"
    )


    if not parts:

        return 0


    command_type = parts[
        0
    ]


    # ========================================================
    # MOVE
    # MOVE|X|100
    # ========================================================

    if command_type == "MOVE":

        return 1


    # ========================================================
    # ROTATE
    # ========================================================

    if command_type == "ROTATE":

        return 1


    # ========================================================
    # RETURN
    # ========================================================

    if command_type == "RETURN":

        return 0


    # ========================================================
    # SHAPE
    # ========================================================

    if command_type == "SHAPE":

        if len(
            parts
        ) < 3:

            return 0


        shape = parts[
            1
        ]


        if shape == "RECTANGLE":

            count = 2

        else:

            # CIRCLE
            # SQUARE
            # TRIANGLE

            count = 1


        if (
            "MOD_LINEAR"
            in parts
        ):

            count += 1


        if (
            "MOD_ROTATE"
            in parts
        ):

            count += 1


        return count


    return 0


# ============================================================
# REPLACE COMMAND NUMBERS
# ============================================================

def replace_numeric_slots(
    command_part,
    replacement_values
):

    parts = command_part.split(
        "|"
    )


    if not parts:

        return command_part


    command_type = parts[
        0
    ]


    replacements = iter(
        replacement_values
    )


    # ========================================================
    # MOVE
    # ========================================================

    if (
        command_type
        ==
        "MOVE"
    ):

        if len(
            parts
        ) >= 3:

            parts[
                2
            ] = next(

                replacements,

                parts[
                    2
                ]

            )


        return "|".join(
            parts
        )


    # ========================================================
    # ROTATE
    # ========================================================

    if (
        command_type
        ==
        "ROTATE"
    ):

        if len(
            parts
        ) >= 3:

            parts[
                2
            ] = next(

                replacements,

                parts[
                    2
                ]

            )


        return "|".join(
            parts
        )


    # ========================================================
    # SHAPE
    # ========================================================

    if (
        command_type
        ==
        "SHAPE"
    ):

        if len(
            parts
        ) < 4:

            return command_part


        shape = parts[
            1
        ]


        cursor = 3


        # ====================================================
        # BASE GEOMETRY
        # ====================================================

        if shape == "RECTANGLE":

            if len(
                parts
            ) > cursor:

                parts[
                    cursor
                ] = next(

                    replacements,

                    parts[
                        cursor
                    ]

                )


            cursor += 1


            if len(
                parts
            ) > cursor:

                parts[
                    cursor
                ] = next(

                    replacements,

                    parts[
                        cursor
                    ]

                )


            cursor += 1


        else:

            if len(
                parts
            ) > cursor:

                parts[
                    cursor
                ] = next(

                    replacements,

                    parts[
                        cursor
                    ]

                )


            cursor += 1


        # ====================================================
        # MODIFIER SEARCH
        # ====================================================

        for index, value in enumerate(
            parts
        ):

            if value == "MOD_LINEAR":

                numeric_index = (
                    index
                    +
                    2
                )


                if (
                    numeric_index
                    <
                    len(
                        parts
                    )
                ):

                    parts[
                        numeric_index
                    ] = next(

                        replacements,

                        parts[
                            numeric_index
                        ]

                    )


            elif value == "MOD_ROTATE":

                numeric_index = (
                    index
                    +
                    2
                )


                if (
                    numeric_index
                    <
                    len(
                        parts
                    )
                ):

                    parts[
                        numeric_index
                    ] = next(

                        replacements,

                        parts[
                            numeric_index
                        ]

                    )


        return "|".join(
            parts
        )


    return command_part


# ============================================================
# SIGN SEMANTICS
# ============================================================
#
# User:
#
# geri
# aşağı
# negatif
# eksi
#
# gibi şeyler yazmışsa
# numeric parser sign görmese bile negatif yapabiliriz.
#
# ============================================================

def infer_sign_from_context(
    text,
    value
):

    lowered = str(
        text
    ).lower()


    negative_words = [

        "geri",

        "aşağı",

        "asagi",

        "negatif",

        "eksi",

    ]


    if any(

        word
        in lowered

        for word
        in negative_words

    ):

        return (
            -abs(
                float(
                    value
                )
            )
        )


    return float(
        value
    )


# ============================================================
# SPLIT USER PROGRAM
# ============================================================
#
# Important:
#
# "çizerken"
# "aynı anda"
#
# split edilmez.
#
# Sequential connectors split edilir.
#
# ============================================================

def split_user_program(
    text
):

    text = str(
        text
    )


    pattern = re.compile(

        r"""
        \s*
        (?:
            ;
            |
            ,?\s+sonra\s+
            |
            ,?\s+ardından\s+
            |
            ,?\s+ardindan\s+
            |
            ,?\s+daha\s+sonra\s+
            |
            ,?\s+sonrasında\s+
            |
            ,?\s+sonrasinda\s+
            |
            ,?\s+ve\s+sonra\s+
        )
        \s*
        """,

        re.IGNORECASE
        |
        re.VERBOSE

    )


    segments = [

        segment.strip()

        for segment
        in pattern.split(
            text
        )

        if segment.strip()

    ]


    return segments


# ============================================================
# NUMERIC REPAIR
# ============================================================

def repair_command_numbers(
    user_text,
    raw_command
):

    command_parts = [

        part.strip()

        for part
        in raw_command.split(
            ";"
        )

        if part.strip()

    ]


    text_segments = split_user_program(
        user_text
    )


    # ========================================================
    # BEST CASE
    #
    # Natural language segments = model commands
    # ========================================================

    if (
        len(
            text_segments
        )
        ==
        len(
            command_parts
        )
    ):

        repaired_parts = []


        for segment, command_part in zip(

            text_segments,

            command_parts

        ):

            expected_count = (
                command_numeric_slot_count(
                    command_part
                )
            )


            numeric_values = (
                extract_numeric_values(
                    segment
                )
            )


            replacement_values = []


            for numeric in numeric_values[
                :expected_count
            ]:

                value = numeric[
                    "converted"
                ]


                value = infer_sign_from_context(

                    segment,

                    value

                )


                replacement_values.append(

                    format_numeric_value(
                        value
                    )

                )


            repaired_parts.append(

                replace_numeric_slots(

                    command_part,

                    replacement_values

                )

            )


        return ";".join(
            repaired_parts
        )


    # ========================================================
    # FALLBACK
    #
    # Global numeric order.
    # ========================================================

    all_numbers = extract_numeric_values(
        user_text
    )


    numeric_cursor = 0

    repaired_parts = []


    for command_part in command_parts:

        slot_count = (
            command_numeric_slot_count(
                command_part
            )
        )


        replacements = []


        for _ in range(
            slot_count
        ):

            if (
                numeric_cursor
                >=
                len(
                    all_numbers
                )
            ):

                break


            numeric = all_numbers[
                numeric_cursor
            ]


            numeric_cursor += 1


            replacements.append(

                format_numeric_value(

                    numeric[
                        "converted"
                    ]

                )

            )


        repaired_parts.append(

            replace_numeric_slots(

                command_part,

                replacements

            )

        )


    return ";".join(
        repaired_parts
    )


# ============================================================
# COMMAND → TASK IR
# ============================================================

def command_part_to_task(
    command_part
):

    parts = command_part.split(
        "|"
    )


    if not parts:

        raise ValueError(
            "Boş command."
        )


    command_type = parts[
        0
    ]


    # ========================================================
    # MOVE
    # ========================================================

    if command_type == "MOVE":

        if len(
            parts
        ) != 3:

            raise ValueError(

                f"Geçersiz MOVE command: {command_part}"

            )


        return {

            "action":
                "MOVE_RELATIVE",

            "axis":
                parts[
                    1
                ],

            "distance":
                float(
                    parts[
                        2
                    ]
                ),

            "frame":
                "WORLD",

        }


    # ========================================================
    # ROTATE
    # ========================================================

    if command_type == "ROTATE":

        if len(
            parts
        ) != 3:

            raise ValueError(

                f"Geçersiz ROTATE command: {command_part}"

            )


        return {

            "action":
                "ROTATE_RELATIVE",

            "axis":
                parts[
                    1
                ],

            "angle":
                float(
                    parts[
                        2
                    ]
                ),

            "frame":
                "WORLD",

        }


    # ========================================================
    # RETURN
    # ========================================================

    if command_type == "RETURN":

        if len(
            parts
        ) != 2:

            raise ValueError(

                f"Geçersiz RETURN command: {command_part}"

            )


        return {

            "action":
                "RETURN_TO_START",

            "mode":
                parts[
                    1
                ],

        }


    # ========================================================
    # SHAPE
    # ========================================================

    if command_type == "SHAPE":

        if len(
            parts
        ) < 4:

            raise ValueError(

                f"Geçersiz SHAPE command: {command_part}"

            )


        shape = parts[
            1
        ]


        plane = parts[
            2
        ]


        task = {

            "action":
                "DRAW_SHAPE",

            "shape":
                shape,

            "plane":
                plane,

            "reference":
                "CURRENT_TCP",

            "orientation_mode":
                "KEEP",

            "modifiers":
                [],

        }


        cursor = 3


        # ====================================================
        # SIZE
        # ====================================================

        if shape == "RECTANGLE":

            task[
                "width"
            ] = float(
                parts[
                    cursor
                ]
            )


            cursor += 1


            task[
                "height"
            ] = float(
                parts[
                    cursor
                ]
            )


            cursor += 1


        elif shape == "CIRCLE":

            task[
                "radius"
            ] = float(
                parts[
                    cursor
                ]
            )


            cursor += 1


        else:

            # SQUARE
            # TRIANGLE

            task[
                "size"
            ] = float(
                parts[
                    cursor
                ]
            )


            cursor += 1


        # ====================================================
        # MODIFIERS
        # ====================================================

        while (
            cursor
            <
            len(
                parts
            )
        ):

            modifier_type = parts[
                cursor
            ]


            # =================================================
            # LINEAR
            # =================================================

            if modifier_type == "MOD_LINEAR":

                if (
                    cursor
                    +
                    2
                    >=
                    len(
                        parts
                    )
                ):

                    raise ValueError(

                        (
                            "MOD_LINEAR command eksik: "
                            f"{command_part}"
                        )

                    )


                task[
                    "modifiers"
                ].append({

                    "type":
                        "LINEAR_PROGRESS",

                    "axis":
                        parts[
                            cursor + 1
                        ],

                    "distance":
                        float(
                            parts[
                                cursor + 2
                            ]
                        ),

                })


                cursor += 3


            # =================================================
            # ROTATE
            # =================================================

            elif modifier_type == "MOD_ROTATE":

                if (
                    cursor
                    +
                    2
                    >=
                    len(
                        parts
                    )
                ):

                    raise ValueError(

                        (
                            "MOD_ROTATE command eksik: "
                            f"{command_part}"
                        )

                    )


                task[
                    "modifiers"
                ].append({

                    "type":
                        "ROTATION_PROGRESS",

                    "axis":
                        parts[
                            cursor + 1
                        ],

                    "angle":
                        float(
                            parts[
                                cursor + 2
                            ]
                        ),

                })


                cursor += 3


            else:

                raise ValueError(

                    (
                        "Bilinmeyen SHAPE modifier: "
                        f"{modifier_type}"
                    )

                )


        return task


    raise ValueError(

        (
            "Desteklenmeyen AI command: "
            f"{command_type}"
        )

    )


# ============================================================
# COMMAND PROGRAM → TASK IR
# ============================================================

def command_to_task_ir(
    command
):

    command_parts = [

        part.strip()

        for part
        in command.split(
            ";"
        )

        if part.strip()

    ]


    if not command_parts:

        raise ValueError(
            "Command program boş."
        )


    steps = [

        command_part_to_task(
            command_part
        )

        for command_part
        in command_parts

    ]


    # ========================================================
    # SINGLE
    # ========================================================

    if len(
        steps
    ) == 1:

        task = steps[
            0
        ]


        return {

            "intent":
                task[
                    "action"
                ],

            **task

        }


    # ========================================================
    # MULTI
    # ========================================================

    return {

        "intent":
            "TASK_SEQUENCE",

        "steps":
            steps

    }


# ============================================================
# PUBLIC RUNTIME FUNCTION
# ============================================================
#
# FastAPI bundan çağıracak.
#
# ============================================================

def interpret_text(
    text
):

    (
        model,
        input_vocab,
        target_vocab,
        device,
        checkpoint

    ) = get_runtime_model()


    # ========================================================
    # AI
    # ========================================================

    model_result = generate_raw_command(

        text=

            text,

        model=

            model,

        input_vocab=

            input_vocab,

        target_vocab=

            target_vocab,

        device=

            device

    )


    raw_command = (
        model_result[
            "raw_command"
        ]
    )


    # ========================================================
    # NUMERIC REPAIR
    # ========================================================

    corrected_command = (
        repair_command_numbers(

            text,

            raw_command

        )
    )


    # ========================================================
    # TASK IR
    # ========================================================

    task_ir = command_to_task_ir(
        corrected_command
    )


    return {

        "success":
            True,

        "input":
            text,

        "raw_model_command":
            raw_command,

        "command":
            corrected_command,

        "task_ir":
            task_ir,

        "input_tokens":
            model_result[
                "input_tokens"
            ],

        "output_tokens":
            model_result[
                "output_tokens"
            ],

        "finished":
            model_result[
                "finished"
            ],

        "checkpoint_epoch":
            checkpoint.get(
                "epoch"
            ),

        "validation_exact_accuracy":
            checkpoint.get(
                "validation_exact_accuracy"
            ),

    }


# ============================================================
# TERMINAL
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "BATUSIM AI RUNTIME V3"
    )

    print(
        "=" * 78
    )


    (
        model,
        input_vocab,
        target_vocab,
        device,
        checkpoint

    ) = get_runtime_model()


    print()

    print(
        "Device:",
        device
    )


    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch"
        )
    )


    validation_exact = checkpoint.get(
        "validation_exact_accuracy"
    )


    if (
        validation_exact
        is not None
    ):

        print(

            "Validation exact:",

            f"{validation_exact * 100:.2f}%"

        )


    print()

    print(
        "Model hazır."
    )


    print(
        "Çıkmak için: exit"
    )


    # ========================================================
    # LOOP
    # ========================================================

    while True:

        print()

        print(
            "-" * 78
        )


        text = input(
            "BatuSim AI > "
        ).strip()


        if (
            text.lower()
            in (
                "exit",
                "quit",
                "q"
            )
        ):

            break


        if not text:

            continue


        try:

            result = interpret_text(
                text
            )


            print()

            print(
                "RAW MODEL:"
            )


            print(
                result[
                    "raw_model_command"
                ]
            )


            print()

            print(
                "NUMERIC REPAIRED:"
            )


            print(
                result[
                    "command"
                ]
            )


            print()

            print(
                "TASK IR:"
            )


            print(
                result[
                    "task_ir"
                ]
            )


        except Exception as error:

            print()

            print(
                "ERROR:"
            )


            print(
                str(
                    error
                )
            )


if __name__ == "__main__":

    main()