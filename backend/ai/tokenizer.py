import csv
import json
import re

from collections import Counter
from pathlib import Path


# ============================================================
# BATUSIM NLP TOKENIZER V2
# ============================================================
#
# V2 FEATURES
#
# 1) Multi-command target support
#
#    MOVE|X|100;ROTATE|Y|90
#
#
# 2) Digit-level number tokenization
#
#    185
#
#    ↓
#
#    "1", "8", "5"
#
#
# 3) Colloquial normalization
#
#    xte   -> x te
#    yde   -> y de
#    zde   -> z de
#
#    xyde  -> xy de
#    xzde  -> xz de
#    yzde  -> yz de
#
#
# 4) Attached units
#
#    50mm
#    ↓
#    5 0 mm
#
#
# 5) Numeric suffixes
#
#    30luk
#    ↓
#    3 0 luk
#
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


DATA_DIR = (
    BASE_DIR
    /
    "data"
)


TRAIN_FILE = (
    DATA_DIR
    /
    "train.csv"
)


TOKENIZER_DIR = (
    BASE_DIR
    /
    "tokenizer_data"
)


INPUT_VOCAB_FILE = (
    TOKENIZER_DIR
    /
    "input_vocab.json"
)


TARGET_VOCAB_FILE = (
    TOKENIZER_DIR
    /
    "target_vocab.json"
)


# ============================================================
# SPECIAL TOKENS
# ============================================================

PAD_TOKEN = "<PAD>"

SOS_TOKEN = "<SOS>"

EOS_TOKEN = "<EOS>"

UNK_TOKEN = "<UNK>"


SPECIAL_TOKENS = [

    PAD_TOKEN,

    SOS_TOKEN,

    EOS_TOKEN,

    UNK_TOKEN,

]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_input_text(
    text
):

    text = str(
        text
        or
        ""
    )


    text = (
        text
        .strip()
        .lower()
    )


    # ========================================================
    # DIFFERENT APOSTROPHE TYPES
    # ========================================================

    text = (
        text
        .replace(
            "’",
            "'"
        )
        .replace(
            "´",
            "'"
        )
        .replace(
            "`",
            "'"
        )
    )


    # ========================================================
    # DECIMAL COMMA
    #
    # 3,5 cm
    # ↓
    # 3.5 cm
    # ========================================================

    text = re.sub(

        r"(?<=\d),(?=\d)",

        ".",

        text

    )


    # ========================================================
    # DEGREE SYMBOL
    # ========================================================

    text = text.replace(

        "°",

        " derece "

    )


    # ========================================================
    # COMMON FUSED PLANE FORMS
    #
    # xyde
    # xzde
    # yzde
    #
    # IMPORTANT:
    #
    # longest patterns first.
    # ========================================================

    replacements = {

        r"\bxyde\b":
            "xy de",

        r"\bxyda\b":
            "xy da",

        r"\bxyte\b":
            "xy te",

        r"\bxyta\b":
            "xy ta",


        r"\bxzde\b":
            "xz de",

        r"\bxzda\b":
            "xz da",

        r"\bxzte\b":
            "xz te",

        r"\bxzta\b":
            "xz ta",


        r"\byzde\b":
            "yz de",

        r"\byzda\b":
            "yz da",

        r"\byzte\b":
            "yz te",

        r"\byzta\b":
            "yz ta",

    }


    for pattern, replacement in replacements.items():

        text = re.sub(

            pattern,

            replacement,

            text

        )


    # ========================================================
    # SINGLE AXIS FUSED FORMS
    #
    # xte
    # xde
    # yde
    # zde
    # ========================================================

    axis_replacements = {

        r"\bxte\b":
            "x te",

        r"\bxta\b":
            "x ta",

        r"\bxde\b":
            "x de",

        r"\bxda\b":
            "x da",


        r"\byte\b":
            "y te",

        r"\byta\b":
            "y ta",

        r"\byde\b":
            "y de",

        r"\byda\b":
            "y da",


        r"\bzte\b":
            "z te",

        r"\bzta\b":
            "z ta",

        r"\bzde\b":
            "z de",

        r"\bzda\b":
            "z da",

    }


    for pattern, replacement in axis_replacements.items():

        text = re.sub(

            pattern,

            replacement,

            text

        )


    # ========================================================
    # APOSTROPHE AXIS FORMS
    #
    # x'te
    # y'de
    # xy'de
    #
    # ↓
    #
    # x te
    # y de
    # xy de
    # ========================================================

    text = re.sub(

        r"\b(x|y|z|xy|xz|yz)'(te|ta|de|da)\b",

        r"\1 \2",

        text

    )


    # ========================================================
    # ATTACHED UNITS
    #
    # 50mm
    # ↓
    # 50 mm
    #
    # 3.5cm
    # ↓
    # 3.5 cm
    # ========================================================

    text = re.sub(

        r"(?<=\d)(mm|cm)\b",

        r" \1",

        text

    )


    # ========================================================
    # NUMBER + WORD SUFFIX
    #
    # 30luk
    # ↓
    # 30 luk
    #
    # 40lık
    # ↓
    # 40 lık
    #
    # ========================================================

    text = re.sub(

        r"(\d)(?=[a-zA-Zçğıöşü])",

        r"\1 ",

        text

    )


    # ========================================================
    # HYPHENATED PLANES
    #
    # x-y
    # ↓
    # xy
    #
    # x-z
    # ↓
    # xz
    # ========================================================

    text = re.sub(

        r"\bx\s*-\s*y\b",

        "xy",

        text

    )


    text = re.sub(

        r"\bx\s*-\s*z\b",

        "xz",

        text

    )


    text = re.sub(

        r"\by\s*-\s*z\b",

        "yz",

        text

    )


    # ========================================================
    # WHITESPACE
    # ========================================================

    text = re.sub(

        r"\s+",

        " ",

        text

    )


    return text.strip()


# ============================================================
# DIGIT TOKENIZER
# ============================================================

def tokenize_number(
    number_text
):

    tokens = []


    for character in str(
        number_text
    ):

        if character.isdigit():

            tokens.append(
                character
            )


        elif character == ".":

            tokens.append(
                "."
            )


    return tokens


# ============================================================
# INPUT TOKENIZATION
#
# Example:
#
# "xte 185 mm geri gel"
#
# normalize:
#
# "x te 185 mm geri gel"
#
# tokens:
#
# [
#   "x",
#   "te",
#   "1",
#   "8",
#   "5",
#   "mm",
#   "geri",
#   "gel"
# ]
#
# ============================================================

def tokenize_input(
    text
):

    text = normalize_input_text(
        text
    )


    # ========================================================
    # RAW TOKENS
    #
    # Numbers are captured as complete numeric strings first.
    # They are split digit-by-digit afterwards.
    # ========================================================

    raw_tokens = re.findall(

        r"""
        [a-zA-ZçğıöşüÇĞİÖŞÜ_]+
        |
        \d+(?:\.\d+)?
        |
        [-+]
        |
        [|;,():]
        """,

        text,

        re.VERBOSE

    )


    tokens = []


    for token in raw_tokens:

        # ====================================================
        # NUMBER
        # ====================================================

        if re.fullmatch(

            r"\d+(?:\.\d+)?",

            token

        ):

            tokens.extend(

                tokenize_number(
                    token
                )

            )


        else:

            tokens.append(
                token
            )


    return tokens


# ============================================================
# TARGET TOKENIZATION
#
# Example:
#
# MOVE|X|185;ROTATE|Y|-90
#
# ↓
#
# MOVE
# |
# X
# |
# 1
# 8
# 5
# ;
# ROTATE
# |
# Y
# |
# -
# 9
# 0
#
# ============================================================

def tokenize_target(
    text
):

    text = str(
        text
        or
        ""
    ).strip().upper()


    raw_tokens = re.findall(

        r"""
        [A-Z_]+
        |
        \d+(?:\.\d+)?
        |
        [-+]
        |
        \|
        |
        ;
        """,

        text,

        re.VERBOSE

    )


    tokens = []


    for token in raw_tokens:

        # ====================================================
        # NUMBERS → DIGITS
        # ====================================================

        if re.fullmatch(

            r"\d+(?:\.\d+)?",

            token

        ):

            tokens.extend(

                tokenize_number(
                    token
                )

            )


        else:

            tokens.append(
                token
            )


    return tokens


# ============================================================
# REBUILD TARGET COMMAND
#
# Tokens:
#
# [
#   MOVE, |, X, |, 1, 0, 0,
#   ;,
#   ROTATE, |, Y, |, 9, 0
# ]
#
# ↓
#
# MOVE|X|100;ROTATE|Y|90
#
# inference.py V2 bunu kullanabilir.
# ============================================================

def rebuild_target_command(
    tokens
):

    command = ""


    for token in tokens:

        if token in SPECIAL_TOKENS:

            continue


        if token in (

            "|",

            ";",

            "-",

            "+",

            "."

        ):

            command += token

            continue


        # Digit
        if token.isdigit():

            command += token

            continue


        # Command word
        #
        # MOVE / ROTATE / SHAPE etc.
        #
        # Target language'ta space gerekmiyor.

        command += token


    return command


# ============================================================
# VOCABULARY CLASS
# ============================================================

class Vocabulary:

    def __init__(
        self
    ):

        self.token_to_id = {}

        self.id_to_token = {}


    # ========================================================
    # SIZE
    # ========================================================

    def __len__(
        self
    ):

        return len(
            self.token_to_id
        )


    # ========================================================
    # ADD TOKEN
    # ========================================================

    def add_token(
        self,
        token
    ):

        if (
            token
            in self.token_to_id
        ):

            return self.token_to_id[
                token
            ]


        token_id = len(
            self.token_to_id
        )


        self.token_to_id[
            token
        ] = token_id


        self.id_to_token[
            token_id
        ] = token


        return token_id


    # ========================================================
    # BUILD VOCAB
    # ========================================================

    def build(
        self,
        token_sequences,
        min_frequency=1
    ):

        counter = Counter()


        for sequence in token_sequences:

            counter.update(
                sequence
            )


        # ====================================================
        # SPECIAL TOKENS ALWAYS FIRST
        # ====================================================

        for token in SPECIAL_TOKENS:

            self.add_token(
                token
            )


        # ====================================================
        # REST OF VOCAB
        # ====================================================

        sorted_tokens = sorted(

            counter.items(),

            key=lambda item: (

                -item[1],

                item[0]

            )

        )


        for token, frequency in sorted_tokens:

            if (
                frequency
                <
                min_frequency
            ):

                continue


            if token in self.token_to_id:

                continue


            self.add_token(
                token
            )


    # ========================================================
    # TOKEN → ID
    # ========================================================

    def token_id(
        self,
        token
    ):

        return self.token_to_id.get(

            token,

            self.token_to_id[
                UNK_TOKEN
            ]

        )


    # ========================================================
    # ID → TOKEN
    # ========================================================

    def id_token(
        self,
        token_id
    ):

        return self.id_to_token.get(

            int(
                token_id
            ),

            UNK_TOKEN

        )


    # ========================================================
    # ENCODE
    # ========================================================

    def encode(
        self,
        tokens,
        add_sos=True,
        add_eos=True
    ):

        token_ids = []


        if add_sos:

            token_ids.append(

                self.token_to_id[
                    SOS_TOKEN
                ]

            )


        for token in tokens:

            token_ids.append(

                self.token_id(
                    token
                )

            )


        if add_eos:

            token_ids.append(

                self.token_to_id[
                    EOS_TOKEN
                ]

            )


        return token_ids


    # ========================================================
    # DECODE
    # ========================================================

    def decode(
        self,
        token_ids,
        remove_special=True
    ):

        tokens = []


        for token_id in token_ids:

            token = self.id_token(
                token_id
            )


            if (
                remove_special
                and
                token
                in SPECIAL_TOKENS
            ):

                continue


            tokens.append(
                token
            )


        return tokens


    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        path
    ):

        path = Path(
            path
        )


        path.parent.mkdir(

            parents=True,

            exist_ok=True

        )


        data = {

            "version":
                2,

            "token_to_id":
                self.token_to_id,

            "special_tokens":
                SPECIAL_TOKENS,

            "digit_level_numbers":
                True,

            "multi_command":
                True,

        }


        with open(

            path,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                data,

                file,

                ensure_ascii=False,

                indent=2

            )


    # ========================================================
    # LOAD
    # ========================================================

    @classmethod
    def load(
        cls,
        path
    ):

        path = Path(
            path
        )


        with open(

            path,

            "r",

            encoding="utf-8"

        ) as file:

            data = json.load(
                file
            )


        vocab = cls()


        vocab.token_to_id = {

            str(
                token
            ):
                int(
                    token_id
                )

            for token, token_id
            in data[
                "token_to_id"
            ].items()

        }


        vocab.id_to_token = {

            int(
                token_id
            ):
                token

            for token, token_id
            in vocab.token_to_id.items()

        }


        return vocab


# ============================================================
# READ TRAIN DATA
# ============================================================

def read_training_rows():

    if not TRAIN_FILE.exists():

        raise FileNotFoundError(

            (
                "Train dataset bulunamadı:\n"
                f"{TRAIN_FILE}\n\n"
                "Önce dataset_generator.py çalıştır."
            )

        )


    rows = []


    with open(

        TRAIN_FILE,

        "r",

        encoding="utf-8"

    ) as file:

        reader = csv.DictReader(
            file
        )


        for row in reader:

            text = (
                row.get(
                    "text"
                )
                or
                ""
            ).strip()


            target = (
                row.get(
                    "target"
                )
                or
                ""
            ).strip()


            if (
                not text
                or
                not target
            ):

                continue


            rows.append({

                "text":
                    text,

                "target":
                    target,

            })


    return rows


# ============================================================
# BUILD TOKENIZERS
# ============================================================

def build_tokenizers():

    rows = read_training_rows()


    input_sequences = []

    target_sequences = []


    # ========================================================
    # TOKENIZE WHOLE TRAIN SET
    # ========================================================

    for row in rows:

        input_tokens = (
            tokenize_input(
                row[
                    "text"
                ]
            )
        )


        target_tokens = (
            tokenize_target(
                row[
                    "target"
                ]
            )
        )


        input_sequences.append(
            input_tokens
        )


        target_sequences.append(
            target_tokens
        )


    # ========================================================
    # BUILD VOCABS
    # ========================================================

    input_vocab = Vocabulary()

    target_vocab = Vocabulary()


    input_vocab.build(
        input_sequences
    )


    target_vocab.build(
        target_sequences
    )


    # ========================================================
    # SAVE
    # ========================================================

    TOKENIZER_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    input_vocab.save(
        INPUT_VOCAB_FILE
    )


    target_vocab.save(
        TARGET_VOCAB_FILE
    )


    return (

        input_vocab,

        target_vocab,

        rows

    )


# ============================================================
# CHECK UNKNOWN TOKENS
# ============================================================

def calculate_unknown_rate(
    rows,
    input_vocab
):

    unknown_id = (
        input_vocab.token_to_id[
            UNK_TOKEN
        ]
    )


    unknown_count = 0

    total_count = 0


    for row in rows:

        tokens = tokenize_input(
            row[
                "text"
            ]
        )


        ids = input_vocab.encode(

            tokens,

            add_sos=False,

            add_eos=False

        )


        for token_id in ids:

            total_count += 1


            if (
                token_id
                ==
                unknown_id
            ):

                unknown_count += 1


    if total_count == 0:

        return 0.0


    return (

        unknown_count

        /
        total_count

    )


# ============================================================
# SAMPLE DISPLAY
# ============================================================

def show_sample_encoding(
    input_vocab,
    target_vocab,
    rows,
    sample_count=12
):

    print()

    print(
        "=" * 78
    )

    print(
        "TOKENIZER V2 SAMPLE"
    )

    print(
        "=" * 78
    )


    # ========================================================
    # Prefer multi-command examples
    # ========================================================

    multi_rows = [

        row

        for row
        in rows

        if ";"
        in row[
            "target"
        ]

    ]


    examples = (

        multi_rows[
            :sample_count // 2
        ]

        +

        rows[
            :sample_count // 2
        ]

    )


    for row in examples:

        text = row[
            "text"
        ]


        target = row[
            "target"
        ]


        normalized = normalize_input_text(
            text
        )


        input_tokens = tokenize_input(
            text
        )


        target_tokens = tokenize_target(
            target
        )


        input_ids = input_vocab.encode(
            input_tokens
        )


        target_ids = target_vocab.encode(
            target_tokens
        )


        rebuilt_target = (
            rebuild_target_command(
                target_tokens
            )
        )


        print()

        print(
            "RAW INPUT:"
        )

        print(
            text
        )


        print()

        print(
            "NORMALIZED:"
        )

        print(
            normalized
        )


        print()

        print(
            "INPUT TOKENS:"
        )

        print(
            input_tokens
        )


        print()

        print(
            "INPUT IDS:"
        )

        print(
            input_ids
        )


        print()

        print(
            "TARGET:"
        )

        print(
            target
        )


        print()

        print(
            "TARGET TOKENS:"
        )

        print(
            target_tokens
        )


        print()

        print(
            "TARGET IDS:"
        )

        print(
            target_ids
        )


        print()

        print(
            "REBUILT TARGET:"
        )

        print(
            rebuilt_target
        )


        if rebuilt_target != target:

            print()

            print(
                "!!! TARGET REBUILD MISMATCH !!!"
            )


        print(
            "-" * 78
        )


# ============================================================
# CRITICAL ROBUSTNESS TESTS
# ============================================================

def run_critical_tokenizer_tests(
    input_vocab
):

    print()

    print(
        "=" * 78
    )

    print(
        "CRITICAL TOKENIZER TESTS"
    )

    print(
        "=" * 78
    )


    tests = [

        "xte 50 geri gel",

        "yde 90 derece dön",

        "zde 135 mm yukarı çık",

        "xyde 30luk kare çiz",

        "xzde 40lık daire çiz",

        (
            "xte 100 git sonra "
            "yde 90 derece dön"
        ),

        (
            "xyde 40 daire çizerken "
            "zde 137 mm yüksel"
        ),

    ]


    for text in tests:

        normalized = normalize_input_text(
            text
        )


        tokens = tokenize_input(
            text
        )


        ids = input_vocab.encode(
            tokens
        )


        print()

        print(
            "INPUT:"
        )

        print(
            text
        )


        print(
            "NORMALIZED:"
        )

        print(
            normalized
        )


        print(
            "TOKENS:"
        )

        print(
            tokens
        )


        print(
            "IDS:"
        )

        print(
            ids
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "BATUSIM NLP TOKENIZER V2"
    )

    print(
        "=" * 78
    )


    (
        input_vocab,
        target_vocab,
        rows

    ) = build_tokenizers()


    print()

    print(
        f"Training rows     : {len(rows)}"
    )


    print(
        f"Input vocab size  : {len(input_vocab)}"
    )


    print(
        f"Target vocab size : {len(target_vocab)}"
    )


    unknown_rate = (
        calculate_unknown_rate(

            rows,

            input_vocab

        )
    )


    print()

    print(
        "Train UNK rate    : "
        f"{unknown_rate * 100:.4f}%"
    )


    print()

    print(
        "Input vocab:"
    )

    print(
        INPUT_VOCAB_FILE
    )


    print()

    print(
        "Target vocab:"
    )

    print(
        TARGET_VOCAB_FILE
    )


    # ========================================================
    # VERIFY CRITICAL TARGET TOKENS
    # ========================================================

    required_target_tokens = [

        ";",
        "|",

        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",

        "MOVE",
        "ROTATE",
        "SHAPE",
        "RETURN",

    ]


    missing_tokens = [

        token

        for token
        in required_target_tokens

        if token
        not in target_vocab.token_to_id

    ]


    print()

    if missing_tokens:

        print(
            "WARNING - missing target tokens:"
        )

        print(
            missing_tokens
        )


    else:

        print(
            "Required target tokens: OK"
        )


    show_sample_encoding(

        input_vocab,

        target_vocab,

        rows

    )


    run_critical_tokenizer_tests(
        input_vocab
    )


    print()

    print(
        "=" * 78
    )

    print(
        "TOKENIZER V2 READY"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()