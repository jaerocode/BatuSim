import csv
from pathlib import Path

import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from backend.ai.tokenizer import (
    Vocabulary,
    tokenize_input,
    tokenize_target,
    INPUT_VOCAB_FILE,
    TARGET_VOCAB_FILE,
    PAD_TOKEN,
    EOS_TOKEN,
)


# ============================================================
# BATUSIM NLP DATASET V2
# ============================================================
#
# FEATURES
#
# - Single command
# - Multi-command
# - Digit-level numbers
# - Long target sequences
# - Dynamic batch padding
# - Truncation monitoring
# - Train / Validation / Test loaders
#
#
# Example target:
#
# MOVE|X|100;ROTATE|Y|90
#
#
# More complex:
#
# SHAPE|RECTANGLE|XY|80|30|MOD_ROTATE|X|-30;
# MOVE|Y|60;
# MOVE|Z|130
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


VALIDATION_FILE = (
    DATA_DIR
    /
    "validation.csv"
)


TEST_FILE = (
    DATA_DIR
    /
    "test.csv"
)


# ============================================================
# CONFIG
# ============================================================
#
# V1:
#
# input  = 48
# target = 32
#
# artık yeterli değil.
#
# Multi-command sample'lar için daha geniş bırakıyoruz.
#
# Dataset generator maksimum 4 sequential command üretiyor.
# ============================================================

DEFAULT_BATCH_SIZE = 64

MAX_INPUT_LENGTH = 128

MAX_TARGET_LENGTH = 128


# ============================================================
# READ CSV
# ============================================================

def read_csv_rows(
    path
):

    path = Path(
        path
    )


    if not path.exists():

        raise FileNotFoundError(

            (
                "Dataset bulunamadı:\n"
                f"{path}"
            )

        )


    rows = []


    with open(

        path,

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
# DATASET
# ============================================================

class BatuSimNLPRobotDataset(
    Dataset
):

    def __init__(
        self,
        csv_path,
        input_vocab,
        target_vocab,
        max_input_length=MAX_INPUT_LENGTH,
        max_target_length=MAX_TARGET_LENGTH
    ):

        self.csv_path = Path(
            csv_path
        )


        self.rows = read_csv_rows(
            self.csv_path
        )


        self.input_vocab = (
            input_vocab
        )


        self.target_vocab = (
            target_vocab
        )


        self.max_input_length = (
            max_input_length
        )


        self.max_target_length = (
            max_target_length
        )


        # ====================================================
        # STATISTICS
        # ====================================================

        self.input_truncated_count = 0

        self.target_truncated_count = 0

        self.max_observed_input_length = 0

        self.max_observed_target_length = 0


        self._analyze_lengths()


    # ========================================================
    # DATASET SIZE
    # ========================================================

    def __len__(
        self
    ):

        return len(
            self.rows
        )


    # ========================================================
    # ANALYZE TOKEN LENGTHS
    # ========================================================

    def _analyze_lengths(
        self
    ):

        input_truncated = 0

        target_truncated = 0


        max_input = 0

        max_target = 0


        for row in self.rows:

            # =================================================
            # INPUT
            # =================================================

            input_tokens = tokenize_input(
                row[
                    "text"
                ]
            )


            # +2:
            #
            # SOS
            # EOS

            input_length = (

                len(
                    input_tokens
                )

                +
                2

            )


            max_input = max(

                max_input,

                input_length

            )


            if (
                input_length
                >
                self.max_input_length
            ):

                input_truncated += 1


            # =================================================
            # TARGET
            # =================================================

            target_tokens = tokenize_target(
                row[
                    "target"
                ]
            )


            target_length = (

                len(
                    target_tokens
                )

                +
                2

            )


            max_target = max(

                max_target,

                target_length

            )


            if (
                target_length
                >
                self.max_target_length
            ):

                target_truncated += 1


        self.input_truncated_count = (
            input_truncated
        )


        self.target_truncated_count = (
            target_truncated
        )


        self.max_observed_input_length = (
            max_input
        )


        self.max_observed_target_length = (
            max_target
        )


    # ========================================================
    # SAFE TRUNCATION
    # ========================================================
    #
    # Eğer sequence limitten uzunsa:
    #
    # EOS mutlaka korunur.
    #
    # ========================================================

    def truncate_with_eos(
        self,
        ids,
        max_length,
        vocab
    ):

        if (
            len(
                ids
            )
            <=
            max_length
        ):

            return ids


        eos_id = (
            vocab.token_to_id[
                EOS_TOKEN
            ]
        )


        truncated = ids[
            :max_length
        ]


        truncated[
            -1
        ] = eos_id


        return truncated


    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        row = self.rows[
            index
        ]


        text = row[
            "text"
        ]


        target = row[
            "target"
        ]


        # ====================================================
        # INPUT TOKENIZATION
        # ====================================================

        input_tokens = tokenize_input(
            text
        )


        input_ids = (
            self.input_vocab.encode(

                input_tokens,

                add_sos=True,

                add_eos=True

            )
        )


        # ====================================================
        # TARGET TOKENIZATION
        # ====================================================

        target_tokens = tokenize_target(
            target
        )


        target_ids = (
            self.target_vocab.encode(

                target_tokens,

                add_sos=True,

                add_eos=True

            )
        )


        # ====================================================
        # SAFE TRUNCATION
        # ====================================================

        input_ids = (
            self.truncate_with_eos(

                input_ids,

                self.max_input_length,

                self.input_vocab

            )
        )


        target_ids = (
            self.truncate_with_eos(

                target_ids,

                self.max_target_length,

                self.target_vocab

            )
        )


        # ====================================================
        # SEQ2SEQ DECODER SHIFT
        # ====================================================
        #
        # TARGET:
        #
        # <SOS>
        # MOVE | X | 1 0 0 ;
        # ROTATE | Y | 9 0
        # <EOS>
        #
        #
        # DECODER INPUT:
        #
        # <SOS>
        # MOVE | X | 1 0 0 ;
        # ROTATE | Y | 9 0
        #
        #
        # EXPECTED OUTPUT:
        #
        # MOVE | X | 1 0 0 ;
        # ROTATE | Y | 9 0
        # <EOS>
        # ====================================================

        decoder_input_ids = (
            target_ids[
                :-1
            ]
        )


        decoder_target_ids = (
            target_ids[
                1:
            ]
        )


        return {

            "text":
                text,

            "target_text":
                target,

            "input_tokens":
                input_tokens,

            "target_tokens":
                target_tokens,

            "input_ids":
                torch.tensor(

                    input_ids,

                    dtype=torch.long

                ),

            "decoder_input_ids":
                torch.tensor(

                    decoder_input_ids,

                    dtype=torch.long

                ),

            "decoder_target_ids":
                torch.tensor(

                    decoder_target_ids,

                    dtype=torch.long

                ),

        }


# ============================================================
# COLLATOR
# ============================================================
#
# Batch içindeki sequence uzunluklarını eşitler.
#
# Örn:
#
# sample A:
#
# [1, 5, 7, 2]
#
# sample B:
#
# [1, 9, 4, 8, 6, 2]
#
#
# batch:
#
# [1,5,7,2,0,0]
# [1,9,4,8,6,2]
#
# ============================================================

class BatuSimCollator:

    def __init__(
        self,
        input_pad_id,
        target_pad_id
    ):

        self.input_pad_id = (
            input_pad_id
        )


        self.target_pad_id = (
            target_pad_id
        )


    # ========================================================
    # PAD ONE TENSOR
    # ========================================================

    def pad_tensor(
        self,
        tensor,
        desired_length,
        pad_id
    ):

        padding_length = (

            desired_length

            -
            len(
                tensor
            )

        )


        if (
            padding_length
            <=
            0
        ):

            return tensor


        padding = torch.full(

            (
                padding_length,
            ),

            pad_id,

            dtype=torch.long

        )


        return torch.cat([

            tensor,

            padding

        ])


    # ========================================================
    # COLLATE
    # ========================================================

    def __call__(
        self,
        batch
    ):

        # ====================================================
        # RAW DATA
        # ====================================================

        texts = [

            item[
                "text"
            ]

            for item
            in batch

        ]


        target_texts = [

            item[
                "target_text"
            ]

            for item
            in batch

        ]


        # ====================================================
        # MAX LENGTH IN CURRENT BATCH
        # ====================================================

        max_input_length = max(

            len(
                item[
                    "input_ids"
                ]
            )

            for item
            in batch

        )


        max_decoder_length = max(

            len(
                item[
                    "decoder_input_ids"
                ]
            )

            for item
            in batch

        )


        # ====================================================
        # PAD
        # ====================================================

        input_batch = []

        decoder_input_batch = []

        decoder_target_batch = []


        for item in batch:

            # =================================================
            # ENCODER INPUT
            # =================================================

            input_batch.append(

                self.pad_tensor(

                    item[
                        "input_ids"
                    ],

                    max_input_length,

                    self.input_pad_id

                )

            )


            # =================================================
            # DECODER INPUT
            # =================================================

            decoder_input_batch.append(

                self.pad_tensor(

                    item[
                        "decoder_input_ids"
                    ],

                    max_decoder_length,

                    self.target_pad_id

                )

            )


            # =================================================
            # DECODER TARGET
            # =================================================

            decoder_target_batch.append(

                self.pad_tensor(

                    item[
                        "decoder_target_ids"
                    ],

                    max_decoder_length,

                    self.target_pad_id

                )

            )


        # ====================================================
        # STACK → BATCH TENSORS
        # ====================================================

        input_ids = torch.stack(
            input_batch
        )


        decoder_input_ids = torch.stack(
            decoder_input_batch
        )


        decoder_target_ids = torch.stack(
            decoder_target_batch
        )


        # ====================================================
        # ATTENTION MASKS
        #
        # True:
        # real token
        #
        # False:
        # PAD
        # ====================================================

        input_attention_mask = (

            input_ids

            !=
            self.input_pad_id

        )


        decoder_attention_mask = (

            decoder_input_ids

            !=
            self.target_pad_id

        )


        return {

            "text":
                texts,

            "target_text":
                target_texts,

            "input_ids":
                input_ids,

            "input_attention_mask":
                input_attention_mask,

            "decoder_input_ids":
                decoder_input_ids,

            "decoder_attention_mask":
                decoder_attention_mask,

            "decoder_target_ids":
                decoder_target_ids,

        }


# ============================================================
# LOAD VOCABS
# ============================================================

def load_vocabularies():

    if not INPUT_VOCAB_FILE.exists():

        raise FileNotFoundError(

            (
                "Input vocabulary bulunamadı:\n"
                f"{INPUT_VOCAB_FILE}\n\n"
                "Önce tokenizer V2 çalıştır."
            )

        )


    if not TARGET_VOCAB_FILE.exists():

        raise FileNotFoundError(

            (
                "Target vocabulary bulunamadı:\n"
                f"{TARGET_VOCAB_FILE}\n\n"
                "Önce tokenizer V2 çalıştır."
            )

        )


    input_vocab = Vocabulary.load(
        INPUT_VOCAB_FILE
    )


    target_vocab = Vocabulary.load(
        TARGET_VOCAB_FILE
    )


    return (
        input_vocab,
        target_vocab
    )


# ============================================================
# CREATE DATASETS
# ============================================================

def create_datasets():

    (
        input_vocab,
        target_vocab

    ) = load_vocabularies()


    train_dataset = (
        BatuSimNLPRobotDataset(

            TRAIN_FILE,

            input_vocab,

            target_vocab

        )
    )


    validation_dataset = (
        BatuSimNLPRobotDataset(

            VALIDATION_FILE,

            input_vocab,

            target_vocab

        )
    )


    test_dataset = (
        BatuSimNLPRobotDataset(

            TEST_FILE,

            input_vocab,

            target_vocab

        )
    )


    return (

        train_dataset,

        validation_dataset,

        test_dataset,

        input_vocab,

        target_vocab

    )


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    batch_size=DEFAULT_BATCH_SIZE,
    num_workers=0
):

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        input_vocab,
        target_vocab

    ) = create_datasets()


    input_pad_id = (
        input_vocab.token_to_id[
            PAD_TOKEN
        ]
    )


    target_pad_id = (
        target_vocab.token_to_id[
            PAD_TOKEN
        ]
    )


    collator = BatuSimCollator(

        input_pad_id,

        target_pad_id

    )


    # ========================================================
    # TRAIN
    # ========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=
            batch_size,

        shuffle=
            True,

        num_workers=
            num_workers,

        collate_fn=
            collator,

        pin_memory=
            torch.cuda.is_available()

    )


    # ========================================================
    # VALIDATION
    # ========================================================

    validation_loader = DataLoader(

        validation_dataset,

        batch_size=
            batch_size,

        shuffle=
            False,

        num_workers=
            num_workers,

        collate_fn=
            collator,

        pin_memory=
            torch.cuda.is_available()

    )


    # ========================================================
    # TEST
    # ========================================================

    test_loader = DataLoader(

        test_dataset,

        batch_size=
            batch_size,

        shuffle=
            False,

        num_workers=
            num_workers,

        collate_fn=
            collator,

        pin_memory=
            torch.cuda.is_available()

    )


    return {

        "train":
            train_loader,

        "validation":
            validation_loader,

        "test":
            test_loader,

        "train_dataset":
            train_dataset,

        "validation_dataset":
            validation_dataset,

        "test_dataset":
            test_dataset,

        "input_vocab":
            input_vocab,

        "target_vocab":
            target_vocab,

    }


# ============================================================
# PRINT DATASET STATISTICS
# ============================================================

def print_dataset_statistics(
    name,
    dataset
):

    print()

    print(
        f"{name}"
    )

    print(
        "-" * 50
    )


    print(

        "Rows                 :",
        len(
            dataset
        )

    )


    print(

        "Max observed input   :",
        dataset.max_observed_input_length

    )


    print(

        "Max allowed input    :",
        dataset.max_input_length

    )


    print(

        "Input truncated      :",
        dataset.input_truncated_count

    )


    print()

    print(

        "Max observed target  :",
        dataset.max_observed_target_length

    )


    print(

        "Max allowed target   :",
        dataset.max_target_length

    )


    print(

        "Target truncated     :",
        dataset.target_truncated_count

    )


# ============================================================
# CHECK TRUNCATION
# ============================================================

def validate_no_truncation(
    datasets
):

    total_input_truncated = 0

    total_target_truncated = 0


    for dataset in datasets:

        total_input_truncated += (
            dataset.input_truncated_count
        )


        total_target_truncated += (
            dataset.target_truncated_count
        )


    print()

    print(
        "=" * 78
    )

    print(
        "TRUNCATION CHECK"
    )

    print(
        "=" * 78
    )


    print()

    print(
        "Input truncated :",
        total_input_truncated
    )


    print(
        "Target truncated:",
        total_target_truncated
    )


    if (
        total_input_truncated
        ==
        0
        and
        total_target_truncated
        ==
        0
    ):

        print()

        print(
            "TRUNCATION CHECK: OK"
        )


    else:

        print()

        print(
            "WARNING:"
        )


        print(
            (
                "Bazı sample'lar truncate ediliyor. "
                "MAX_INPUT_LENGTH veya "
                "MAX_TARGET_LENGTH artırılmalı."
            )
        )


# ============================================================
# DISPLAY SAMPLE BATCH
# ============================================================

def display_sample_batch(
    batch,
    target_vocab,
    sample_count=5
):

    print()

    print(
        "=" * 78
    )

    print(
        "SAMPLE BATCH"
    )

    print(
        "=" * 78
    )


    count = min(

        sample_count,

        len(
            batch[
                "text"
            ]
        )

    )


    for index in range(
        count
    ):

        print()

        print(
            "TEXT:"
        )


        print(
            batch[
                "text"
            ][
                index
            ]
        )


        print()

        print(
            "TARGET:"
        )


        print(
            batch[
                "target_text"
            ][
                index
            ]
        )


        print()

        print(
            "INPUT IDS:"
        )


        print(
            batch[
                "input_ids"
            ][
                index
            ].tolist()
        )


        print()

        print(
            "DECODER INPUT IDS:"
        )


        print(
            batch[
                "decoder_input_ids"
            ][
                index
            ].tolist()
        )


        print()

        print(
            "DECODER TARGET IDS:"
        )


        print(
            batch[
                "decoder_target_ids"
            ][
                index
            ].tolist()
        )


        # ====================================================
        # DECODE EXPECTED TARGET
        # ====================================================

        decoded_target = (
            target_vocab.decode(

                batch[
                    "decoder_target_ids"
                ][
                    index
                ].tolist(),

                remove_special=True

            )
        )


        print()

        print(
            "DECODED TARGET TOKENS:"
        )


        print(
            decoded_target
        )


        print(
            "-" * 78
        )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "BATUSIM NLP DATASET LOADER V2"
    )

    print(
        "=" * 78
    )


    # ========================================================
    # LOAD
    # ========================================================

    data = create_dataloaders(
        batch_size=8
    )


    train_loader = data[
        "train"
    ]


    validation_loader = data[
        "validation"
    ]


    test_loader = data[
        "test"
    ]


    train_dataset = data[
        "train_dataset"
    ]


    validation_dataset = data[
        "validation_dataset"
    ]


    test_dataset = data[
        "test_dataset"
    ]


    input_vocab = data[
        "input_vocab"
    ]


    target_vocab = data[
        "target_vocab"
    ]


    # ========================================================
    # BASIC INFO
    # ========================================================

    print()

    print(
        "Input vocab size :",
        len(
            input_vocab
        )
    )


    print(
        "Target vocab size:",
        len(
            target_vocab
        )
    )


    print()

    print(
        "Train batches    :",
        len(
            train_loader
        )
    )


    print(
        "Validation batches:",
        len(
            validation_loader
        )
    )


    print(
        "Test batches     :",
        len(
            test_loader
        )
    )


    # ========================================================
    # DATASET STATISTICS
    # ========================================================

    print_dataset_statistics(

        "TRAIN DATASET",

        train_dataset

    )


    print_dataset_statistics(

        "VALIDATION DATASET",

        validation_dataset

    )


    print_dataset_statistics(

        "TEST DATASET",

        test_dataset

    )


    # ========================================================
    # TRUNCATION
    # ========================================================

    validate_no_truncation([

        train_dataset,

        validation_dataset,

        test_dataset,

    ])


    # ========================================================
    # FIRST BATCH
    # ========================================================

    batch = next(

        iter(
            train_loader
        )

    )


    print()

    print(
        "=" * 78
    )

    print(
        "BATCH SHAPES"
    )

    print(
        "=" * 78
    )


    print()

    print(
        "input_ids:"
    )

    print(
        batch[
            "input_ids"
        ].shape
    )


    print()

    print(
        "input_attention_mask:"
    )

    print(
        batch[
            "input_attention_mask"
        ].shape
    )


    print()

    print(
        "decoder_input_ids:"
    )

    print(
        batch[
            "decoder_input_ids"
        ].shape
    )


    print()

    print(
        "decoder_attention_mask:"
    )

    print(
        batch[
            "decoder_attention_mask"
        ].shape
    )


    print()

    print(
        "decoder_target_ids:"
    )

    print(
        batch[
            "decoder_target_ids"
        ].shape
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    display_sample_batch(

        batch,

        target_vocab,

        sample_count=5

    )


    print()

    print(
        "=" * 78
    )

    print(
        "DATASET V2 READY"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()