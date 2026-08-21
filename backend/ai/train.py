from pathlib import Path

import math
import time

import torch
import torch.nn as nn


from backend.ai.dataset import (
    create_dataloaders,
)

from backend.ai.model import (
    BatuSimTransformer,
    count_parameters,
)

from backend.ai.tokenizer import (
    PAD_TOKEN,
)


# ============================================================
# BATUSIM NLP TRAINING V2
# ============================================================
#
# Dataset V2:
#
# - 50.000 total sample
# - 40.000 train
# - 5.000 validation
# - 5.000 test
#
# Supports:
#
# - single command
# - multi-command
# - MOVE
# - ROTATE
# - SHAPE
# - RETURN
# - MOD_LINEAR
# - MOD_ROTATE
# - digit-level numeric output
#
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
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


CHECKPOINT_DIR = (
    BASE_DIR
    /
    "checkpoints"
)


BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    /
    "batusim_transformer_best.pt"
)


LAST_MODEL_PATH = (
    CHECKPOINT_DIR
    /
    "batusim_transformer_last.pt"
)


# ============================================================
# TRAINING CONFIG
# ============================================================
#
# CPU'da uzun V2 sequence'ları nedeniyle
# batch size 32 daha güvenli.
#
# CUDA varsa aşağıda otomatik 64 kullanılacak.
# ============================================================

CPU_BATCH_SIZE = 32

CUDA_BATCH_SIZE = 64


NUM_EPOCHS = 30


LEARNING_RATE = 3e-4


WEIGHT_DECAY = 1e-4


GRAD_CLIP_NORM = 1.0


EARLY_STOPPING_PATIENCE = 6


# ============================================================
# LR SCHEDULER
# ============================================================

LR_SCHEDULER_FACTOR = 0.5

LR_SCHEDULER_PATIENCE = 2

MIN_LEARNING_RATE = 1e-6


# ============================================================
# PRINT CONFIG
# ============================================================

PRINT_BATCH_PROGRESS = True

PRINT_EVERY_N_BATCHES = 100


# ============================================================
# MODEL CONFIG
# ============================================================
#
# V1 model yapısını koruyoruz.
#
# İlk V2 eğitiminde modeli büyütmüyoruz.
# Önce dataset değişiminin etkisini görelim.
# ============================================================

D_MODEL = 128

NHEAD = 4

NUM_ENCODER_LAYERS = 3

NUM_DECODER_LAYERS = 3

DIM_FEEDFORWARD = 256

DROPOUT = 0.1


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42


def set_random_seed():

    torch.manual_seed(
        RANDOM_SEED
    )


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )


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
# BATCH SIZE
# ============================================================

def get_batch_size(
    device
):

    if device.type == "cuda":

        return CUDA_BATCH_SIZE


    return CPU_BATCH_SIZE


# ============================================================
# LOSS FUNCTION
# ============================================================

def create_loss_function(
    target_pad_id
):

    return nn.CrossEntropyLoss(

        ignore_index=
            target_pad_id

    )


# ============================================================
# TOKEN ACCURACY
# ============================================================
#
# Her output token ayrı ayrı değerlendirilir.
#
#
# Example:
#
# TARGET:
#
# MOVE|X|100
#
#
# PRED:
#
# MOVE|Y|100
#
#
# Çoğu token doğru olsa bile
# robot komutu yanlış olabilir.
#
# Bu yüzden exact sequence accuracy de ayrıca hesaplanıyor.
# ============================================================

def calculate_token_accuracy(
    logits,
    targets,
    pad_id
):

    predictions = torch.argmax(

        logits,

        dim=-1

    )


    valid_mask = (

        targets
        !=
        pad_id

    )


    correct_mask = (

        predictions
        ==
        targets

    ) & valid_mask


    correct_count = (
        correct_mask.sum().item()
    )


    total_count = (
        valid_mask.sum().item()
    )


    return (
        correct_count,
        total_count
    )


# ============================================================
# EXACT SEQUENCE ACCURACY
# ============================================================
#
# Bir sample ancak TÜM gerçek target tokenları doğruysa
# doğru sayılır.
#
#
# TARGET:
#
# MOVE|X|100;ROTATE|Y|90
#
#
# Tek token bile yanlışsa:
#
# exact = false
#
#
# Bu metric token accuracy'den çok daha anlamlı.
#
# NOT:
#
# Training / validation sırasında teacher forcing kullanılıyor.
# Gerçek final kaliteyi daha sonra autoregressive
# evaluate.py ile ölçeceğiz.
# ============================================================

def calculate_exact_sequence_accuracy(
    logits,
    targets,
    pad_id
):

    predictions = torch.argmax(

        logits,

        dim=-1

    )


    valid_mask = (

        targets
        !=
        pad_id

    )


    token_correct = (

        predictions
        ==
        targets

    )


    # Padding pozisyonlarını doğru kabul ediyoruz.
    #
    # Çünkü onlar gerçek command parçası değil.

    sequence_correct = (

        token_correct

        |
        ~valid_mask

    ).all(
        dim=1
    )


    correct_sequences = (
        sequence_correct.sum().item()
    )


    total_sequences = (
        targets.size(
            0
        )
    )


    return (
        correct_sequences,
        total_sequences
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_function,
    device,
    target_pad_id,
    epoch
):

    model.train()


    total_loss = 0.0


    total_correct_tokens = 0

    total_tokens = 0


    total_correct_sequences = 0

    total_sequences = 0


    epoch_start = time.time()


    # ========================================================
    # BATCH LOOP
    # ========================================================

    for batch_index, batch in enumerate(
        loader,
        start=1
    ):

        # ====================================================
        # MOVE DATA TO DEVICE
        # ====================================================

        input_ids = (
            batch[
                "input_ids"
            ]
            .to(
                device,
                non_blocking=True
            )
        )


        decoder_input_ids = (
            batch[
                "decoder_input_ids"
            ]
            .to(
                device,
                non_blocking=True
            )
        )


        decoder_target_ids = (
            batch[
                "decoder_target_ids"
            ]
            .to(
                device,
                non_blocking=True
            )
        )


        # ====================================================
        # CLEAR GRADIENT
        # ====================================================

        optimizer.zero_grad(
            set_to_none=True
        )


        # ====================================================
        # FORWARD
        # ====================================================

        logits = model(

            input_ids,

            decoder_input_ids

        )


        # ====================================================
        # LOSS
        # ====================================================

        vocab_size = logits.size(
            -1
        )


        loss = loss_function(

            logits.reshape(
                -1,
                vocab_size
            ),

            decoder_target_ids.reshape(
                -1
            )

        )


        # ====================================================
        # BACKPROPAGATION
        # ====================================================

        loss.backward()


        # ====================================================
        # GRADIENT CLIPPING
        # ====================================================

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            GRAD_CLIP_NORM

        )


        # ====================================================
        # OPTIMIZER
        # ====================================================

        optimizer.step()


        # ====================================================
        # LOSS METRIC
        # ====================================================

        total_loss += (
            loss.item()
        )


        # ====================================================
        # TOKEN ACCURACY
        # ====================================================

        correct_tokens, token_count = (
            calculate_token_accuracy(

                logits,

                decoder_target_ids,

                target_pad_id

            )
        )


        total_correct_tokens += (
            correct_tokens
        )


        total_tokens += (
            token_count
        )


        # ====================================================
        # EXACT SEQUENCE ACCURACY
        # ====================================================

        correct_sequences, sequence_count = (
            calculate_exact_sequence_accuracy(

                logits,

                decoder_target_ids,

                target_pad_id

            )
        )


        total_correct_sequences += (
            correct_sequences
        )


        total_sequences += (
            sequence_count
        )


        # ====================================================
        # OPTIONAL PROGRESS
        # ====================================================

        if (
            PRINT_BATCH_PROGRESS
            and
            (
                batch_index
                %
                PRINT_EVERY_N_BATCHES
                ==
                0
                or
                batch_index
                ==
                len(
                    loader
                )
            )
        ):

            current_token_accuracy = (

                total_correct_tokens

                /
                max(
                    total_tokens,
                    1
                )

            )


            current_exact_accuracy = (

                total_correct_sequences

                /
                max(
                    total_sequences,
                    1
                )

            )


            elapsed = (

                time.time()

                -
                epoch_start

            )


            print(

                f"\rEpoch {epoch:02d} | "
                f"Batch {batch_index:4d}/{len(loader):4d} | "
                f"Loss {loss.item():.4f} | "
                f"Token {current_token_accuracy * 100:6.2f}% | "
                f"Exact {current_exact_accuracy * 100:6.2f}% | "
                f"{format_seconds(elapsed)}",

                end="",

                flush=True

            )


    if PRINT_BATCH_PROGRESS:

        print()


    # ========================================================
    # FINAL TRAIN METRICS
    # ========================================================

    average_loss = (

        total_loss

        /
        max(
            len(
                loader
            ),
            1
        )

    )


    token_accuracy = (

        total_correct_tokens

        /
        max(
            total_tokens,
            1
        )

    )


    exact_accuracy = (

        total_correct_sequences

        /
        max(
            total_sequences,
            1
        )

    )


    return {

        "loss":
            average_loss,

        "token_accuracy":
            token_accuracy,

        "exact_accuracy":
            exact_accuracy,

    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    loss_function,
    device,
    target_pad_id
):

    model.eval()


    total_loss = 0.0


    total_correct_tokens = 0

    total_tokens = 0


    total_correct_sequences = 0

    total_sequences = 0


    # ========================================================
    # VALIDATION LOOP
    # ========================================================

    for batch in loader:

        input_ids = (
            batch[
                "input_ids"
            ]
            .to(
                device,
                non_blocking=True
            )
        )


        decoder_input_ids = (
            batch[
                "decoder_input_ids"
            ]
            .to(
                device,
                non_blocking=True
            )
        )


        decoder_target_ids = (
            batch[
                "decoder_target_ids"
            ]
            .to(
                device,
                non_blocking=True
            )
        )


        # ====================================================
        # FORWARD
        # ====================================================

        logits = model(

            input_ids,

            decoder_input_ids

        )


        # ====================================================
        # LOSS
        # ====================================================

        vocab_size = logits.size(
            -1
        )


        loss = loss_function(

            logits.reshape(
                -1,
                vocab_size
            ),

            decoder_target_ids.reshape(
                -1
            )

        )


        total_loss += (
            loss.item()
        )


        # ====================================================
        # TOKEN ACCURACY
        # ====================================================

        correct_tokens, token_count = (
            calculate_token_accuracy(

                logits,

                decoder_target_ids,

                target_pad_id

            )
        )


        total_correct_tokens += (
            correct_tokens
        )


        total_tokens += (
            token_count
        )


        # ====================================================
        # EXACT SEQUENCE ACCURACY
        # ====================================================

        correct_sequences, sequence_count = (
            calculate_exact_sequence_accuracy(

                logits,

                decoder_target_ids,

                target_pad_id

            )
        )


        total_correct_sequences += (
            correct_sequences
        )


        total_sequences += (
            sequence_count
        )


    # ========================================================
    # VALIDATION RESULTS
    # ========================================================

    average_loss = (

        total_loss

        /
        max(
            len(
                loader
            ),
            1
        )

    )


    token_accuracy = (

        total_correct_tokens

        /
        max(
            total_tokens,
            1
        )

    )


    exact_accuracy = (

        total_correct_sequences

        /
        max(
            total_sequences,
            1
        )

    )


    return {

        "loss":
            average_loss,

        "token_accuracy":
            token_accuracy,

        "exact_accuracy":
            exact_accuracy,

    }


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    input_vocab,
    target_vocab,
    validation_metrics,
    train_metrics,
    batch_size
):

    path.parent.mkdir(

        parents=True,

        exist_ok=True

    )


    checkpoint = {

        # ====================================================
        # VERSION
        # ====================================================

        "batusim_ai_version":
            2,


        "epoch":
            epoch,


        # ====================================================
        # MODEL
        # ====================================================

        "model_state_dict":
            model.state_dict(),


        # ====================================================
        # OPTIMIZER
        # ====================================================

        "optimizer_state_dict":
            optimizer.state_dict(),


        # ====================================================
        # SCHEDULER
        # ====================================================

        "scheduler_state_dict":
            scheduler.state_dict(),


        # ====================================================
        # METRICS
        # ====================================================

        "train_loss":
            train_metrics[
                "loss"
            ],

        "train_token_accuracy":
            train_metrics[
                "token_accuracy"
            ],

        "train_exact_accuracy":
            train_metrics[
                "exact_accuracy"
            ],


        "validation_loss":
            validation_metrics[
                "loss"
            ],

        "validation_token_accuracy":
            validation_metrics[
                "token_accuracy"
            ],

        "validation_exact_accuracy":
            validation_metrics[
                "exact_accuracy"
            ],


        # ====================================================
        # TRAINING CONFIG
        # ====================================================

        "training_config": {

            "batch_size":
                batch_size,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "grad_clip_norm":
                GRAD_CLIP_NORM,

            "random_seed":
                RANDOM_SEED,

        },


        # ====================================================
        # MODEL CONFIG
        # ====================================================

        "model_config": {

            "input_vocab_size":
                len(
                    input_vocab
                ),

            "target_vocab_size":
                len(
                    target_vocab
                ),

            "d_model":
                D_MODEL,

            "nhead":
                NHEAD,

            "num_encoder_layers":
                NUM_ENCODER_LAYERS,

            "num_decoder_layers":
                NUM_DECODER_LAYERS,

            "dim_feedforward":
                DIM_FEEDFORWARD,

            "dropout":
                DROPOUT,

            "input_pad_id":
                input_vocab.token_to_id[
                    PAD_TOKEN
                ],

            "target_pad_id":
                target_vocab.token_to_id[
                    PAD_TOKEN
                ],

        }

    }


    torch.save(

        checkpoint,

        path

    )


# ============================================================
# FORMAT TIME
# ============================================================

def format_seconds(
    seconds
):

    seconds = int(
        seconds
    )


    hours = (

        seconds

        //
        3600

    )


    minutes = (

        (
            seconds
            %
            3600
        )

        //
        60

    )


    remaining_seconds = (

        seconds

        %
        60

    )


    if hours > 0:

        return (

            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{remaining_seconds:02d}"

        )


    return (

        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"

    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_random_seed()


    print(
        "=" * 78
    )


    print(
        "BATUSIM NLP TRANSFORMER TRAINING V2"
    )


    print(
        "=" * 78
    )


    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()


    batch_size = get_batch_size(
        device
    )


    print()

    print(
        "Device:",
        device
    )


    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )


    else:

        print(
            "CUDA kullanılamıyor."
        )


        print(
            "CPU batch size:",
            batch_size
        )


    # ========================================================
    # DATA
    # ========================================================

    print()

    print(
        "Dataset yükleniyor..."
    )


    data = create_dataloaders(

        batch_size=
            batch_size,

        num_workers=
            0

    )


    train_loader = data[
        "train"
    ]


    validation_loader = data[
        "validation"
    ]


    input_vocab = data[
        "input_vocab"
    ]


    target_vocab = data[
        "target_vocab"
    ]


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


    # ========================================================
    # MODEL
    # ========================================================

    model = BatuSimTransformer(

        input_vocab_size=
            len(
                input_vocab
            ),

        target_vocab_size=
            len(
                target_vocab
            ),

        d_model=
            D_MODEL,

        nhead=
            NHEAD,

        num_encoder_layers=
            NUM_ENCODER_LAYERS,

        num_decoder_layers=
            NUM_DECODER_LAYERS,

        dim_feedforward=
            DIM_FEEDFORWARD,

        dropout=
            DROPOUT,

        input_pad_id=
            input_pad_id,

        target_pad_id=
            target_pad_id

    )


    model = model.to(
        device
    )


    # ========================================================
    # MODEL INFO
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


    print(
        "Parameters       :",
        f"{count_parameters(model):,}"
    )


    print(
        "Batch size       :",
        batch_size
    )


    print(
        "Train batches    :",
        len(
            train_loader
        )
    )


    print(
        "Val batches      :",
        len(
            validation_loader
        )
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=
            LEARNING_RATE,

        weight_decay=
            WEIGHT_DECAY

    )


    # ========================================================
    # LR SCHEDULER
    # ========================================================
    #
    # Validation loss birkaç epoch gelişmezse
    # learning rate yarıya düşer.
    # ========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode=
            "min",

        factor=
            LR_SCHEDULER_FACTOR,

        patience=
            LR_SCHEDULER_PATIENCE,

        min_lr=
            MIN_LEARNING_RATE

    )


    # ========================================================
    # LOSS
    # ========================================================

    loss_function = (
        create_loss_function(
            target_pad_id
        )
    )


    # ========================================================
    # TRAIN STATE
    # ========================================================

    best_validation_loss = (
        math.inf
    )


    best_validation_exact = (
        0.0
    )


    epochs_without_improvement = 0


    CHECKPOINT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    # ========================================================
    # TRAINING START
    # ========================================================

    print()

    print(
        "=" * 78
    )


    print(
        "TRAINING START"
    )


    print(
        "=" * 78
    )


    print()

    print(
        "ÖNEMLİ:"
    )


    print(
        (
            "V2 eğitimi eski best/last checkpoint "
            "dosyalarının üzerine yazacak."
        )
    )


    training_start_time = (
        time.time()
    )


    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(

        1,

        NUM_EPOCHS + 1

    ):

        epoch_start_time = (
            time.time()
        )


        # ====================================================
        # CURRENT LR
        # ====================================================

        current_lr = (

            optimizer.param_groups[
                0
            ][
                "lr"
            ]

        )


        print()

        print(
            "-" * 78
        )


        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS}"
        )


        print(
            f"Learning rate: {current_lr:.8f}"
        )


        # ====================================================
        # TRAIN
        # ====================================================

        train_metrics = (
            train_one_epoch(

                model=

                    model,

                loader=

                    train_loader,

                optimizer=

                    optimizer,

                loss_function=

                    loss_function,

                device=

                    device,

                target_pad_id=

                    target_pad_id,

                epoch=

                    epoch

            )
        )


        # ====================================================
        # VALIDATE
        # ====================================================

        validation_metrics = (
            validate(

                model=

                    model,

                loader=

                    validation_loader,

                loss_function=

                    loss_function,

                device=

                    device,

                target_pad_id=

                    target_pad_id

            )
        )


        epoch_seconds = (

            time.time()

            -
            epoch_start_time

        )


        # ====================================================
        # SCHEDULER
        # ====================================================

        scheduler.step(

            validation_metrics[
                "loss"
            ]

        )


        # ====================================================
        # RESULT
        # ====================================================

        print()

        print(
            f"Time        : "
            f"{format_seconds(epoch_seconds)}"
        )


        print()


        print(
            f"Train loss  : "
            f"{train_metrics['loss']:.6f}"
        )


        print(
            f"Train token : "
            f"{train_metrics['token_accuracy'] * 100:.2f}%"
        )


        print(
            f"Train exact : "
            f"{train_metrics['exact_accuracy'] * 100:.2f}%"
        )


        print()


        print(
            f"Val loss    : "
            f"{validation_metrics['loss']:.6f}"
        )


        print(
            f"Val token   : "
            f"{validation_metrics['token_accuracy'] * 100:.2f}%"
        )


        print(
            f"Val exact   : "
            f"{validation_metrics['exact_accuracy'] * 100:.2f}%"
        )


        # ====================================================
        # SAVE LAST
        # ====================================================

        save_checkpoint(

            path=
                LAST_MODEL_PATH,

            model=
                model,

            optimizer=
                optimizer,

            scheduler=
                scheduler,

            epoch=
                epoch,

            input_vocab=
                input_vocab,

            target_vocab=
                target_vocab,

            validation_metrics=
                validation_metrics,

            train_metrics=
                train_metrics,

            batch_size=
                batch_size

        )


        # ====================================================
        # BEST MODEL
        # ====================================================
        #
        # Ana kriter:
        #
        # validation loss
        #
        # Exact accuracy yalnız bilgi amaçlı saklanıyor.
        # ====================================================

        if (
            validation_metrics[
                "loss"
            ]
            <
            best_validation_loss
        ):

            best_validation_loss = (
                validation_metrics[
                    "loss"
                ]
            )


            best_validation_exact = (
                validation_metrics[
                    "exact_accuracy"
                ]
            )


            epochs_without_improvement = 0


            save_checkpoint(

                path=
                    BEST_MODEL_PATH,

                model=
                    model,

                optimizer=
                    optimizer,

                scheduler=
                    scheduler,

                epoch=
                    epoch,

                input_vocab=
                    input_vocab,

                target_vocab=
                    target_vocab,

                validation_metrics=
                    validation_metrics,

                train_metrics=
                    train_metrics,

                batch_size=
                    batch_size

            )


            print()

            print(
                ">>> BEST MODEL SAVED"
            )


            print(
                (
                    ">>> Best Val Exact: "
                    f"{best_validation_exact * 100:.2f}%"
                )
            )


        else:

            epochs_without_improvement += 1


            print()

            print(

                "No validation improvement: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"

            )


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            epochs_without_improvement
            >=
            EARLY_STOPPING_PATIENCE
        ):

            print()

            print(
                "=" * 78
            )


            print(
                "EARLY STOPPING TRIGGERED"
            )


            print(
                "=" * 78
            )


            break


    # ========================================================
    # COMPLETE
    # ========================================================

    total_training_seconds = (

        time.time()

        -
        training_start_time

    )


    print()

    print(
        "=" * 78
    )


    print(
        "TRAINING V2 COMPLETE"
    )


    print(
        "=" * 78
    )


    print()

    print(
        "Total training time:"
    )


    print(
        format_seconds(
            total_training_seconds
        )
    )


    print()

    print(
        "Best validation loss:"
    )


    print(
        f"{best_validation_loss:.6f}"
    )


    print()

    print(
        "Best validation exact:"
    )


    print(
        f"{best_validation_exact * 100:.2f}%"
    )


    print()

    print(
        "Best checkpoint:"
    )


    print(
        BEST_MODEL_PATH
    )


    print()

    print(
        "Last checkpoint:"
    )


    print(
        LAST_MODEL_PATH
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()