import math

import torch
import torch.nn as nn


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(
    nn.Module
):

    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_len=512
    ):

        super().__init__()


        self.dropout = nn.Dropout(
            dropout
        )


        pe = torch.zeros(
            max_len,
            d_model
        )


        position = torch.arange(
            0,
            max_len,
            dtype=torch.float32
        ).unsqueeze(
            1
        )


        div_term = torch.exp(

            torch.arange(
                0,
                d_model,
                2
            ).float()

            *

            (
                -math.log(
                    10000.0
                )
                /
                d_model
            )

        )


        pe[
            :,
            0::2
        ] = torch.sin(

            position
            *
            div_term

        )


        pe[
            :,
            1::2
        ] = torch.cos(

            position
            *
            div_term

        )


        pe = pe.unsqueeze(
            0
        )


        self.register_buffer(
            "pe",
            pe
        )


    def forward(
        self,
        x
    ):

        # x:
        # [batch, seq, d_model]

        x = (

            x

            +

            self.pe[
                :,
                :x.size(1),
                :
            ]

        )


        return self.dropout(
            x
        )


# ============================================================
# BATU-SIM TRANSFORMER
# ============================================================

class BatuSimTransformer(
    nn.Module
):

    def __init__(
        self,
        input_vocab_size,
        target_vocab_size,

        d_model=128,
        nhead=4,

        num_encoder_layers=3,
        num_decoder_layers=3,

        dim_feedforward=256,

        dropout=0.1,

        input_pad_id=0,
        target_pad_id=0
    ):

        super().__init__()


        self.d_model = (
            d_model
        )


        self.input_pad_id = (
            input_pad_id
        )


        self.target_pad_id = (
            target_pad_id
        )


        # ====================================================
        # EMBEDDINGS
        # ====================================================

        self.input_embedding = (
            nn.Embedding(

                input_vocab_size,

                d_model,

                padding_idx=
                    input_pad_id

            )
        )


        self.target_embedding = (
            nn.Embedding(

                target_vocab_size,

                d_model,

                padding_idx=
                    target_pad_id

            )
        )


        # ====================================================
        # POSITIONAL ENCODING
        # ====================================================

        self.input_position = (
            PositionalEncoding(

                d_model,

                dropout

            )
        )


        self.target_position = (
            PositionalEncoding(

                d_model,

                dropout

            )
        )


        # ====================================================
        # TRANSFORMER
        # ====================================================

        self.transformer = (
            nn.Transformer(

                d_model=
                    d_model,

                nhead=
                    nhead,

                num_encoder_layers=
                    num_encoder_layers,

                num_decoder_layers=
                    num_decoder_layers,

                dim_feedforward=
                    dim_feedforward,

                dropout=
                    dropout,

                batch_first=
                    True,

                norm_first=
                    True

            )
        )


        # ====================================================
        # OUTPUT HEAD
        # ====================================================

        self.output_layer = (
            nn.Linear(

                d_model,

                target_vocab_size

            )
        )


        # ====================================================
        # INITIALIZATION
        # ====================================================

        self._reset_parameters()


    # ========================================================
    # PARAM INIT
    # ========================================================

    def _reset_parameters(
        self
    ):

        for parameter in self.parameters():

            if (
                parameter.dim()
                >
                1
            ):

                nn.init.xavier_uniform_(
                    parameter
                )


    # ========================================================
    # CAUSAL MASK
    # ========================================================

    def generate_causal_mask(
        self,
        target_length,
        device
    ):

        mask = torch.triu(

            torch.ones(

                target_length,

                target_length,

                device=
                    device,

                dtype=
                    torch.bool

            ),

            diagonal=1

        )


        return mask


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        input_ids,
        decoder_input_ids
    ):

        # ====================================================
        # PADDING MASKS
        # ====================================================

        src_key_padding_mask = (

            input_ids
            ==
            self.input_pad_id

        )


        tgt_key_padding_mask = (

            decoder_input_ids
            ==
            self.target_pad_id

        )


        # ====================================================
        # EMBEDDINGS
        # ====================================================

        src = self.input_embedding(
            input_ids
        )


        src = (

            src

            *
            math.sqrt(
                self.d_model
            )

        )


        src = self.input_position(
            src
        )


        tgt = self.target_embedding(
            decoder_input_ids
        )


        tgt = (

            tgt

            *
            math.sqrt(
                self.d_model
            )

        )


        tgt = self.target_position(
            tgt
        )


        # ====================================================
        # CAUSAL MASK
        #
        # Decoder gelecekteki tokenları göremez.
        # ====================================================

        tgt_mask = (
            self.generate_causal_mask(

                decoder_input_ids.size(
                    1
                ),

                decoder_input_ids.device

            )
        )


        # ====================================================
        # TRANSFORMER
        # ====================================================

        output = self.transformer(

            src=
                src,

            tgt=
                tgt,

            tgt_mask=
                tgt_mask,

            src_key_padding_mask=
                src_key_padding_mask,

            tgt_key_padding_mask=
                tgt_key_padding_mask,

            memory_key_padding_mask=
                src_key_padding_mask

        )


        # ====================================================
        # VOCAB LOGITS
        # ====================================================

        logits = self.output_layer(
            output
        )


        return logits


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(
    model
):

    return sum(

        parameter.numel()

        for parameter
        in model.parameters()

        if parameter.requires_grad

    )


# ============================================================
# TEST MODEL
# ============================================================

def main():

    from backend.ai.dataset import (
        create_dataloaders
    )


    from backend.ai.tokenizer import (
        PAD_TOKEN
    )


    print(
        "=" * 70
    )

    print(
        "BATUSIM TRANSFORMER MODEL TEST"
    )

    print(
        "=" * 70
    )


    data = create_dataloaders(
        batch_size=8
    )


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
    # CREATE MODEL
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
            128,

        nhead=
            4,

        num_encoder_layers=
            3,

        num_decoder_layers=
            3,

        dim_feedforward=
            256,

        dropout=
            0.1,

        input_pad_id=
            input_pad_id,

        target_pad_id=
            target_pad_id

    )


    print()

    print(
        "Input vocab:",
        len(
            input_vocab
        )
    )


    print(
        "Target vocab:",
        len(
            target_vocab
        )
    )


    print()

    print(
        "Trainable parameters:",
        f"{count_parameters(model):,}"
    )


    # ========================================================
    # GET ONE BATCH
    # ========================================================

    batch = next(

        iter(
            data[
                "train"
            ]
        )

    )


    input_ids = batch[
        "input_ids"
    ]


    decoder_input_ids = batch[
        "decoder_input_ids"
    ]


    print()

    print(
        "Input shape:"
    )

    print(
        input_ids.shape
    )


    print()

    print(
        "Decoder input shape:"
    )

    print(
        decoder_input_ids.shape
    )


    # ========================================================
    # FORWARD PASS
    # ========================================================

    logits = model(

        input_ids,

        decoder_input_ids

    )


    print()

    print(
        "Output logits shape:"
    )

    print(
        logits.shape
    )


    # ========================================================
    # EXPECTED:
    #
    # [batch, target_seq_len, target_vocab_size]
    # ========================================================

    expected_shape = (

        input_ids.size(
            0
        ),

        decoder_input_ids.size(
            1
        ),

        len(
            target_vocab
        )

    )


    print()

    print(
        "Expected shape:"
    )

    print(
        expected_shape
    )


    print()

    if (
        tuple(
            logits.shape
        )
        ==
        expected_shape
    ):

        print(
            "MODEL FORWARD PASS: OK"
        )

    else:

        print(
            "MODEL FORWARD PASS: ERROR"
        )


if __name__ == "__main__":

    main()