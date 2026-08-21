from __future__ import annotations

from typing import (
    Annotated,
    Literal,
    Union,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


# ============================================================
# BATU-SIM LLM TASK SCHEMA
# ============================================================
#
# Natural Language
#       ↓
#      LLM
#       ↓
#   RobotTaskResponse
#       ↓
#   path_generator.py
#       ↓
#     director.py
#
#
# IMPORTANT
# ---------
#
# LLM:
#
# - IK çözmez.
# - Joint açıları üretmez.
# - DH hesabı yapmaz.
# - Reachability kararı vermez.
# - Trajectory'nin gerçekten mümkün olduğunu iddia etmez.
#
# LLM yalnızca kullanıcının isteğini bu dosyada tanımlanan
# robotik Task IR formatına dönüştürür.
#
# Fiziksel/geometrik doğrulama daha sonra Director tarafından
# yapılır.
# ============================================================


# ============================================================
# COMMON CONFIG
# ============================================================

class StrictModel(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )


# ============================================================
# MODIFIERS
# ============================================================
#
# Modifier:
#
# Bir shape/path çalışırken EŞZAMANLI gerçekleştirilecek
# ikinci hareketi tanımlar.
#
# Örnek:
#
# "XY'de daire çizerken Z'de 50 mm yüksel"
#
# DRAW_SHAPE(CIRCLE)
#       +
# LINEAR_PROGRESS(Z, +50)
#
# ============================================================


class LinearProgressModifier(
    StrictModel
):

    type: Literal[
        "LINEAR_PROGRESS"
    ]

    axis: Literal[
        "X",
        "Y",
        "Z"
    ]

    distance: float = Field(
        description=(
            "Total simultaneous Cartesian displacement "
            "in millimeters over the complete path."
        )
    )


class RotationProgressModifier(
    StrictModel
):

    type: Literal[
        "ROTATION_PROGRESS"
    ]

    axis: Literal[
        "X",
        "Y",
        "Z"
    ]

    angle: float = Field(
        description=(
            "Total simultaneous WORLD-axis TCP rotation "
            "in degrees over the complete path."
        )
    )


PathModifier = Annotated[
    Union[
        LinearProgressModifier,
        RotationProgressModifier,
    ],
    Field(
        discriminator="type"
    ),
]


# ============================================================
# MOVE RELATIVE
# ============================================================
#
# Examples:
#
# "X'te 50 mm git"
#
# "Z'de 20 mm aşağı in"
#
# ============================================================


class MoveRelativeTask(
    StrictModel
):

    action: Literal[
        "MOVE_RELATIVE"
    ]

    axis: Literal[
        "X",
        "Y",
        "Z"
    ]

    distance: float = Field(
        description=(
            "Signed relative Cartesian displacement "
            "in millimeters."
        )
    )

    frame: Literal[
        "WORLD"
    ] = "WORLD"


# ============================================================
# ROTATE RELATIVE
# ============================================================
#
# IMPORTANT:
#
# Current Director convention:
#
# X -> world X rotation
# Y -> world Y rotation
# Z -> world Z rotation
#
# Natural language aliases:
#
# roll  -> X
# pitch -> Y
# yaw   -> Z
#
# ============================================================


class RotateRelativeTask(
    StrictModel
):

    action: Literal[
        "ROTATE_RELATIVE"
    ]

    axis: Literal[
        "X",
        "Y",
        "Z"
    ]

    angle: float = Field(
        description=(
            "Signed relative TCP rotation in degrees."
        )
    )

    frame: Literal[
        "WORLD"
    ] = "WORLD"


# ============================================================
# SQUARE
# ============================================================


class DrawSquareTask(
    StrictModel
):

    action: Literal[
        "DRAW_SHAPE"
    ]

    shape: Literal[
        "SQUARE"
    ]

    plane: Literal[
        "XY",
        "XZ",
        "YZ"
    ]

    size: float = Field(
        gt=0,
        description=(
            "Square side length in millimeters."
        )
    )

    reference: Literal[
        "CURRENT_TCP"
    ] = "CURRENT_TCP"

    orientation_mode: Literal[
        "KEEP"
    ] = "KEEP"

    modifiers: list[
        PathModifier
    ] = Field(
        default_factory=list
    )


# ============================================================
# RECTANGLE
# ============================================================


class DrawRectangleTask(
    StrictModel
):

    action: Literal[
        "DRAW_SHAPE"
    ]

    shape: Literal[
        "RECTANGLE"
    ]

    plane: Literal[
        "XY",
        "XZ",
        "YZ"
    ]

    width: float = Field(
        gt=0,
        description=(
            "Rectangle width in millimeters."
        )
    )

    height: float = Field(
        gt=0,
        description=(
            "Rectangle height in millimeters."
        )
    )

    reference: Literal[
        "CURRENT_TCP"
    ] = "CURRENT_TCP"

    orientation_mode: Literal[
        "KEEP"
    ] = "KEEP"

    modifiers: list[
        PathModifier
    ] = Field(
        default_factory=list
    )


# ============================================================
# TRIANGLE
# ============================================================


class DrawTriangleTask(
    StrictModel
):

    action: Literal[
        "DRAW_SHAPE"
    ]

    shape: Literal[
        "TRIANGLE"
    ]

    plane: Literal[
        "XY",
        "XZ",
        "YZ"
    ]

    size: float = Field(
        gt=0,
        description=(
            "Equilateral triangle side length "
            "in millimeters."
        )
    )

    reference: Literal[
        "CURRENT_TCP"
    ] = "CURRENT_TCP"

    orientation_mode: Literal[
        "KEEP"
    ] = "KEEP"

    modifiers: list[
        PathModifier
    ] = Field(
        default_factory=list
    )


# ============================================================
# CIRCLE
# ============================================================
#
# Circle convention:
#
# radius = mm
#
# path_generator.py circle'ı CURRENT_TCP noktasından
# başlatır ve bir tur sonunda aynı XY/XZ/YZ noktasına döner.
#
# Modifier varsa son Cartesian konum farklı olabilir.
#
# Example:
#
# circle XY + Z rise 50
#
# final:
#
# X = start X
# Y = start Y
# Z = start Z + 50
#
# ============================================================


class DrawCircleTask(
    StrictModel
):

    action: Literal[
        "DRAW_SHAPE"
    ]

    shape: Literal[
        "CIRCLE"
    ]

    plane: Literal[
        "XY",
        "XZ",
        "YZ"
    ]

    radius: float = Field(
        gt=0,
        description=(
            "Circle radius in millimeters."
        )
    )

    reference: Literal[
        "CURRENT_TCP"
    ] = "CURRENT_TCP"

    orientation_mode: Literal[
        "KEEP"
    ] = "KEEP"

    modifiers: list[
        PathModifier
    ] = Field(
        default_factory=list
    )


# ============================================================
# DRAW SHAPE UNION
# ============================================================

DrawShapeTask = Annotated[
    Union[
        DrawSquareTask,
        DrawRectangleTask,
        DrawTriangleTask,
        DrawCircleTask,
    ],
    Field(
        discriminator="shape"
    ),
]


# ============================================================
# RETURN TO START
# ============================================================


class ReturnToStartTask(
    StrictModel
):

    action: Literal[
        "RETURN_TO_START"
    ]

    mode: Literal[
        "TCP",
        "JOINTS"
    ] = "TCP"


# ============================================================
# ROBOT TASK UNION
# ============================================================

RobotTask = Annotated[
    Union[
        MoveRelativeTask,
        RotateRelativeTask,
        DrawShapeTask,
        ReturnToStartTask,
    ],
    Field(
        discriminator="action"
    ),
]


# ============================================================
# READY RESPONSE
# ============================================================
#
# Kullanıcının isteği yeterince açıksa:
#
# {
#     "status": "READY",
#     "steps": [...]
# }
#
# ============================================================


class ReadyTaskResponse(
    StrictModel
):

    status: Literal[
        "READY"
    ]

    steps: list[
        RobotTask
    ] = Field(
        min_length=1
    )

    message: str = Field(
        default=(
            "Task understood."
        )
    )


# ============================================================
# NEEDS CLARIFICATION
# ============================================================
#
# Example:
#
# User:
#
# "XY düzleminde kare çiz."
#
# Missing:
#
# size
#
# LLM MUST NOT invent 50 mm / 100 mm etc.
#
# Instead:
#
# {
#   "status": "NEEDS_CLARIFICATION",
#   "question": "Karenin kenar uzunluğu kaç mm olsun?"
# }
#
# ============================================================


class NeedsClarificationResponse(
    StrictModel
):

    status: Literal[
        "NEEDS_CLARIFICATION"
    ]

    question: str = Field(
        min_length=1
    )

    missing_fields: list[
        str
    ] = Field(
        default_factory=list
    )


# ============================================================
# UNSUPPORTED
# ============================================================
#
# Kullanıcı sistemin henüz yapamadığı bir şey isterse:
#
# Example:
#
# "Kameradan kırmızı parçayı bul ve tut."
#
# Şu an:
#
# vision yok
# gripper command yok
#
# Dolayısıyla LLM bunu MOVE_RELATIVE gibi göstermemeli.
#
# ============================================================


class UnsupportedTaskResponse(
    StrictModel
):

    status: Literal[
        "UNSUPPORTED"
    ]

    reason: str = Field(
        min_length=1
    )


# ============================================================
# FINAL LLM RESPONSE TYPE
# ============================================================

RobotTaskResponse = Annotated[
    Union[
        ReadyTaskResponse,
        NeedsClarificationResponse,
        UnsupportedTaskResponse,
    ],
    Field(
        discriminator="status"
    ),
]


# ============================================================
# CONVERT READY RESPONSE → PATH GENERATOR IR
# ============================================================
#
# LLM schema:
#
# {
#   status: READY,
#   steps: [...]
# }
#
# path_generator.py currently expects:
#
# {
#   intent: TASK_SEQUENCE,
#   steps: [...]
# }
#
# veya single task.
#
# Bu function iki sistemi birbirinden ayırır.
#
# Böylece ileride LLM response formatını değiştirsek bile
# path_generator.py'yi değiştirmek zorunda kalmayız.
#
# ============================================================


def ready_response_to_task_ir(
    response: ReadyTaskResponse
):

    steps = [

        step.model_dump()

        for step
        in response.steps

    ]


    if len(
        steps
    ) == 1:

        single = steps[
            0
        ]


        return {

            "intent":
                single[
                    "action"
                ],

            **single

        }


    return {

        "intent":
            "TASK_SEQUENCE",

        "steps":
            steps

    }


# ============================================================
# RESPONSE SERIALIZER
# ============================================================

def serialize_task_response(
    response
):

    return response.model_dump()


# ============================================================
# JSON SCHEMA
# ============================================================
#
# llm_interpreter.py gerektiğinde bu function üzerinden
# schema'yı okuyabilir.
#
# ============================================================

def get_task_response_json_schema():

    from pydantic import TypeAdapter


    adapter = TypeAdapter(
        RobotTaskResponse
    )


    return adapter.json_schema()


# ============================================================
# VALIDATE RAW LLM OUTPUT
# ============================================================
#
# LLM'den dictionary geldiğinde Pydantic validation.
#
# Schema dışındaki alanlar:
#
# extra="forbid"
#
# nedeniyle reddedilir.
#
# ============================================================

def validate_task_response(
    data
):

    from pydantic import TypeAdapter


    adapter = TypeAdapter(
        RobotTaskResponse
    )


    return adapter.validate_python(
        data
    )