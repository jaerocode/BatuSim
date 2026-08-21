import json
import os

from openai import OpenAI

from backend.task_schema import (
    get_task_response_json_schema,
    validate_task_response,
    ready_response_to_task_ir,
    ReadyTaskResponse,
    NeedsClarificationResponse,
    UnsupportedTaskResponse,
)


# ============================================================
# LLM INTERPRETER
#
# Natural Language
#       ↓
# OpenAI LLM
#       ↓
# STRICT RobotTaskResponse
#       ↓
# Task IR
#       ↓
# path_generator.py
#       ↓
# director.py
#
#
# IMPORTANT:
#
# LLM:
#
# - IK çözmez.
# - q1/q2/... üretmez.
# - Reachability hesaplamaz.
# - DH çözmez.
# - Trajectory waypoint hesaplamaz.
#
# Sadece kullanıcı niyetini structured robot task'a çevirir.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_LLM_MODEL = "gpt-5.4-mini"


def get_llm_model():

    return os.getenv(
        "BATUSIM_LLM_MODEL",
        DEFAULT_LLM_MODEL
    )


# ============================================================
# ERROR
# ============================================================

class LLMInterpreterError(
    RuntimeError
):
    pass


# ============================================================
# OPENAI CLIENT
#
# OPENAI_API_KEY environment variable otomatik kullanılır.
# ============================================================

def create_openai_client():

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )


    if not api_key:

        raise LLMInterpreterError(

            (
                "OPENAI_API_KEY bulunamadı. "
                "Environment variable olarak eklenmeli."
            )

        )


    return OpenAI(
        api_key=api_key
    )


# ============================================================
# SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTIONS = """
You are the natural-language robot task interpreter for BatuSim.

Your ONLY responsibility is to convert the user's natural-language robot
instruction into the supplied structured RobotTaskResponse schema.

You do NOT control the robot directly.

You do NOT calculate inverse kinematics.

You do NOT calculate joint angles.

You do NOT decide whether a requested trajectory is physically reachable.

You do NOT modify DH parameters.

You do NOT invent missing geometric values.


============================================================
COORDINATE CONVENTION
============================================================

BatuSim currently uses WORLD Cartesian coordinates.

Linear directions:

X = world X
Y = world Y
Z = world Z

Rotational aliases:

ROLL  = world X rotation
PITCH = world Y rotation
YAW   = world Z rotation

If the user explicitly says:

"Y ekseninde 90 derece dön"

this means:

ROTATE_RELATIVE
axis = Y
angle = 90

If the user says:

"90 derece pitch yap"

this means exactly the same thing.

If the user says:

"90 derece yaw yap"

use:

axis = Z


============================================================
UNITS
============================================================

All distance values in the structured output MUST be millimeters.

Convert units when needed.

Examples:

50 mm
→ 50

5 cm
→ 50

0.2 m
→ 200


All rotation angles MUST be degrees.


============================================================
SIGNED MOTION
============================================================

Preserve explicit signs.

Examples:

"X ekseninde -50 mm git"
→ distance = -50

"Z ekseninde 50 mm yukarı çık"
→ distance = +50

"Z ekseninde 50 mm aşağı in"
→ distance = -50


Do not arbitrarily infer left/right unless the corresponding world axis
direction is clearly stated by the user.


============================================================
TASK SEQUENCES
============================================================

A user may request multiple sequential operations.

Example:

"X'te 50 mm geri git, sonra XZ düzleminde 20 mm kare çiz
ve başlangıç konumuna geri dön."

Return the steps IN THE SAME ORDER.

Do not merge sequential operations unless they are explicitly simultaneous.


============================================================
SIMULTANEOUS MOTION
============================================================

Words such as:

"çizerken"
"aynı anda"
"eş zamanlı"
"while drawing"
"simultaneously"

mean that the second motion should usually become a modifier of the path.

Example:

"XY düzleminde 40 mm yarıçaplı daire çizerken
Z'de 50 mm yukarı çık."

Interpret as ONE DRAW_SHAPE step:

shape = CIRCLE
plane = XY
radius = 40

modifier:

type = LINEAR_PROGRESS
axis = Z
distance = 50


Do NOT turn this into:

1. draw circle
2. move Z

because those would be sequential rather than simultaneous.


Another example:

"XY düzleminde kare çizerken 90 derece yaw yap."

Use:

ROTATION_PROGRESS
axis = Z
angle = 90


============================================================
SHAPES
============================================================

Supported shapes:

SQUARE
RECTANGLE
TRIANGLE
CIRCLE

For SQUARE:
size is required.

For RECTANGLE:
width and height are required.

For TRIANGLE:
size is required.

For CIRCLE:
radius is required.

Current reference is CURRENT_TCP.

Current orientation mode is KEEP.


============================================================
RETURN TO START
============================================================

"başlangıç konumuna dön"
"başladığın noktaya dön"

normally means:

RETURN_TO_START
mode = TCP


If the user explicitly asks to return to the original joint configuration,
robot pose, or joint pose:

RETURN_TO_START
mode = JOINTS


============================================================
MISSING INFORMATION
============================================================

NEVER invent a required dimension.

Example:

"XY düzleminde bir kare çiz."

The square size is missing.

Return:

status = NEEDS_CLARIFICATION

Ask a concise question such as:

"Karenin kenar uzunluğu kaç mm olsun?"

missing_fields should include:

"size"


Another example:

"50 mm git."

Axis is missing.

Ask which axis.


============================================================
UNSUPPORTED REQUESTS
============================================================

Return UNSUPPORTED if the request requires capabilities outside the current
task schema.

Examples:

- camera recognition
- gripping an object
- force control
- obstacle detection from camera
- welding
- changing robot hardware
- unknown unsupported shape when it cannot be represented by the schema

Do not fake unsupported actions using existing commands.


============================================================
IMPORTANT SAFETY / RELIABILITY RULE
============================================================

READY means only:

"The user request was understood and represented successfully."

READY does NOT mean:

"The robot can physically perform it."

Reachability, joint limits, singularity, and IK validation happen later in
BatuSim's deterministic Director planner.


============================================================
OUTPUT STYLE
============================================================

Return only data that conforms to the supplied schema.

Keep message/question/reason fields concise.

Do not include explanations outside the structured response.
"""


# ============================================================
# BUILD USER INPUT
# ============================================================

def build_user_prompt(
    text
):

    return (
        "Convert the following user instruction into "
        "a BatuSim robot task:\n\n"
        f"{text}"
    )


# ============================================================
# CALL OPENAI
# ============================================================

def call_llm(
    text
):

    if (
        text is None
        or
        not str(text).strip()
    ):

        raise LLMInterpreterError(
            "Komut metni boş olamaz."
        )


    client = (
        create_openai_client()
    )


    schema = (
        get_task_response_json_schema()
    )


    try:

        response = client.responses.create(

            model=
                get_llm_model(),

            instructions=
                SYSTEM_INSTRUCTIONS,

            input=
                build_user_prompt(
                    text
                ),

            text={

                "format": {

                    "type":
                        "json_schema",

                    "name":
                        "batusim_robot_task",

                    "schema":
                        schema,

                    "strict":
                        True

                }

            }

        )


    except Exception as error:

        raise LLMInterpreterError(

            (
                "LLM isteği başarısız: "
                f"{str(error)}"
            )

        ) from error


    # ========================================================
    # OUTPUT TEXT
    # ========================================================

    output_text = getattr(
        response,
        "output_text",
        None
    )


    if (
        output_text is None
        or
        not str(
            output_text
        ).strip()
    ):

        raise LLMInterpreterError(

            "LLM boş structured output döndürdü."

        )


    # ========================================================
    # JSON PARSE
    # ========================================================

    try:

        raw_data = json.loads(
            output_text
        )


    except json.JSONDecodeError as error:

        raise LLMInterpreterError(

            (
                "LLM çıktısı JSON olarak okunamadı: "
                f"{str(error)}"
            )

        ) from error


    # ========================================================
    # SECOND VALIDATION
    #
    # Structured Outputs zaten schema'yı zorlar.
    #
    # Buna rağmen backend sınırında Pydantic ile tekrar
    # validation yapıyoruz.
    # ========================================================

    try:

        validated = (
            validate_task_response(
                raw_data
            )
        )


    except Exception as error:

        raise LLMInterpreterError(

            (
                "LLM çıktısı BatuSim Task Schema "
                f"validation'dan geçemedi: {str(error)}"
            )

        ) from error


    return validated


# ============================================================
# MAIN INTERPRETER
# ============================================================

def interpret_task_with_llm(
    text
):

    response = (
        call_llm(
            text
        )
    )


    # ========================================================
    # READY
    # ========================================================

    if isinstance(
        response,
        ReadyTaskResponse
    ):

        task_ir = (
            ready_response_to_task_ir(
                response
            )
        )


        return {

            "success":
                True,

            "status":
                "READY",

            "input":
                text,

            "model":
                get_llm_model(),

            "message":
                response.message,

            "task_ir":
                task_ir,

            "llm_response":
                response.model_dump()

        }


    # ========================================================
    # NEEDS CLARIFICATION
    # ========================================================

    if isinstance(
        response,
        NeedsClarificationResponse
    ):

        return {

            "success":
                False,

            "status":
                "NEEDS_CLARIFICATION",

            "input":
                text,

            "model":
                get_llm_model(),

            "question":
                response.question,

            "missing_fields":
                response.missing_fields,

            "task_ir":
                None,

            "llm_response":
                response.model_dump()

        }


    # ========================================================
    # UNSUPPORTED
    # ========================================================

    if isinstance(
        response,
        UnsupportedTaskResponse
    ):

        return {

            "success":
                False,

            "status":
                "UNSUPPORTED",

            "input":
                text,

            "model":
                get_llm_model(),

            "reason":
                response.reason,

            "task_ir":
                None,

            "llm_response":
                response.model_dump()

        }


    # ========================================================
    # SHOULD NEVER HAPPEN
    # ========================================================

    raise LLMInterpreterError(

        "Bilinmeyen LLM response tipi."

    )