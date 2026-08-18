from pathlib import Path
import json
import numpy as np

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

# ============================================================
# ROBOT FK CACHE
# ============================================================

FK_CACHE = {}

# ============================================================
# ROBOT MODEL
# ============================================================

from backend.robot_model import (
    SYMBOLS,
    parse_value,
    robot_to_dict,
    create_default_values,
    symbolic_forward_kinematics
)


# ============================================================
# KINEMATICS
# ============================================================

from backend.kinematics import (
    prepare_fk,
    forward_kinematics
)


# ============================================================
# JOG CONTROL
# ============================================================

from backend.jog_control import (
    joint_jog,
    linear_jog
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BASE_DIR.parent

FRONTEND_DIR = PROJECT_DIR / "frontend"

PRESETS_DIR = PROJECT_DIR / "presets"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Generic DH Robot Simulator",
    version="0.3.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# REQUEST MODELS
# ============================================================

class DHRow(BaseModel):

    theta: Any = 0
    d: Any = 0
    a: Any = 0
    alpha: Any = 0

    type: str = "FIXED"

    min: float | None = None
    max: float | None = None


class BuildRobotRequest(BaseModel):

    dh_table: list[DHRow]

    values: dict[str, float | None] = Field(
        default_factory=dict
    )


class JointJogRequest(BaseModel):

    dh_table: list[DHRow]

    values: dict[str, float | None]

    joint_name: str

    direction: int

    revolute_step: float = 1.0
    prismatic_step: float = 5.0


class LinearJogRequest(BaseModel):

    dh_table: list[DHRow]

    values: dict[str, float | None]

    axis: str

    direction: int

    step: float = 10.0


# ============================================================
# HELPERS
# ============================================================

def clean_values(values):

    return {
        name: value
        for name, value in values.items()
        if value is not None
    }


def get_parameters(values):

    return {
        name: value
        for name, value in values.items()
        if not name.startswith("q")
    }


def frames_to_json(frames):

    return [
        {
            "matrix": T.tolist(),

            "position":
                T[:3, 3].tolist(),

            "rotation":
                T[:3, :3].tolist()
        }

        for T in frames
    ]

# ============================================================
# GET CACHED FK
# ============================================================

def get_cached_fk(
    dh_table,
    joints,
    parameters
):

    # --------------------------------------------------------
    # Robot geometrisini unique bir key'e çevir
    # --------------------------------------------------------

    cache_data = {
        "dh": dh_table,
        "parameters": parameters
    }

    cache_key = json.dumps(
        cache_data,
        sort_keys=True,
        default=str
    )


    # --------------------------------------------------------
    # Daha önce hazırlandıysa direkt kullan
    # --------------------------------------------------------

    if cache_key in FK_CACHE:

        return FK_CACHE[
            cache_key
        ]


    # --------------------------------------------------------
    # İlk kez kullanılıyorsa hazırla
    # --------------------------------------------------------

    print(
        "FK CACHE MISS -> preparing robot..."
    )

    FK = prepare_fk(
        dh_table,
        joints,
        parameters
    )


    FK_CACHE[
        cache_key
    ] = FK


    return FK

# ============================================================
# CREATE GENERIC JOINT LIST
# ============================================================

def create_joints(
    dh_table,
    values
):

    joints = []


    for row in dh_table:

        joint_type = row.get(
            "type",
            "FIXED"
        )


        if joint_type not in (
            "R",
            "P"
        ):

            continue


        # ====================================================
        # FIND q SYMBOL
        # ====================================================

        symbols = set()


        for field in (
            "theta",
            "d",
            "a",
            "alpha"
        ):

            expression = parse_value(
                row.get(
                    field,
                    0
                )
            )

            symbols.update(
                expression.free_symbols
            )


        q_symbols = [
            symbol
            for symbol in symbols
            if str(symbol).startswith("q")
        ]


        if len(q_symbols) != 1:

            raise ValueError(
                "Her R/P DH satırında tam olarak "
                "bir adet q değişkeni bulunmalıdır."
            )


        symbol = q_symbols[0]

        name = str(symbol)


        # ====================================================
        # LIMITS
        # ====================================================

        if joint_type == "R":

            default_min = -180.0
            default_max = 180.0

        else:

            default_min = -1000.0
            default_max = 1000.0


        row_min = row.get("min")

        row_max = row.get("max")


        q_min = (
            float(row_min)
            if row_min is not None
            else default_min
        )


        q_max = (
            float(row_max)
            if row_max is not None
            else default_max
        )


        q0 = float(
            values.get(
                name,
                0.0
            )
        )


        joints.append({
            "symbol":
                SYMBOLS[name],

            "name":
                name,

            "type":
                joint_type,

            "q0":
                q0,

            "min":
                q_min,

            "max":
                q_max
        })


    # q1 q2 q3...
    joints.sort(
        key=lambda joint:
            int(
                joint["name"][1:]
            )
    )


    return joints


# ============================================================
# CREATE q VECTOR
# ============================================================

def create_q_vector(
    joints,
    values
):

    return np.array(
        [
            values.get(
                joint["name"],
                joint["q0"]
            )

            for joint in joints
        ],

        dtype=float
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    index_file = (
        FRONTEND_DIR
        / "index.html"
    )


    if index_file.exists():

        return FileResponse(
            index_file
        )


    return {
        "message":
            "DH Robot Backend çalışıyor."
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok"
    }


# ============================================================
# DEFAULT VALUES
# ============================================================

@app.post("/api/default-values")
def default_values(
    request: BuildRobotRequest
):

    try:

        dh_table = [
            row.model_dump()
            for row in request.dh_table
        ]


        defaults = (
            create_default_values(
                dh_table
            )
        )


        return {
            "values": defaults
        }


    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


# ============================================================
# BUILD / FK
# ============================================================

@app.post("/api/build")
def build_robot(
    request: BuildRobotRequest
):

    try:

        dh_table = [
            row.model_dump()
            for row in request.dh_table
        ]


        result = robot_to_dict(
            dh_table,
            request.values
        )


        return {
            "success": True,
            **result
        }


    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


# ============================================================
# SYMBOLIC FK
# ============================================================

@app.post("/api/symbolic-fk")
def symbolic_fk(
    request: BuildRobotRequest
):

    try:

        dh_table = [
            row.model_dump()
            for row in request.dh_table
        ]


        frames = (
            symbolic_forward_kinematics(
                dh_table
            )
        )


        T_tool = frames[-1]


        return {
            "success": True,
            "matrix": str(T_tool)
        }


    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


# ============================================================
# JOINT JOG
# ============================================================

@app.post("/api/jog/joint")
def api_joint_jog(
    request: JointJogRequest
):

    try:

        dh_table = [
            row.model_dump()
            for row in request.dh_table
        ]


        values = clean_values(
            request.values
        )


        joints = create_joints(
            dh_table,
            values
        )


        if not joints:

            raise ValueError(
                "Robot hareketli joint içermiyor."
            )


        parameters = get_parameters(
            values
        )


        FK = get_cached_fk(
            dh_table,
            joints,
            parameters
        )


        q_current = create_q_vector(
            joints,
            values
        )


        joint_index = None


        for i, joint in enumerate(
            joints
        ):

            if (
                joint["name"]
                ==
                request.joint_name
            ):

                joint_index = i
                break


        if joint_index is None:

            raise ValueError(
                f"Joint bulunamadı: "
                f"{request.joint_name}"
            )


        q_new = joint_jog(
            q_current,
            joint_index,
            request.direction,
            joints,
            request.revolute_step,
            request.prismatic_step
        )


        frames, T_tool = (
            forward_kinematics(
                q_new,
                FK
            )
        )


        return {
            "success": True,

            "q": {
                joint["name"]:
                    float(value)

                for joint, value
                in zip(
                    joints,
                    q_new
                )
            },

            "frames":
                frames_to_json(
                    frames
                ),

            "tcp":
                T_tool[
                    :3,
                    3
                ].tolist()
        }


    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


# ============================================================
# LINEAR JOG
# ============================================================

@app.post("/api/jog/linear")
def api_linear_jog(
    request: LinearJogRequest
):

    try:

        axis = (
            request.axis
            .upper()
        )


        if axis not in (
            "X",
            "Y",
            "Z"
        ):

            raise ValueError(
                "Axis X, Y veya Z olmalıdır."
            )


        if request.direction not in (
            -1,
            1
        ):

            raise ValueError(
                "Direction yalnızca -1 veya +1 olabilir."
            )


        dh_table = [
            row.model_dump()
            for row in request.dh_table
        ]


        values = clean_values(
            request.values
        )


        joints = create_joints(
            dh_table,
            values
        )


        if not joints:

            raise ValueError(
                "Robot hareketli joint içermiyor."
            )


        parameters = get_parameters(
            values
        )


        FK = get_cached_fk(
            dh_table,
            joints,
            parameters
        )


        q_current = create_q_vector(
            joints,
            values
        )


        result = linear_jog(
            q_current,
            axis,
            request.direction,
            request.step,
            joints,
            FK
        )


        q_new = result["q"]


        position_error = float(
            result[
                "position_error"
            ]
        )


        # Hedef gerçekten erişilebilir mi?
        if position_error > 1.0:

            return {
                "success": False,

                "message":
                    "Hedefe yeterince "
                    "yaklaşılamadı.",

                "position_error":
                    position_error,

                "q": {
                    joint["name"]:
                        float(value)

                    for joint, value
                    in zip(
                        joints,
                        q_new
                    )
                }
            }


        return {
            "success":
                bool(
                    result["success"]
                ),

            "position_error":
                position_error,

            "q": {
                joint["name"]:
                    float(value)

                for joint, value
                in zip(
                    joints,
                    q_new
                )
            },

            "frames":
                frames_to_json(
                    result["frames"]
                ),

            "tcp":
                result[
                    "T_solution"
                ][:3, 3]
                .tolist(),

            "target":
                result[
                    "target_position"
                ].tolist(),

            "axis":
                result["axis"],

            "distance":
                float(
                    result["distance"]
                )
        }


    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


# ============================================================
# PRESET LIST
# ============================================================

@app.get("/api/presets")
def list_presets():

    PRESETS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    robots = []


    for file in sorted(
        PRESETS_DIR.glob(
            "*.json"
        )
    ):

        try:

            with open(
                file,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)


            robots.append({
                "id":
                    file.stem,

                "name":
                    data.get(
                        "name",
                        file.stem
                    ),

                "dof":
                    data.get(
                        "dof"
                    )
            })


        except Exception:

            continue


    return {
        "presets": robots
    }


# ============================================================
# LOAD PRESET
# ============================================================

@app.get("/api/presets/{preset_id}")
def load_preset(
    preset_id: str
):

    safe_name = (
        Path(
            preset_id
        ).name
    )


    file = (
        PRESETS_DIR
        / f"{safe_name}.json"
    )


    if not file.exists():

        raise HTTPException(
            status_code=404,
            detail="Robot preset bulunamadı."
        )


    try:

        with open(
            file,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# STATIC FRONTEND
# ============================================================

if FRONTEND_DIR.exists():

    app.mount(
        "/static",

        StaticFiles(
            directory=FRONTEND_DIR
        ),

        name="static"
    )