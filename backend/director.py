import numpy as np

from backend.kinematics import (
    forward_kinematics,
    inverse_kinematics,
    inverse_kinematics_position
)


# ============================================================
# DIRECTOR DEFAULT SETTINGS
# ============================================================

DEFAULT_LINEAR_STEP_MM = 5.0
DEFAULT_ROTATION_STEP_DEG = 2.0

DEFAULT_REVOLUTE_STEP_DEG = 2.0
DEFAULT_PRISMATIC_STEP_MM = 5.0

POSITION_TOLERANCE_MM = 1.0

# rotation_error() çıktısı doğrudan derece değildir.
# ~0.01 yaklaşık küçük açılarda oldukça sıkı toleranstır.
ORIENTATION_TOLERANCE = 0.02

# Numerical Jacobian singularity kontrolü
SINGULARITY_CONDITION_WARNING = 300.0
SINGULARITY_CONDITION_ERROR = 1000.0

JACOBIAN_EPS_REVOLUTE = 0.1
JACOBIAN_EPS_PRISMATIC = 0.1


# ============================================================
# SMALL HELPERS
# ============================================================

def clamp(
    value,
    low,
    high
):
    return max(
        low,
        min(
            high,
            value
        )
    )


def lerp(
    start,
    end,
    t
):
    return (
        start
        +
        (
            end
            -
            start
        )
        * t
    )


def normalize_axis(
    axis
):
    axis = str(
        axis
    ).upper()

    if axis not in (
        "X",
        "Y",
        "Z"
    ):

        raise ValueError(
            "Axis X, Y veya Z olmalı."
        )

    return axis


# ============================================================
# ROTATION MATRICES
#
# Director V1:
# Rotate X/Y/Z = WORLD eksenlerinde rotation.
#
# Linear Jog da world X/Y/Z kullandığı için
# ilk sürümde bu davranış daha tutarlı.
# ============================================================

def rotation_matrix_x(
    angle_deg
):

    angle = np.deg2rad(
        angle_deg
    )

    c = np.cos(
        angle
    )

    s = np.sin(
        angle
    )

    return np.array([

        [1, 0, 0],

        [0, c, -s],

        [0, s, c]

    ], dtype=float)


def rotation_matrix_y(
    angle_deg
):

    angle = np.deg2rad(
        angle_deg
    )

    c = np.cos(
        angle
    )

    s = np.sin(
        angle
    )

    return np.array([

        [c, 0, s],

        [0, 1, 0],

        [-s, 0, c]

    ], dtype=float)


def rotation_matrix_z(
    angle_deg
):

    angle = np.deg2rad(
        angle_deg
    )

    c = np.cos(
        angle
    )

    s = np.sin(
        angle
    )

    return np.array([

        [c, -s, 0],

        [s, c, 0],

        [0, 0, 1]

    ], dtype=float)


def axis_rotation_matrix(
    axis,
    angle_deg
):

    axis = normalize_axis(
        axis
    )

    if axis == "X":

        return rotation_matrix_x(
            angle_deg
        )

    if axis == "Y":

        return rotation_matrix_y(
            angle_deg
        )

    return rotation_matrix_z(
        angle_deg
    )


# ============================================================
# JOINT UTILITIES
# ============================================================

def joint_names(
    joints
):

    return [

        joint["name"]

        for joint in joints

    ]


def joint_vector_from_values(
    joints,
    values
):

    return np.array([

        float(
            values.get(
                joint["name"],
                joint.get(
                    "q0",
                    0.0
                )
            )
        )

        for joint in joints

    ], dtype=float)


def joint_dict_from_vector(
    joints,
    q_vector
):

    return {

        joint["name"]:
            float(
                q_vector[index]
            )

        for index, joint
        in enumerate(
            joints
        )

    }


def find_joint_index(
    joints,
    joint_name
):

    for index, joint in enumerate(
        joints
    ):

        if (
            joint["name"]
            ==
            joint_name
        ):

            return index

    raise ValueError(
        f"Joint bulunamadı: {joint_name}"
    )


# ============================================================
# JOINT LIMIT CHECK
# ============================================================

def check_joint_limits(
    q_vector,
    joints,
    tolerance=1e-9
):

    violations = []


    for index, joint in enumerate(
        joints
    ):

        value = float(
            q_vector[index]
        )

        lower = float(
            joint["min"]
        )

        upper = float(
            joint["max"]
        )


        if (
            value
            <
            lower - tolerance
        ):

            violations.append({

                "joint":
                    joint["name"],

                "value":
                    value,

                "min":
                    lower,

                "max":
                    upper

            })


        elif (
            value
            >
            upper + tolerance
        ):

            violations.append({

                "joint":
                    joint["name"],

                "value":
                    value,

                "min":
                    lower,

                "max":
                    upper

            })


    return violations


# ============================================================
# NUMERICAL POSITION JACOBIAN
#
# İlk Director sürümünde singularity analizi için
# TCP position Jacobian kullanıyoruz.
#
# Böylece:
#
# 3 DOF Cartesian
# RRR
# V-Robot
# 6 DOF
#
# gibi farklı DOF'lardaki robotlarda full 6xN Jacobian'ın
# düşük DOF robotları yanlışlıkla sürekli singular
# göstermesini engelliyoruz.
# ============================================================

def numerical_position_jacobian(
    q_vector,
    joints,
    fk_function
):

    q_vector = np.asarray(
        q_vector,
        dtype=float
    )


    _, T_base = (
        forward_kinematics(
            q_vector,
            fk_function
        )
    )


    p_base = T_base[
        :3,
        3
    ]


    number_of_joints = len(
        joints
    )


    J = np.zeros(
        (
            3,
            number_of_joints
        ),
        dtype=float
    )


    for index, joint in enumerate(
        joints
    ):

        if (
            joint["type"]
            ==
            "R"
        ):

            epsilon = (
                JACOBIAN_EPS_REVOLUTE
            )

        else:

            epsilon = (
                JACOBIAN_EPS_PRISMATIC
            )


        q_test = (
            q_vector.copy()
        )


        q_test[index] += (
            epsilon
        )


        # Limit dışına taşarsak ters yönde perturb et
        if (
            q_test[index]
            >
            joint["max"]
        ):

            q_test[index] = (
                q_vector[index]
                -
                epsilon
            )

            epsilon = (
                -epsilon
            )


        # Diğer tarafta da hareket edemiyorsak
        # bu sütunu sıfır bırak.
        if (
            q_test[index]
            <
            joint["min"]
        ):

            continue


        _, T_test = (
            forward_kinematics(
                q_test,
                fk_function
            )
        )


        p_test = T_test[
            :3,
            3
        ]


        J[
            :,
            index
        ] = (

            p_test
            -
            p_base

        ) / epsilon


    return J


# ============================================================
# SINGULARITY METRIC
# ============================================================

def singularity_metric(
    q_vector,
    joints,
    fk_function
):

    J = numerical_position_jacobian(

        q_vector,

        joints,

        fk_function

    )


    try:

        singular_values = (
            np.linalg.svd(
                J,
                compute_uv=False
            )
        )

    except np.linalg.LinAlgError:

        return {

            "condition":
                float("inf"),

            "min_singular_value":
                0.0,

            "rank":
                0,

            "warning":
                True,

            "error":
                True

        }


    if (
        len(
            singular_values
        )
        ==
        0
    ):

        condition = float(
            "inf"
        )

        min_sv = 0.0

    else:

        max_sv = float(
            np.max(
                singular_values
            )
        )

        min_sv = float(
            np.min(
                singular_values
            )
        )


        if (
            min_sv
            <
            1e-9
        ):

            condition = float(
                "inf"
            )

        else:

            condition = (
                max_sv
                /
                min_sv
            )


    rank = int(
        np.linalg.matrix_rank(
            J,
            tol=1e-7
        )
    )


    return {

        "condition":
            float(
                condition
            ),

        "min_singular_value":
            float(
                min_sv
            ),

        "rank":
            rank,

        "warning":
            bool(
                condition
                >=
                SINGULARITY_CONDITION_WARNING
            ),

        "error":
            bool(
                condition
                >=
                SINGULARITY_CONDITION_ERROR
            )

    }


# ============================================================
# TRAJECTORY POINT SERIALIZER
# ============================================================

def create_trajectory_point(
    q_vector,
    joints,
    fk_function,
    command_index,
    command_type
):

    frames, T_tool = (
        forward_kinematics(
            q_vector,
            fk_function
        )
    )


    tcp = T_tool[
        :3,
        3
    ]


    singularity = (
        singularity_metric(
            q_vector,
            joints,
            fk_function
        )
    )


    return {

        "q":
            joint_dict_from_vector(
                joints,
                q_vector
            ),

        "q_vector":
            [
                float(value)
                for value
                in q_vector
            ],

        "tcp":
            [
                float(value)
                for value
                in tcp
            ],

        "command_index":
            int(
                command_index
            ),

        "command_type":
            command_type,

        "singularity":
            singularity

    }


# ============================================================
# ERROR RESPONSE
# ============================================================

def director_error(
    error_type,
    command_index,
    message,
    **extra
):

    result = {

        "success":
            False,

        "error_type":
            error_type,

        "command_index":
            int(
                command_index
            ),

        "message":
            message

    }


    result.update(
        extra
    )


    return result


# ============================================================
# LINEAR COMMAND
#
# MOVE_LINEAR
#
# {
#     "type": "MOVE_LINEAR",
#     "axis": "X",
#     "value": 100
# }
#
# value relative movement'tır.
# ============================================================

def plan_linear_command(
    q_start,
    command,
    joints,
    fk_function,
    command_index,
    linear_step_mm
):

    axis = normalize_axis(
        command.get(
            "axis"
        )
    )


    value = float(
        command.get(
            "value",
            0.0
        )
    )


    _, T_start = (
        forward_kinematics(
            q_start,
            fk_function
        )
    )


    start_position = (
        T_start[
            :3,
            3
        ].copy()
    )


    axis_index = {

        "X": 0,

        "Y": 1,

        "Z": 2

    }[
        axis
    ]


    target_position = (
        start_position.copy()
    )


    target_position[
        axis_index
    ] += (
        value
    )


    number_of_steps = max(

        1,

        int(
            np.ceil(

                abs(
                    value
                )

                /
                max(
                    linear_step_mm,
                    1e-6
                )

            )
        )

    )


    q_current = (
        np.asarray(
            q_start,
            dtype=float
        ).copy()
    )


    points = []


    for step_index in range(
        1,
        number_of_steps + 1
    ):

        t = (
            step_index
            /
            number_of_steps
        )


        intermediate_target = (
            start_position.copy()
        )


        intermediate_target[
            axis_index
        ] = lerp(

            start_position[
                axis_index
            ],

            target_position[
                axis_index
            ],

            t

        )


        ik_result = (
            inverse_kinematics_position(

                intermediate_target,

                q_current,

                joints,

                fk_function

            )
        )


        q_solution = np.asarray(
            ik_result["q"],
            dtype=float
        )


        # ----------------------------------------------------
        # REACH ERROR
        # ----------------------------------------------------

        if (
            not ik_result["success"]
            or
            ik_result[
                "position_error"
            ]
            >
            POSITION_TOLERANCE_MM
        ):

            return (

                None,

                director_error(

                    "REACH_ERROR",

                    command_index,

                    (
                        f"Command {command_index + 1}: "
                        f"{axis} yönündeki hedefe ulaşılamadı."
                    ),

                    position_error=float(
                        ik_result[
                            "position_error"
                        ]
                    ),

                    target=[
                        float(value)
                        for value
                        in intermediate_target
                    ]

                )

            )


        # ----------------------------------------------------
        # JOINT LIMIT
        #
        # Solver bounds kullandığı için normalde taşmaz.
        # Yine de doğruluyoruz.
        # ----------------------------------------------------

        violations = (
            check_joint_limits(
                q_solution,
                joints
            )
        )


        if violations:

            return (

                None,

                director_error(

                    "JOINT_LIMIT_ERROR",

                    command_index,

                    (
                        f"Command {command_index + 1}: "
                        f"joint limiti aşıldı."
                    ),

                    violations=
                        violations

                )

            )


        singularity = (
            singularity_metric(
                q_solution,
                joints,
                fk_function
            )
        )


        if (
            singularity[
                "error"
            ]
        ):

            return (

                None,

                director_error(

                    "SINGULARITY_ERROR",

                    command_index,

                    (
                        f"Command {command_index + 1}: "
                        f"singular konfigürasyona girildi."
                    ),

                    singularity=
                        singularity

                )

            )


        q_current = (
            q_solution
        )


        points.append(

            create_trajectory_point(

                q_current,

                joints,

                fk_function,

                command_index,

                "MOVE_LINEAR"

            )

        )


    return (
        {
            "q_end":
                q_current,

            "points":
                points
        },
        None
    )


# ============================================================
# ROTATE TCP COMMAND
#
# {
#     "type": "ROTATE_TCP",
#     "axis": "Z",
#     "value": 30
# }
#
# WORLD X/Y/Z axis rotation.
# ============================================================

def plan_rotation_command(
    q_start,
    command,
    joints,
    fk_function,
    command_index,
    rotation_step_deg
):

    axis = normalize_axis(
        command.get(
            "axis"
        )
    )


    value = float(
        command.get(
            "value",
            0.0
        )
    )


    _, T_start = (
        forward_kinematics(
            q_start,
            fk_function
        )
    )


    T_start = np.asarray(
        T_start,
        dtype=float
    )


    R_start = T_start[
        :3,
        :3
    ].copy()


    position = T_start[
        :3,
        3
    ].copy()


    number_of_steps = max(

        1,

        int(
            np.ceil(

                abs(
                    value
                )

                /
                max(
                    rotation_step_deg,
                    1e-6
                )

            )
        )

    )


    q_current = np.asarray(
        q_start,
        dtype=float
    ).copy()


    points = []


    for step_index in range(
        1,
        number_of_steps + 1
    ):

        t = (
            step_index
            /
            number_of_steps
        )


        angle = (
            value
            *
            t
        )


        R_delta = (
            axis_rotation_matrix(
                axis,
                angle
            )
        )


        # WORLD axis rotation
        R_target = (
            R_delta
            @
            R_start
        )


        T_target = np.eye(
            4,
            dtype=float
        )


        T_target[
            :3,
            :3
        ] = (
            R_target
        )


        T_target[
            :3,
            3
        ] = (
            position
        )


        ik_result = (
            inverse_kinematics(

                T_target,

                q_current,

                joints,

                fk_function

            )
        )


        q_solution = np.asarray(
            ik_result["q"],
            dtype=float
        )


        # ----------------------------------------------------
        # REACH / POSE ERROR
        # ----------------------------------------------------

        if (
            not ik_result["success"]
            or
            ik_result[
                "position_error"
            ]
            >
            POSITION_TOLERANCE_MM
            or
            ik_result[
                "orientation_error"
            ]
            >
            ORIENTATION_TOLERANCE
        ):

            return (

                None,

                director_error(

                    "REACH_ERROR",

                    command_index,

                    (
                        f"Command {command_index + 1}: "
                        f"istenen TCP orientation gerçekleştirilemedi."
                    ),

                    position_error=float(
                        ik_result[
                            "position_error"
                        ]
                    ),

                    orientation_error=float(
                        ik_result[
                            "orientation_error"
                        ]
                    )

                )

            )


        violations = (
            check_joint_limits(
                q_solution,
                joints
            )
        )


        if violations:

            return (

                None,

                director_error(

                    "JOINT_LIMIT_ERROR",

                    command_index,

                    (
                        f"Command {command_index + 1}: "
                        f"joint limiti aşıldı."
                    ),

                    violations=
                        violations

                )

            )


        singularity = (
            singularity_metric(
                q_solution,
                joints,
                fk_function
            )
        )


        if (
            singularity[
                "error"
            ]
        ):

            return (

                None,

                director_error(

                    "SINGULARITY_ERROR",

                    command_index,

                    (
                        f"Command {command_index + 1}: "
                        f"singular konfigürasyon."
                    ),

                    singularity=
                        singularity

                )

            )


        q_current = (
            q_solution
        )


        points.append(

            create_trajectory_point(

                q_current,

                joints,

                fk_function,

                command_index,

                "ROTATE_TCP"

            )

        )


    return (
        {
            "q_end":
                q_current,

            "points":
                points
        },
        None
    )


# ============================================================
# MOVE JOINT COMMAND
#
# V1:
#
# value = RELATIVE joint movement
#
# {
#     "type": "MOVE_JOINT",
#     "joint": "q2",
#     "value": 30
# }
#
# Revolute -> degree
# Prismatic -> mm
# ============================================================

def plan_joint_command(
    q_start,
    command,
    joints,
    fk_function,
    command_index,
    revolute_step_deg,
    prismatic_step_mm
):

    joint_name = str(
        command.get(
            "joint",
            ""
        )
    )


    joint_index = (
        find_joint_index(
            joints,
            joint_name
        )
    )


    joint = joints[
        joint_index
    ]


    value = float(
        command.get(
            "value",
            0.0
        )
    )


    q_start = np.asarray(
        q_start,
        dtype=float
    )


    target_value = (

        q_start[
            joint_index
        ]

        +
        value

    )


    # --------------------------------------------------------
    # LIMIT BEFORE PLANNING
    # --------------------------------------------------------

    if (
        target_value
        <
        joint["min"]
        or
        target_value
        >
        joint["max"]
    ):

        return (

            None,

            director_error(

                "JOINT_LIMIT_ERROR",

                command_index,

                (
                    f"Command {command_index + 1}: "
                    f"{joint_name} hedefi limit dışında."
                ),

                joint=
                    joint_name,

                target=
                    float(
                        target_value
                    ),

                min=
                    float(
                        joint["min"]
                    ),

                max=
                    float(
                        joint["max"]
                    )

            )

        )


    if (
        joint["type"]
        ==
        "R"
    ):

        step_size = (
            revolute_step_deg
        )

    else:

        step_size = (
            prismatic_step_mm
        )


    number_of_steps = max(

        1,

        int(
            np.ceil(

                abs(
                    value
                )

                /
                max(
                    step_size,
                    1e-6
                )

            )
        )

    )


    q_current = (
        q_start.copy()
    )


    points = []


    for step_index in range(
        1,
        number_of_steps + 1
    ):

        t = (
            step_index
            /
            number_of_steps
        )


        q_current = (
            q_start.copy()
        )


        q_current[
            joint_index
        ] = lerp(

            q_start[
                joint_index
            ],

            target_value,

            t

        )


        violations = (
            check_joint_limits(
                q_current,
                joints
            )
        )


        if violations:

            return (

                None,

                director_error(

                    "JOINT_LIMIT_ERROR",

                    command_index,

                    (
                        f"Command {command_index + 1}: "
                        f"joint limiti aşıldı."
                    ),

                    violations=
                        violations

                )

            )


        singularity = (
            singularity_metric(
                q_current,
                joints,
                fk_function
            )
        )


        if (
            singularity[
                "error"
            ]
        ):

            return (

                None,

                director_error(

                    "SINGULARITY_ERROR",

                    command_index,

                    (
                        f"Command {command_index + 1}: "
                        f"singular konfigürasyon."
                    ),

                    singularity=
                        singularity

                )

            )


        points.append(

            create_trajectory_point(

                q_current,

                joints,

                fk_function,

                command_index,

                "MOVE_JOINT"

            )

        )


    return (
        {
            "q_end":
                q_current,

            "points":
                points
        },
        None
    )


# ============================================================
# DIRECTOR PROGRAM PLANNER
#
# Ana fonksiyon.
#
# app.py bunu çağıracak.
# ============================================================

def plan_director_program(
    commands,
    values,
    joints,
    fk_function,
    linear_step_mm=DEFAULT_LINEAR_STEP_MM,
    rotation_step_deg=DEFAULT_ROTATION_STEP_DEG,
    revolute_step_deg=DEFAULT_REVOLUTE_STEP_DEG,
    prismatic_step_mm=DEFAULT_PRISMATIC_STEP_MM
):

    if (
        commands is None
        or
        len(
            commands
        )
        ==
        0
    ):

        return director_error(

            "EMPTY_PROGRAM",

            0,

            "Director programında en az bir komut olmalı."

        )


    q_current = (
        joint_vector_from_values(
            joints,
            values
        )
    )


    initial_violations = (
        check_joint_limits(
            q_current,
            joints
        )
    )


    if initial_violations:

        return director_error(

            "JOINT_LIMIT_ERROR",

            0,

            "Başlangıç joint konfigürasyonu limit dışında.",

            violations=
                initial_violations

        )


    # ========================================================
    # TRAJECTORY
    # ========================================================

    trajectory = []


    # İlk noktayı da gönderiyoruz.
    trajectory.append(

        create_trajectory_point(

            q_current,

            joints,

            fk_function,

            -1,

            "START"

        )

    )


    command_results = []


    # ========================================================
    # PROGRAM
    # ========================================================

    for command_index, command in enumerate(
        commands
    ):

        command_type = str(

            command.get(
                "type",
                ""
            )

        ).upper()


        # ----------------------------------------------------
        # MOVE LINEAR
        # ----------------------------------------------------

        if (
            command_type
            ==
            "MOVE_LINEAR"
        ):

            result, error = (
                plan_linear_command(

                    q_current,

                    command,

                    joints,

                    fk_function,

                    command_index,

                    linear_step_mm

                )
            )


        # ----------------------------------------------------
        # ROTATE TCP
        # ----------------------------------------------------

        elif (
            command_type
            ==
            "ROTATE_TCP"
        ):

            result, error = (
                plan_rotation_command(

                    q_current,

                    command,

                    joints,

                    fk_function,

                    command_index,

                    rotation_step_deg

                )
            )


        # ----------------------------------------------------
        # MOVE JOINT
        # ----------------------------------------------------

        elif (
            command_type
            ==
            "MOVE_JOINT"
        ):

            result, error = (
                plan_joint_command(

                    q_current,

                    command,

                    joints,

                    fk_function,

                    command_index,

                    revolute_step_deg,

                    prismatic_step_mm

                )
            )


        # ----------------------------------------------------
        # UNKNOWN
        # ----------------------------------------------------

        else:

            return director_error(

                "UNKNOWN_COMMAND",

                command_index,

                (
                    f"Bilinmeyen Director komutu: "
                    f"{command_type}"
                )

            )


        # ----------------------------------------------------
        # COMMAND ERROR
        # ----------------------------------------------------

        if (
            error
            is not None
        ):

            error[
                "validated_points"
            ] = len(
                trajectory
            )


            error[
                "trajectory"
            ] = (
                trajectory
            )


            return error


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        q_current = np.asarray(
            result[
                "q_end"
            ],
            dtype=float
        )


        trajectory.extend(
            result[
                "points"
            ]
        )


        command_results.append({

            "command_index":
                command_index,

            "type":
                command_type,

            "point_count":
                len(
                    result[
                        "points"
                    ]
                )

        })


    # ========================================================
    # COMPLETE PROGRAM STATISTICS
    # ========================================================

    singularity_conditions = [

        point[
            "singularity"
        ][
            "condition"
        ]

        for point
        in trajectory

        if np.isfinite(
            point[
                "singularity"
            ][
                "condition"
            ]
        )

    ]


    if singularity_conditions:

        max_condition = float(
            max(
                singularity_conditions
            )
        )

    else:

        max_condition = None


    tcp_path = [

        point[
            "tcp"
        ]

        for point
        in trajectory

    ]


    q_end_dict = (
        joint_dict_from_vector(
            joints,
            q_current
        )
    )


    return {

        "success":
            True,

        "error_type":
            None,

        "message":
            "Director programı başarıyla planlandı.",

        "trajectory":
            trajectory,

        "tcp_path":
            tcp_path,

        "total_points":
            len(
                trajectory
            ),

        "commands":
            command_results,

        "q_final":
            q_end_dict,

        "stats": {

            "max_singularity_condition":
                max_condition,

            "singularity_warning":
                bool(

                    max_condition
                    is not None
                    and
                    max_condition
                    >=
                    SINGULARITY_CONDITION_WARNING

                )

        }

    }