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

ORIENTATION_TOLERANCE = 0.02


# ============================================================
# SINGULARITY SETTINGS
# ============================================================

SINGULARITY_CONDITION_WARNING = 300.0
SINGULARITY_CONDITION_ERROR = 1000.0

JACOBIAN_EPS_REVOLUTE = 0.1
JACOBIAN_EPS_PRISMATIC = 0.1


# ============================================================
# SMALL HELPERS
# ============================================================

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
        *
        t
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
# WORLD XYZ ORIENTATION DELTA
#
# orientation_delta:
#
# [rx, ry, rz]
#
# degree cinsinden.
#
# WORLD eksenlerinde uygulanır.
#
# R_target =
#
#     Rz @ Ry @ Rx @ R_start
#
# ============================================================

def apply_world_orientation_delta(
    R_start,
    orientation_delta
):

    orientation_delta = np.asarray(
        orientation_delta,
        dtype=float
    )


    if (
        orientation_delta.shape
        !=
        (3,)
    ):

        raise ValueError(

            (
                "orientation_delta "
                "[rx, ry, rz] biçiminde olmalı."
            )

        )


    rx = float(
        orientation_delta[0]
    )


    ry = float(
        orientation_delta[1]
    )


    rz = float(
        orientation_delta[2]
    )


    Rx = rotation_matrix_x(
        rx
    )


    Ry = rotation_matrix_y(
        ry
    )


    Rz = rotation_matrix_z(
        rz
    )


    return (

        Rz
        @
        Ry
        @
        Rx
        @
        R_start

    )

# ============================================================
# JOINT UTILITIES
# ============================================================

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

        for joint
        in joints

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
# VALIDATE SOLUTION
# ============================================================

def validate_solution(
    q_solution,
    joints,
    fk_function,
    command_index
):

    violations = (
        check_joint_limits(
            q_solution,
            joints
        )
    )


    if violations:

        return director_error(

            "JOINT_LIMIT_ERROR",

            command_index,

            (
                f"Command {command_index + 1}: "
                f"joint limiti aşıldı."
            ),

            violations=
                violations

        )


    singularity = (
        singularity_metric(
            q_solution,
            joints,
            fk_function
        )
    )


    if singularity[
        "error"
    ]:

        return director_error(

            "SINGULARITY_ERROR",

            command_index,

            (
                f"Command {command_index + 1}: "
                f"singular konfigürasyon."
            ),

            singularity=
                singularity

        )


    return None


# ============================================================
# GENERIC CARTESIAN VECTOR MOVE
#
# YENİ
#
# {
#     "type": "MOVE_CARTESIAN",
#     "delta": [dx, dy, dz]
# }
#
# Bu artık:
#
# - diagonal line
# - triangle
# - circle chord
# - arbitrary 3D line
#
# gibi hareketleri destekler.
# ============================================================

def plan_cartesian_command(
    q_start,
    command,
    joints,
    fk_function,
    command_index,
    linear_step_mm
):

    delta = command.get(
        "delta"
    )


    if (
        not isinstance(
            delta,
            (
                list,
                tuple
            )
        )
        or
        len(
            delta
        )
        !=
        3
    ):

        return (

            None,

            director_error(

                "INVALID_COMMAND",

                command_index,

                (
                    "MOVE_CARTESIAN delta "
                    "[dx, dy, dz] biçiminde olmalı."
                )

            )

        )


    delta = np.asarray(
        delta,
        dtype=float
    )


    if not np.all(
        np.isfinite(
            delta
        )
    ):

        return (

            None,

            director_error(

                "INVALID_COMMAND",

                command_index,

                "MOVE_CARTESIAN delta geçersiz."

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


    target_position = (

        start_position

        +
        delta

    )


    distance = float(
        np.linalg.norm(
            delta
        )
    )


    number_of_steps = max(

        1,

        int(
            np.ceil(

                distance

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

            start_position

            +
            delta
            *
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
                        f"Cartesian hedefe ulaşılamadı."
                    ),

                    position_error=
                        float(
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


        validation_error = (
            validate_solution(

                q_solution,

                joints,

                fk_function,

                command_index

            )
        )


        if (
            validation_error
            is not None
        ):

            return (
                None,
                validation_error
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

                "MOVE_CARTESIAN"

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
# FOLLOW PATH
#
# YENİ - WAYPOINT BASED CARTESIAN PATH
#
# Command:
#
# {
#     "type": "FOLLOW_PATH",
#
#     "reference": "CURRENT_TCP",
#
#     "waypoints": [
#
#         {
#             "position": [0, 0, 0],
#             "progress": 0.0,
#             "orientation_delta": [0, 0, 0]
#         },
#
#         ...
#
#     ]
# }
#
#
# position:
#
# command başladığı andaki TCP'ye göre RELATIVE.
#
#
# orientation_delta:
#
# command başladığı andaki TCP orientation'a göre
# WORLD X/Y/Z rotation.
#
#
# Eğer path boyunca orientation değişimi yoksa:
#
#     inverse_kinematics_position()
#
# kullanılır.
#
# Böylece 3-DOF / 4-DOF robotlar gereksiz yere
# full-pose IK zorlamasına maruz kalmaz.
#
#
# Eğer orientation modifier varsa:
#
#     inverse_kinematics()
#
# kullanılır.
# ============================================================

def plan_follow_path_command(
    q_start,
    command,
    joints,
    fk_function,
    command_index
):

    # ========================================================
    # WAYPOINTS
    # ========================================================

    waypoints = command.get(
        "waypoints"
    )


    if (
        not isinstance(
            waypoints,
            list
        )
        or
        len(
            waypoints
        )
        <
        2
    ):

        return (

            None,

            director_error(

                "INVALID_COMMAND",

                command_index,

                (
                    "FOLLOW_PATH en az "
                    "iki waypoint içermeli."
                )

            )

        )


    # ========================================================
    # REFERENCE
    # ========================================================

    reference = str(

        command.get(
            "reference",
            "CURRENT_TCP"
        )

    ).upper()


    if (
        reference
        !=
        "CURRENT_TCP"
    ):

        return (

            None,

            director_error(

                "INVALID_COMMAND",

                command_index,

                (
                    "FOLLOW_PATH şu an yalnızca "
                    "CURRENT_TCP reference destekliyor."
                )

            )

        )


    # ========================================================
    # COMMAND START POSE
    #
    # Tüm waypoint'ler buna göre relative.
    # ========================================================

    _, T_command_start = (
        forward_kinematics(

            q_start,

            fk_function

        )
    )


    T_command_start = np.asarray(
        T_command_start,
        dtype=float
    )


    start_position = (
        T_command_start[
            :3,
            3
        ].copy()
    )


    start_rotation = (
        T_command_start[
            :3,
            :3
        ].copy()
    )


    # ========================================================
    # DETECT ORIENTATION PATH
    #
    # Waypoint'lerin herhangi birinde orientation delta varsa
    # bütün path boyunca full-pose IK kullanacağız.
    #
    # Böylece:
    #
    # "circle çizerken yaw 90 dön"
    #
    # gibi işler orientation=0 başlangıç waypoint'inde bile
    # başlangıç orientation'ını korur.
    # ========================================================

    orientation_requested = False


    for waypoint in waypoints:

        orientation_delta = (
            waypoint.get(
                "orientation_delta",
                [
                    0.0,
                    0.0,
                    0.0
                ]
            )
        )


        try:

            orientation_vector = np.asarray(

                orientation_delta,

                dtype=float

            )


        except Exception:

            return (

                None,

                director_error(

                    "INVALID_COMMAND",

                    command_index,

                    (
                        "Waypoint orientation_delta "
                        "geçersiz."
                    )

                )

            )


        if (
            orientation_vector.shape
            !=
            (3,)
        ):

            return (

                None,

                director_error(

                    "INVALID_COMMAND",

                    command_index,

                    (
                        "orientation_delta "
                        "[rx, ry, rz] olmalı."
                    )

                )

            )


        if (
            np.linalg.norm(
                orientation_vector
            )
            >
            1e-9
        ):

            orientation_requested = True

            break


    # ========================================================
    # START SOLUTION
    # ========================================================

    q_current = np.asarray(

        q_start,

        dtype=float

    ).copy()


    points = []


    # ========================================================
    # WAYPOINT LOOP
    # ========================================================

    for waypoint_index, waypoint in enumerate(
        waypoints
    ):

        # ====================================================
        # POSITION
        # ====================================================

        relative_position = (
            waypoint.get(
                "position"
            )
        )


        if (
            not isinstance(
                relative_position,
                (
                    list,
                    tuple
                )
            )
            or
            len(
                relative_position
            )
            !=
            3
        ):

            return (

                None,

                director_error(

                    "INVALID_COMMAND",

                    command_index,

                    (
                        f"Waypoint {waypoint_index}: "
                        "position [x,y,z] olmalı."
                    ),

                    waypoint_index=
                        int(
                            waypoint_index
                        )

                )

            )


        try:

            relative_position = np.asarray(

                relative_position,

                dtype=float

            )


        except Exception:

            return (

                None,

                director_error(

                    "INVALID_COMMAND",

                    command_index,

                    (
                        f"Waypoint {waypoint_index}: "
                        "position sayısal değil."
                    ),

                    waypoint_index=
                        int(
                            waypoint_index
                        )

                )

            )


        if not np.all(
            np.isfinite(
                relative_position
            )
        ):

            return (

                None,

                director_error(

                    "INVALID_COMMAND",

                    command_index,

                    (
                        f"Waypoint {waypoint_index}: "
                        "position geçersiz."
                    ),

                    waypoint_index=
                        int(
                            waypoint_index
                        )

                )

            )


        target_position = (

            start_position

            +
            relative_position

        )


        # ====================================================
        # ORIENTATION DELTA
        # ====================================================

        orientation_delta = np.asarray(

            waypoint.get(

                "orientation_delta",

                [
                    0.0,
                    0.0,
                    0.0
                ]

            ),

            dtype=float

        )


        # ====================================================
        # FULL POSE IK
        #
        # orientation modifier varsa.
        # ====================================================

        if orientation_requested:

            try:

                target_rotation = (
                    apply_world_orientation_delta(

                        start_rotation,

                        orientation_delta

                    )
                )


            except Exception as error:

                return (

                    None,

                    director_error(

                        "INVALID_COMMAND",

                        command_index,

                        (
                            f"Waypoint {waypoint_index}: "
                            f"{str(error)}"
                        ),

                        waypoint_index=
                            int(
                                waypoint_index
                            )

                    )

                )


            T_target = np.eye(
                4,
                dtype=float
            )


            T_target[
                :3,
                :3
            ] = (
                target_rotation
            )


            T_target[
                :3,
                3
            ] = (
                target_position
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

                ik_result[
                    "q"
                ],

                dtype=float

            )


            # =================================================
            # POSE VALIDATION
            # =================================================

            if (
                not ik_result[
                    "success"
                ]
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
                            f"Waypoint {waypoint_index + 1} "
                            "pose gerçekleştirilemedi."
                        ),

                        waypoint_index=
                            int(
                                waypoint_index
                            ),

                        position_error=
                            float(
                                ik_result[
                                    "position_error"
                                ]
                            ),

                        orientation_error=
                            float(
                                ik_result[
                                    "orientation_error"
                                ]
                            ),

                        target_position=
                            [
                                float(value)
                                for value
                                in target_position
                            ],

                        orientation_delta=
                            [
                                float(value)
                                for value
                                in orientation_delta
                            ]

                    )

                )


        # ====================================================
        # POSITION-ONLY IK
        #
        # Circle / square / helix gibi path'te TCP
        # orientation değişmiyorsa.
        # ====================================================

        else:

            ik_result = (
                inverse_kinematics_position(

                    target_position,

                    q_current,

                    joints,

                    fk_function

                )
            )


            q_solution = np.asarray(

                ik_result[
                    "q"
                ],

                dtype=float

            )


            if (
                not ik_result[
                    "success"
                ]
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
                            f"Waypoint {waypoint_index + 1} "
                            "ulaşılamıyor."
                        ),

                        waypoint_index=
                            int(
                                waypoint_index
                            ),

                        position_error=
                            float(
                                ik_result[
                                    "position_error"
                                ]
                            ),

                        target_position=
                            [
                                float(value)
                                for value
                                in target_position
                            ]

                    )

                )


        # ====================================================
        # JOINT LIMIT + SINGULARITY
        # ====================================================

        validation_error = (
            validate_solution(

                q_solution,

                joints,

                fk_function,

                command_index

            )
        )


        if (
            validation_error
            is not None
        ):

            validation_error[
                "waypoint_index"
            ] = int(
                waypoint_index
            )


            validation_error[
                "target_position"
            ] = [

                float(value)

                for value
                in target_position

            ]


            return (

                None,

                validation_error

            )


        # ====================================================
        # ACCEPT SOLUTION
        # ====================================================

        q_current = (
            q_solution
        )


        # ====================================================
        # SERIALIZE TRAJECTORY POINT
        # ====================================================

        trajectory_point = (
            create_trajectory_point(

                q_current,

                joints,

                fk_function,

                command_index,

                "FOLLOW_PATH"

            )
        )


        # ====================================================
        # PATH METADATA
        # ====================================================

        trajectory_point[
            "waypoint_index"
        ] = int(
            waypoint_index
        )


        trajectory_point[
            "path_progress"
        ] = float(

            waypoint.get(

                "progress",

                (
                    waypoint_index

                    /

                    max(
                        len(
                            waypoints
                        )
                        -
                        1,

                        1
                    )
                )

            )

        )


        trajectory_point[
            "path_label"
        ] = str(

            waypoint.get(
                "label",
                "Waypoint"
            )

        )


        trajectory_point[
            "target_tcp"
        ] = [

            float(value)

            for value
            in target_position

        ]


        trajectory_point[
            "orientation_delta"
        ] = [

            float(value)

            for value
            in orientation_delta

        ]


        points.append(
            trajectory_point
        )


    # ========================================================
    # SUCCESS
    # ========================================================

    return (

        {

            "q_end":
                q_current,

            "points":
                points,

            "waypoint_count":
                len(
                    waypoints
                ),

            "orientation_controlled":
                bool(
                    orientation_requested
                )

        },

        None

    )

# ============================================================
# OLD AXIS-BASED LINEAR MOVE
#
# Geriye dönük uyumluluk için koruyoruz.
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


    delta = {

        "X":
            [
                value,
                0.0,
                0.0
            ],

        "Y":
            [
                0.0,
                value,
                0.0
            ],

        "Z":
            [
                0.0,
                0.0,
                value
            ]

    }[
        axis
    ]


    converted_command = {

        "type":
            "MOVE_CARTESIAN",

        "delta":
            delta

    }


    return plan_cartesian_command(

        q_start,

        converted_command,

        joints,

        fk_function,

        command_index,

        linear_step_mm

    )


# ============================================================
# ROTATE TCP
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
                        f"istenen TCP orientation "
                        f"gerçekleştirilemedi."
                    ),

                    position_error=
                        float(
                            ik_result[
                                "position_error"
                            ]
                        ),

                    orientation_error=
                        float(
                            ik_result[
                                "orientation_error"
                            ]
                        )

                )

            )


        validation_error = (
            validate_solution(

                q_solution,

                joints,

                fk_function,

                command_index

            )
        )


        if (
            validation_error
            is not None
        ):

            return (
                None,
                validation_error
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
# MOVE JOINT
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


        validation_error = (
            validate_solution(

                q_current,

                joints,

                fk_function,

                command_index

            )
        )


        if (
            validation_error
            is not None
        ):

            return (
                None,
                validation_error
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
# RETURN TCP TO PROGRAM START
#
# YENİ
#
# TCP position programın başladığı noktaya
# düz bir Cartesian yol ile döner.
#
# Orientation zorlanmaz.
# Bu sayede 3-DOF robotlarda da kullanılabilir.
# ============================================================

def plan_return_tcp_to_start(
    q_current,
    start_tcp_position,
    joints,
    fk_function,
    command_index,
    linear_step_mm
):

    _, T_current = (
        forward_kinematics(
            q_current,
            fk_function
        )
    )


    current_position = (
        T_current[
            :3,
            3
        ].copy()
    )


    start_tcp_position = np.asarray(

        start_tcp_position,

        dtype=float

    )


    delta = (

        start_tcp_position

        -
        current_position

    )


    return plan_cartesian_command(

        q_current,

        {

            "type":
                "MOVE_CARTESIAN",

            "delta":
                delta.tolist()

        },

        joints,

        fk_function,

        command_index,

        linear_step_mm

    )


# ============================================================
# RETURN JOINTS TO PROGRAM START
#
# YENİ
#
# Program başladığında kaydedilen q_start'a
# joint-space interpolation ile döner.
# ============================================================

def plan_return_joints_to_start(
    q_current,
    q_program_start,
    joints,
    fk_function,
    command_index,
    revolute_step_deg,
    prismatic_step_mm
):

    q_current = np.asarray(
        q_current,
        dtype=float
    )


    q_program_start = np.asarray(
        q_program_start,
        dtype=float
    )


    delta = (

        q_program_start

        -
        q_current

    )


    step_counts = []


    for index, joint in enumerate(
        joints
    ):

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


        step_counts.append(

            int(
                np.ceil(

                    abs(
                        delta[index]
                    )

                    /
                    max(
                        step_size,
                        1e-6
                    )

                )
            )

        )


    number_of_steps = max(

        1,

        max(
            step_counts,
            default=1
        )

    )


    points = []


    q_step = (
        q_current.copy()
    )


    for step_index in range(
        1,
        number_of_steps + 1
    ):

        t = (
            step_index
            /
            number_of_steps
        )


        q_step = (

            q_current

            +
            delta
            *
            t

        )


        validation_error = (
            validate_solution(

                q_step,

                joints,

                fk_function,

                command_index

            )
        )


        if (
            validation_error
            is not None
        ):

            return (
                None,
                validation_error
            )


        points.append(

            create_trajectory_point(

                q_step,

                joints,

                fk_function,

                command_index,

                "RETURN_JOINTS_TO_START"

            )

        )


    return (

        {

            "q_end":
                q_program_start.copy(),

            "points":
                points

        },

        None

    )


# ============================================================
# DIRECTOR PROGRAM PLANNER
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


    # ========================================================
    # INITIAL JOINT CONFIGURATION
    # ========================================================

    q_current = (
        joint_vector_from_values(
            joints,
            values
        )
    )


    q_program_start = (
        q_current.copy()
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

            (
                "Başlangıç joint konfigürasyonu "
                "limit dışında."
            ),

            violations=
                initial_violations

        )


    # ========================================================
    # SAVE PROGRAM START TCP
    # ========================================================

    _, T_program_start = (
        forward_kinematics(
            q_program_start,
            fk_function
        )
    )


    start_tcp_position = (
        T_program_start[
            :3,
            3
        ].copy()
    )


    start_tcp_rotation = (
        T_program_start[
            :3,
            :3
        ].copy()
    )


    # ========================================================
    # TRAJECTORY
    # ========================================================

    trajectory = []


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
    # PROGRAM LOOP
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


        result = None
        error = None


        # ====================================================
        # MOVE LINEAR
        # ====================================================

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


        # ====================================================
        # MOVE CARTESIAN
        # ====================================================

        elif (
            command_type
            ==
            "MOVE_CARTESIAN"
        ):

            result, error = (
                plan_cartesian_command(

                    q_current,

                    command,

                    joints,

                    fk_function,

                    command_index,

                    linear_step_mm

                )
            )

                # ====================================================
        # FOLLOW PATH
        #
        # Waypoint based arbitrary 3D Cartesian trajectory.
        #
        # Circle
        # Helix
        # Triangle
        # Rising square
        # Orientation-progress path
        #
        # ====================================================

        elif (
            command_type
            ==
            "FOLLOW_PATH"
        ):

            result, error = (
                plan_follow_path_command(

                    q_current,

                    command,

                    joints,

                    fk_function,

                    command_index

                )
            )

        # ====================================================
        # ROTATE TCP
        # ====================================================

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


        # ====================================================
        # MOVE JOINT
        # ====================================================

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


        # ====================================================
        # RETURN TCP TO START
        # ====================================================

        elif (
            command_type
            ==
            "RETURN_TCP_TO_START"
        ):

            result, error = (
                plan_return_tcp_to_start(

                    q_current,

                    start_tcp_position,

                    joints,

                    fk_function,

                    command_index,

                    linear_step_mm

                )
            )


        # ====================================================
        # RETURN JOINTS TO START
        # ====================================================

        elif (
            command_type
            ==
            "RETURN_JOINTS_TO_START"
        ):

            result, error = (
                plan_return_joints_to_start(

                    q_current,

                    q_program_start,

                    joints,

                    fk_function,

                    command_index,

                    revolute_step_deg,

                    prismatic_step_mm

                )
            )


        # ====================================================
        # UNKNOWN
        # ====================================================

        else:

            return director_error(

                "UNKNOWN_COMMAND",

                command_index,

                (
                    f"Bilinmeyen Director komutu: "
                    f"{command_type}"
                )

            )


        # ====================================================
        # COMMAND ERROR
        # ====================================================

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


        # ====================================================
        # COMMAND SUCCESS
        # ====================================================

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
    # PROGRAM STATISTICS
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


    q_start_dict = (
        joint_dict_from_vector(
            joints,
            q_program_start
        )
    )


    # ========================================================
    # RESPONSE
    # ========================================================

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

        "q_start":
            q_start_dict,

        "q_final":
            q_end_dict,

        "start_tcp":
            [
                float(value)
                for value
                in start_tcp_position
            ],

        "start_rotation":
            start_tcp_rotation.tolist(),

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