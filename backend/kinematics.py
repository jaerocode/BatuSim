import numpy as np
import sympy as sp

from scipy.optimize import least_squares

from backend.robot_model import (
    dh_matrix,
    SYMBOLS
)


# ============================================================
# FAST FK PREPARATION
# ============================================================

def prepare_fk(
    dh_table,
    joints,
    parameters
):
    """
    DH tablosundan hızlı sayısal FK fonksiyonu hazırlar.

    Bir kere çağrılır.

    dh_table:
        Robotun DH tablosu

    joints:
        Aktif eklem listesi

    parameters:
        L1, L2, L3... gibi sabit geometrik parametreler

    Dönüş:
        fk_function(q_vector)
    """

    T = sp.eye(4)

    symbolic_frames = [
        T
    ]


    # ========================================================
    # SYMBOLIC TRANSFORMATION CHAIN
    # ========================================================

    for row in dh_table:

        A = dh_matrix(
            row["theta"],
            row["d"],
            row["a"],
            row["alpha"]
        )

        T = (
            T * A
        )

        symbolic_frames.append(
            T
        )


    # ========================================================
    # ROBOT GEOMETRIC PARAMETERS
    # ========================================================

    parameter_subs = {}


    for name, value in parameters.items():

        if (
            value is not None
            and name in SYMBOLS
        ):

            parameter_subs[
                SYMBOLS[name]
            ] = value


    symbolic_frames = [

        frame.subs(
            parameter_subs
        )

        for frame in symbolic_frames

    ]


    # ========================================================
    # JOINT SYMBOLS
    #
    # q1 q2 q3...
    #
    # Joint sayısı hard-coded değil.
    # ========================================================

    joint_symbols = [

        joint["symbol"]

        for joint in joints

    ]


    # ========================================================
    # LAMBDIFY
    #
    # SymPy -> hızlı NumPy fonksiyonları
    # ========================================================

    frame_functions = [

        sp.lambdify(
            joint_symbols,
            frame,
            "numpy"
        )

        for frame in symbolic_frames

    ]


    # ========================================================
    # FAST FK FUNCTION
    # ========================================================

    def fk_function(
        q_vector
    ):

        q_vector = np.asarray(
            q_vector,
            dtype=float
        )


        if len(q_vector) != len(
            joint_symbols
        ):

            raise ValueError(

                f"Beklenen joint sayısı: "
                f"{len(joint_symbols)}, "
                f"gelen: {len(q_vector)}"

            )


        numeric_frames = []


        for function in frame_functions:

            T_numeric = np.array(

                function(
                    *q_vector
                ),

                dtype=float

            )

            numeric_frames.append(
                T_numeric
            )


        return numeric_frames


    return fk_function


# ============================================================
# GENERIC FORWARD KINEMATICS
# ============================================================

def forward_kinematics(
    q_vector,
    fk_function
):
    """
    q_vector
        ↓
    T00, T01, T02 ... T0Tool
    """

    frames = fk_function(
        q_vector
    )

    T_tool = frames[-1]


    return (
        frames,
        T_tool
    )


# ============================================================
# ROTATION ERROR
# ============================================================

def rotation_error(
    R_current,
    R_target
):
    """
    3D orientation error vector.

    Output:

        [ex, ey, ez]
    """

    R_err = (

        R_target

        @ R_current.T

    )


    error = 0.5 * np.array([

        R_err[2, 1]
        - R_err[1, 2],

        R_err[0, 2]
        - R_err[2, 0],

        R_err[1, 0]
        - R_err[0, 1]

    ])


    return error


# ============================================================
# GENERIC FULL POSE IK
#
# X Y Z + Rx Ry Rz
# ============================================================

def inverse_kinematics(
    T_target,
    q_start,
    joints,
    fk_function,
    position_weight=1.0,
    orientation_weight=100.0
):
    """
    Genel 6D pose IK.

    T_target:
        4x4 homogeneous target matrix

    Kullanım açısından özellikle 6-DOF ve üzeri
    robotlarda anlamlıdır.

    3-4 DOF robotlarda full 6D pose her zaman
    gerçekleştirilemeyebilir.
    """

    q_start = np.asarray(
        q_start,
        dtype=float
    )


    T_target = np.asarray(
        T_target,
        dtype=float
    )


    # ========================================================
    # JOINT LIMITS
    # ========================================================

    lower = np.array([

        joint["min"]

        for joint in joints

    ], dtype=float)


    upper = np.array([

        joint["max"]

        for joint in joints

    ], dtype=float)


    # ========================================================
    # ERROR FUNCTION
    # ========================================================

    def error_function(
        q_vector
    ):

        _, T_current = (
            forward_kinematics(
                q_vector,
                fk_function
            )
        )


        # ----------------------------------------------------
        # POSITION ERROR
        # ----------------------------------------------------

        p_current = T_current[
            :3,
            3
        ]

        p_target = T_target[
            :3,
            3
        ]


        p_error = (

            p_target

            - p_current

        )


        # ----------------------------------------------------
        # ORIENTATION ERROR
        # ----------------------------------------------------

        R_current = T_current[
            :3,
            :3
        ]

        R_target = T_target[
            :3,
            :3
        ]


        r_error = rotation_error(

            R_current,

            R_target

        )


        # ----------------------------------------------------
        # COMPLETE 6D ERROR
        # ----------------------------------------------------

        return np.concatenate([

            position_weight
            * p_error,

            orientation_weight
            * r_error

        ])


    # ========================================================
    # SOLVE
    # ========================================================

    result = least_squares(

        error_function,

        q_start,

        bounds=(
            lower,
            upper
        ),

        max_nfev=300,

        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10
    )


    q_solution = (
        result.x
    )


    # ========================================================
    # FK VERIFICATION
    # ========================================================

    frames, T_solution = (
        forward_kinematics(

            q_solution,

            fk_function

        )
    )


    # --------------------------------------------------------
    # POSITION ERROR
    # --------------------------------------------------------

    position_error = np.linalg.norm(

        T_target[
            :3,
            3
        ]

        - T_solution[
            :3,
            3
        ]

    )


    # --------------------------------------------------------
    # ORIENTATION ERROR
    # --------------------------------------------------------

    orientation_error_vector = (

        rotation_error(

            T_solution[
                :3,
                :3
            ],

            T_target[
                :3,
                :3
            ]

        )

    )


    orientation_error = np.linalg.norm(
        orientation_error_vector
    )


    return {

        "q":
            q_solution,

        "success":
            result.success,

        "position_error":
            float(
                position_error
            ),

        "orientation_error":
            float(
                orientation_error
            ),

        "frames":
            frames,

        "T_solution":
            T_solution

    }


# ============================================================
# GENERIC POSITION IK
#
# X Y Z ONLY
# ============================================================

def inverse_kinematics_position(
    target_position,
    q_start,
    joints,
    fk_function
):
    """
    Generic Cartesian position IK.

    Yalnızca:

        X
        Y
        Z

    çözülür.

    Orientation serbesttir.

    Bu yüzden:

        3 DOF
        4 DOF
        6 DOF
        R/P karışık robotlar

    için Linear Jog tarafında kullanılabilir.
    """

    q_start = np.asarray(
        q_start,
        dtype=float
    )


    target_position = np.asarray(
        target_position,
        dtype=float
    )


    # ========================================================
    # JOINT LIMITS
    # ========================================================

    lower = np.array([

        joint["min"]

        for joint in joints

    ], dtype=float)


    upper = np.array([

        joint["max"]

        for joint in joints

    ], dtype=float)


    # ========================================================
    # JOINT SCALE
    #
    # Revolute: degree
    # Prismatic: mm
    #
    # Farklı fiziksel birimlerin solver'a etkisini azaltıyoruz.
    # ========================================================

    joint_scale = np.array([

        45.0
        if joint["type"] == "R"

        else 100.0

        for joint in joints

    ], dtype=float)


    # ========================================================
    # ERROR FUNCTION
    # ========================================================

    def error_function(
        q_vector
    ):

        _, T_current = (
            forward_kinematics(

                q_vector,

                fk_function

            )
        )


        current_position = T_current[
            :3,
            3
        ]


        # ----------------------------------------------------
        # CARTESIAN ERROR
        # ----------------------------------------------------

        position_error = (

            target_position

            - current_position

        )


        # ----------------------------------------------------
        # REGULARIZATION
        #
        # Birden fazla IK çözümü varsa mevcut konfigürasyona
        # yakın olanı tercih et.
        # ----------------------------------------------------

        q_change = (

            q_vector

            - q_start

        ) / joint_scale


        return np.concatenate([

            position_error,

            0.01
            * q_change

        ])


    # ========================================================
    # SOLVE
    # ========================================================

    result = least_squares(

        error_function,

        q_start,

        bounds=(
            lower,
            upper
        ),

        max_nfev=300,

        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9
    )


    q_solution = (
        result.x
    )


    # ========================================================
    # FK VERIFICATION
    # ========================================================

    frames, T_solution = (
        forward_kinematics(

            q_solution,

            fk_function

        )
    )


    actual_position = T_solution[
        :3,
        3
    ]


    position_error = np.linalg.norm(

        actual_position

        - target_position

    )


    return {

        "q":
            q_solution,

        "success":
            result.success,

        "position_error":
            float(
                position_error
            ),

        "frames":
            frames,

        "T_solution":
            T_solution

    }