import numpy as np

from backend.kinematics import (
    forward_kinematics,
    inverse_kinematics_position
)


# ============================================================
# JOINT JOG
# ============================================================

def joint_jog(
    q_current,
    joint_index,
    direction,
    joints,
    revolute_step=1.0,
    prismatic_step=5.0
):

    q_new = np.array(
        q_current,
        dtype=float
    ).copy()


    joint = joints[
        joint_index
    ]


    # ========================================================
    # STEP
    # ========================================================

    if joint["type"] == "R":

        step = abs(
            revolute_step
        )

    else:

        step = abs(
            prismatic_step
        )


    # ========================================================
    # MOVE
    # ========================================================

    q_new[
        joint_index
    ] += (
        direction * step
    )


    # ========================================================
    # LIMIT
    # ========================================================

    q_new[
        joint_index
    ] = np.clip(

        q_new[joint_index],

        joint["min"],

        joint["max"]

    )


    return q_new


# ============================================================
# LINEAR JOG
# ============================================================

def linear_jog(
    q_current,
    axis,
    direction,
    step,
    joints,
    fk_function
):

    axis = axis.upper()


    axis_map = {
        "X": 0,
        "Y": 1,
        "Z": 2
    }


    if axis not in axis_map:

        raise ValueError(
            "Axis X, Y veya Z olmalı."
        )


    # ========================================================
    # CURRENT TCP
    # ========================================================

    _, T_current = forward_kinematics(
        q_current,
        fk_function
    )


    current_position = T_current[
        :3,
        3
    ].copy()


    # ========================================================
    # TARGET TCP
    # ========================================================

    target_position = (
        current_position.copy()
    )


    target_position[
        axis_map[axis]
    ] += (
        direction
        * abs(step)
    )


    # ========================================================
    # IK
    # ========================================================

    result = inverse_kinematics_position(

        target_position,

        q_current,

        joints,

        fk_function

    )


    result["target_position"] = (
        target_position
    )

    result["axis"] = axis

    result["distance"] = (
        direction
        * abs(step)
    )


    return result