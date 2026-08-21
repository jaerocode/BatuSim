import math
from copy import deepcopy


# ============================================================
# PATH GENERATOR V2
#
# High-level semantic robot task
#               ↓
# 3D Cartesian waypoint path
#               ↓
# FOLLOW_PATH
#               ↓
# Director / IK / validation
#
#
# Bu modül:
#
# - IK çözmez.
# - Joint açılarını hesaplamaz.
# - DH modelini bilmez.
#
# Görevi yalnızca geometrik bir Cartesian path'e çevirir.
#
#
# Örnek:
#
# DRAW_SHAPE:
#   CIRCLE
#   plane = XY
#   radius = 50
#
# modifier:
#   LINEAR_PROGRESS
#   axis = Z
#   distance = 50
#
# sonuç:
#
#     circle + Z ramp
#         =
#     3D helix path
#
# ============================================================


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_CIRCLE_SEGMENTS = 64

DEFAULT_LINE_SEGMENTS = 10

DEFAULT_SHAPE_EDGE_STEP_MM = 5.0

MIN_CIRCLE_SEGMENTS = 12

MAX_CIRCLE_SEGMENTS = 360


# ============================================================
# ERROR
# ============================================================

class PathGeneratorError(
    ValueError
):
    pass


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_text(
    value
):

    return str(
        value
        or
        ""
    ).strip().upper()


def finite_number(
    value,
    name
):

    try:

        number = float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        raise PathGeneratorError(
            f"{name} sayısal olmalı."
        )


    if not math.isfinite(
        number
    ):

        raise PathGeneratorError(
            f"{name} geçerli bir sayı olmalı."
        )


    return number


def positive_number(
    value,
    name
):

    number = finite_number(
        value,
        name
    )


    if number <= 0:

        raise PathGeneratorError(
            f"{name} sıfırdan büyük olmalı."
        )


    return number


# ============================================================
# AXIS
# ============================================================

def normalize_axis(
    axis
):

    axis = normalize_text(
        axis
    )


    aliases = {

        "X":
            "X",

        "Y":
            "Y",

        "Z":
            "Z",

        "ROLL":
            "X",

        "PITCH":
            "Y",

        "YAW":
            "Z"

    }


    if axis not in aliases:

        raise PathGeneratorError(

            (
                "Axis X, Y, Z, "
                "ROLL, PITCH veya YAW olmalı."
            )

        )


    return aliases[
        axis
    ]


# ============================================================
# PLANE
# ============================================================

def normalize_plane(
    plane
):

    plane = (
        normalize_text(
            plane
        )
        .replace(
            "-",
            ""
        )
        .replace(
            "_",
            ""
        )
        .replace(
            " ",
            ""
        )
    )


    aliases = {

        "XY":
            "XY",

        "YX":
            "XY",

        "XZ":
            "XZ",

        "ZX":
            "XZ",

        "YZ":
            "YZ",

        "ZY":
            "YZ"

    }


    if plane not in aliases:

        raise PathGeneratorError(
            "Plane yalnızca XY, XZ veya YZ olabilir."
        )


    return aliases[
        plane
    ]


# ============================================================
# AXIS VECTOR
# ============================================================

def axis_vector(
    axis,
    value
):

    axis = normalize_axis(
        axis
    )


    value = float(
        value
    )


    if axis == "X":

        return [
            value,
            0.0,
            0.0
        ]


    if axis == "Y":

        return [
            0.0,
            value,
            0.0
        ]


    return [
        0.0,
        0.0,
        value
    ]


# ============================================================
# PLANE VECTOR
#
# Generic 2D:
#
#     u
#     v
#
# world XYZ'ye dönüştürülür.
#
# ============================================================

def plane_vector(
    plane,
    u,
    v
):

    plane = normalize_plane(
        plane
    )


    u = float(
        u
    )


    v = float(
        v
    )


    if plane == "XY":

        return [
            u,
            v,
            0.0
        ]


    if plane == "XZ":

        return [
            u,
            0.0,
            v
        ]


    return [
        0.0,
        u,
        v
    ]


# ============================================================
# VECTOR OPERATIONS
# ============================================================

def vector_add(
    a,
    b
):

    return [

        float(
            a[0]
        )
        +
        float(
            b[0]
        ),

        float(
            a[1]
        )
        +
        float(
            b[1]
        ),

        float(
            a[2]
        )
        +
        float(
            b[2]
        )

    ]


def vector_subtract(
    a,
    b
):

    return [

        float(
            a[0]
        )
        -
        float(
            b[0]
        ),

        float(
            a[1]
        )
        -
        float(
            b[1]
        ),

        float(
            a[2]
        )
        -
        float(
            b[2]
        )

    ]


def vector_scale(
    vector,
    scale
):

    return [

        float(
            vector[0]
        )
        *
        float(
            scale
        ),

        float(
            vector[1]
        )
        *
        float(
            scale
        ),

        float(
            vector[2]
        )
        *
        float(
            scale
        )

    ]


def vector_length(
    vector
):

    return math.sqrt(

        float(
            vector[0]
        ) ** 2

        +

        float(
            vector[1]
        ) ** 2

        +

        float(
            vector[2]
        ) ** 2

    )


# ============================================================
# WAYPOINT
#
# position:
#
# Program başlangıç TCP'sine göre RELATIVE position.
#
# orientation_delta:
#
# WORLD X/Y/Z rotation,
# degree.
#
# ============================================================

def create_waypoint(
    position,
    progress,
    orientation_delta=None,
    label=None
):

    if orientation_delta is None:

        orientation_delta = [
            0.0,
            0.0,
            0.0
        ]


    return {

        "position": [

            float(
                position[0]
            ),

            float(
                position[1]
            ),

            float(
                position[2]
            )

        ],

        "progress":
            float(
                progress
            ),

        "orientation_delta": [

            float(
                orientation_delta[0]
            ),

            float(
                orientation_delta[1]
            ),

            float(
                orientation_delta[2]
            )

        ],

        "label":
            label
            or
            "Waypoint"

    }


# ============================================================
# FOLLOW PATH COMMAND
#
# Director V3'te bunu çözecek:
#
# {
#     type: FOLLOW_PATH,
#     waypoints: [...]
# }
#
# ============================================================

def follow_path_command(
    waypoints,
    label="Generated Path"
):

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

        raise PathGeneratorError(

            (
                "FOLLOW_PATH en az "
                "iki waypoint içermeli."
            )

        )


    return {

        "type":
            "FOLLOW_PATH",

        "reference":
            "CURRENT_TCP",

        "waypoints":
            waypoints,

        "label":
            label

    }


# ============================================================
# APPLY MODIFIERS
#
# Waypoint path hazırlandıktan sonra:
#
# LINEAR_PROGRESS
# ROTATION_PROGRESS
#
# gibi eşzamanlı hareketler eklenir.
#
#
# Örn:
#
# circle XY
#
# +
#
# {
#   type: LINEAR_PROGRESS,
#   axis: Z,
#   distance: 50
# }
#
# = helix
#
# ============================================================

def apply_modifiers(
    waypoints,
    modifiers
):

    if not modifiers:

        return waypoints


    result = deepcopy(
        waypoints
    )


    for modifier in modifiers:

        modifier_type = normalize_text(

            modifier.get(
                "type"
            )

        )


        # ====================================================
        # LINEAR PROGRESS
        #
        # Shape ilerlerken başka bir Cartesian eksende
        # lineer ilerleme.
        #
        # Örn:
        #
        # circle XY + Z +50
        # ====================================================

        if (
            modifier_type
            ==
            "LINEAR_PROGRESS"
        ):

            axis = normalize_axis(

                modifier.get(
                    "axis"
                )

            )


            distance = finite_number(

                modifier.get(
                    "distance"
                ),

                "modifier distance"

            )


            for waypoint in result:

                progress = float(
                    waypoint[
                        "progress"
                    ]
                )


                offset = axis_vector(

                    axis,

                    distance
                    *
                    progress

                )


                waypoint[
                    "position"
                ] = vector_add(

                    waypoint[
                        "position"
                    ],

                    offset

                )


        # ====================================================
        # ROTATION PROGRESS
        #
        # Shape boyunca TCP orientation değişir.
        #
        # Örn:
        #
        # square XY
        # +
        # yaw +90°
        #
        # path sonunda yaw toplam +90° olur.
        # ====================================================

        elif (
            modifier_type
            ==
            "ROTATION_PROGRESS"
        ):

            axis = normalize_axis(

                modifier.get(
                    "axis"
                )

            )


            angle = finite_number(

                modifier.get(
                    "angle"
                ),

                "modifier angle"

            )


            axis_index = {

                "X":
                    0,

                "Y":
                    1,

                "Z":
                    2

            }[
                axis
            ]


            for waypoint in result:

                progress = float(
                    waypoint[
                        "progress"
                    ]
                )


                waypoint[
                    "orientation_delta"
                ][
                    axis_index
                ] += (

                    angle

                    *
                    progress

                )


        else:

            raise PathGeneratorError(

                (
                    "Desteklenmeyen modifier: "
                    f"{modifier_type}"
                )

            )


    return result


# ============================================================
# LINEAR INTERPOLATION
#
# start -> end arasında N waypoint
#
# ============================================================

def interpolate_line(
    start,
    end,
    segments,
    progress_start=0.0,
    progress_end=1.0,
    label="Line"
):

    segments = max(
        1,
        int(
            segments
        )
    )


    result = []


    for index in range(
        segments + 1
    ):

        t = (
            index
            /
            segments
        )


        position = [

            start[0]
            +
            (
                end[0]
                -
                start[0]
            )
            *
            t,

            start[1]
            +
            (
                end[1]
                -
                start[1]
            )
            *
            t,

            start[2]
            +
            (
                end[2]
                -
                start[2]
            )
            *
            t

        ]


        progress = (

            progress_start

            +

            (
                progress_end
                -
                progress_start
            )

            *
            t

        )


        result.append(

            create_waypoint(

                position,

                progress,

                label=
                    label

            )

        )


    return result


# ============================================================
# REMOVE DUPLICATE ADJACENT WAYPOINTS
# ============================================================

def remove_duplicate_waypoints(
    waypoints,
    tolerance=1e-9
):

    if not waypoints:

        return []


    result = [
        waypoints[0]
    ]


    for waypoint in waypoints[
        1:
    ]:

        previous = result[
            -1
        ]


        delta = vector_subtract(

            waypoint[
                "position"
            ],

            previous[
                "position"
            ]

        )


        if (
            vector_length(
                delta
            )
            >
            tolerance
        ):

            result.append(
                waypoint
            )


        else:

            # Geometrik nokta aynıysa ama progress/orientation
            # değişmiş olabilir.
            #
            # En son olanı koruyoruz.

            result[
                -1
            ] = waypoint


    return result


# ============================================================
# POLYLINE → WAYPOINTS
#
# vertices relative Cartesian positions.
#
# ============================================================

def polyline_to_waypoints(
    vertices,
    step_mm=DEFAULT_SHAPE_EDGE_STEP_MM,
    closed=False,
    label="Polyline"
):

    if (
        not isinstance(
            vertices,
            list
        )
        or
        len(
            vertices
        )
        <
        2
    ):

        raise PathGeneratorError(

            "Polyline en az iki vertex içermeli."

        )


    path_vertices = deepcopy(
        vertices
    )


    if closed:

        first = path_vertices[
            0
        ]


        last = path_vertices[
            -1
        ]


        if (
            vector_length(
                vector_subtract(
                    last,
                    first
                )
            )
            >
            1e-9
        ):

            path_vertices.append(
                deepcopy(
                    first
                )
            )


    # ========================================================
    # TOTAL LENGTH
    #
    # progress bütün shape boyunca 0→1 olacak.
    # ========================================================

    lengths = []


    total_length = 0.0


    for index in range(
        len(
            path_vertices
        )
        -
        1
    ):

        length = vector_length(

            vector_subtract(

                path_vertices[
                    index + 1
                ],

                path_vertices[
                    index
                ]

            )

        )


        lengths.append(
            length
        )


        total_length += (
            length
        )


    if total_length <= 1e-12:

        raise PathGeneratorError(
            "Shape path uzunluğu sıfır."
        )


    waypoints = []


    travelled = 0.0


    for edge_index, edge_length in enumerate(
        lengths
    ):

        start = path_vertices[
            edge_index
        ]


        end = path_vertices[
            edge_index + 1
        ]


        segments = max(

            1,

            int(
                math.ceil(

                    edge_length

                    /
                    max(
                        step_mm,
                        1e-6
                    )

                )
            )

        )


        progress_start = (

            travelled

            /
            total_length

        )


        progress_end = (

            (
                travelled
                +
                edge_length
            )

            /
            total_length

        )


        edge_waypoints = (
            interpolate_line(

                start,

                end,

                segments,

                progress_start,

                progress_end,

                label=
                    f"{label} {edge_index + 1}"

            )
        )


        # İlk edge hariç:
        # ortak vertex'i duplicate etme.

        if (
            edge_index
            >
            0
        ):

            edge_waypoints = (
                edge_waypoints[
                    1:
                ]
            )


        waypoints.extend(
            edge_waypoints
        )


        travelled += (
            edge_length
        )


    return remove_duplicate_waypoints(
        waypoints
    )


# ============================================================
# GENERATE SQUARE WAYPOINTS
#
# Program başlangıç TCP:
#
# [0,0,0]
#
# şeklinde kabul edilir.
#
# ============================================================

def generate_square_waypoints(
    task
):

    plane = normalize_plane(

        task.get(
            "plane",
            "XY"
        )

    )


    size = positive_number(

        task.get(
            "size"
        ),

        "square size"

    )


    step_mm = positive_number(

        task.get(
            "path_step_mm",
            DEFAULT_SHAPE_EDGE_STEP_MM
        ),

        "path_step_mm"

    )


    points_2d = [

        (
            0.0,
            0.0
        ),

        (
            size,
            0.0
        ),

        (
            size,
            size
        ),

        (
            0.0,
            size
        ),

        (
            0.0,
            0.0
        )

    ]


    vertices = [

        plane_vector(

            plane,

            u,

            v

        )

        for u, v
        in points_2d

    ]


    return polyline_to_waypoints(

        vertices,

        step_mm=
            step_mm,

        closed=
            False,

        label=
            "Square"

    )


# ============================================================
# RECTANGLE WAYPOINTS
# ============================================================

def generate_rectangle_waypoints(
    task
):

    plane = normalize_plane(

        task.get(
            "plane",
            "XY"
        )

    )


    width = positive_number(

        task.get(
            "width"
        ),

        "rectangle width"

    )


    height = positive_number(

        task.get(
            "height"
        ),

        "rectangle height"

    )


    step_mm = positive_number(

        task.get(
            "path_step_mm",
            DEFAULT_SHAPE_EDGE_STEP_MM
        ),

        "path_step_mm"

    )


    points_2d = [

        (
            0.0,
            0.0
        ),

        (
            width,
            0.0
        ),

        (
            width,
            height
        ),

        (
            0.0,
            height
        ),

        (
            0.0,
            0.0
        )

    ]


    vertices = [

        plane_vector(

            plane,

            u,

            v

        )

        for u, v
        in points_2d

    ]


    return polyline_to_waypoints(

        vertices,

        step_mm=
            step_mm,

        label=
            "Rectangle"

    )


# ============================================================
# EQUILATERAL TRIANGLE WAYPOINTS
# ============================================================

def generate_triangle_waypoints(
    task
):

    plane = normalize_plane(

        task.get(
            "plane",
            "XY"
        )

    )


    size = positive_number(

        task.get(
            "size"
        ),

        "triangle size"

    )


    step_mm = positive_number(

        task.get(
            "path_step_mm",
            DEFAULT_SHAPE_EDGE_STEP_MM
        ),

        "path_step_mm"

    )


    height = (

        math.sqrt(
            3.0
        )

        /
        2.0

        *
        size

    )


    points_2d = [

        (
            0.0,
            0.0
        ),

        (
            size,
            0.0
        ),

        (
            size / 2.0,
            height
        ),

        (
            0.0,
            0.0
        )

    ]


    vertices = [

        plane_vector(

            plane,

            u,

            v

        )

        for u, v
        in points_2d

    ]


    return polyline_to_waypoints(

        vertices,

        step_mm=
            step_mm,

        label=
            "Triangle"

    )


# ============================================================
# CIRCLE WAYPOINTS
#
# Başlangıç noktası:
#
# program TCP'si = circle üzerinde angle 0.
#
# Circle center relative olarak:
#
# (-radius, 0)
#
# kabul edilir.
#
# Böylece path ilk ve son noktada:
#
# [0,0,0]
#
# olur.
#
# ============================================================

def generate_circle_waypoints(
    task
):

    plane = normalize_plane(

        task.get(
            "plane",
            "XY"
        )

    )


    radius = positive_number(

        task.get(
            "radius"
        ),

        "circle radius"

    )


    segments = task.get(

        "segments",

        DEFAULT_CIRCLE_SEGMENTS

    )


    try:

        segments = int(
            segments
        )

    except (
        TypeError,
        ValueError
    ):

        raise PathGeneratorError(

            "Circle segments integer olmalı."

        )


    segments = max(

        MIN_CIRCLE_SEGMENTS,

        min(

            MAX_CIRCLE_SEGMENTS,

            segments

        )

    )


    waypoints = []


    for index in range(
        segments + 1
    ):

        progress = (

            index

            /
            segments

        )


        theta = (

            2.0

            *
            math.pi

            *
            progress

        )


        # Circle başlangıcı origin olsun:
        #
        # u(0) = 0
        # v(0) = 0
        #
        # bir tam tur sonunda da tekrar 0.

        u = (

            radius

            *
            (
                math.cos(
                    theta
                )
                -
                1.0
            )

        )


        v = (

            radius

            *
            math.sin(
                theta
            )

        )


        position = plane_vector(

            plane,

            u,

            v

        )


        waypoints.append(

            create_waypoint(

                position,

                progress,

                label=
                    "Circle"

            )

        )


    return waypoints


# ============================================================
# LINE WAYPOINTS
# ============================================================

def generate_line_waypoints(
    task
):

    vector = task.get(
        "vector"
    )


    if (
        not isinstance(
            vector,
            (
                list,
                tuple
            )
        )
        or
        len(
            vector
        )
        !=
        3
    ):

        # Alternative:
        #
        # plane + du + dv

        plane = normalize_plane(

            task.get(
                "plane",
                "XY"
            )

        )


        du = finite_number(

            task.get(
                "du",
                0.0
            ),

            "du"

        )


        dv = finite_number(

            task.get(
                "dv",
                0.0
            ),

            "dv"

        )


        vector = plane_vector(

            plane,

            du,

            dv

        )


    vector = [

        finite_number(
            vector[0],
            "dx"
        ),

        finite_number(
            vector[1],
            "dy"
        ),

        finite_number(
            vector[2],
            "dz"
        )

    ]


    distance = vector_length(
        vector
    )


    if distance <= 1e-12:

        raise PathGeneratorError(

            "Line vector uzunluğu sıfır."

        )


    step_mm = positive_number(

        task.get(
            "path_step_mm",
            DEFAULT_SHAPE_EDGE_STEP_MM
        ),

        "path_step_mm"

    )


    segments = max(

        1,

        int(
            math.ceil(

                distance

                /
                step_mm

            )
        )

    )


    return interpolate_line(

        [
            0.0,
            0.0,
            0.0
        ],

        vector,

        segments,

        progress_start=
            0.0,

        progress_end=
            1.0,

        label=
            "Line"

    )


# ============================================================
# SHAPE DISPATCH
# ============================================================

def generate_shape_waypoints(
    task
):

    shape = normalize_text(

        task.get(
            "shape"
        )

    )


    if shape == "SQUARE":

        return generate_square_waypoints(
            task
        )


    if shape == "RECTANGLE":

        return generate_rectangle_waypoints(
            task
        )


    if shape in (

        "TRIANGLE",

        "EQUILATERAL_TRIANGLE"

    ):

        return generate_triangle_waypoints(
            task
        )


    if shape == "CIRCLE":

        return generate_circle_waypoints(
            task
        )


    if shape == "LINE":

        return generate_line_waypoints(
            task
        )


    raise PathGeneratorError(

        f"Desteklenmeyen shape: {shape}"

    )


# ============================================================
# DRAW SHAPE
# ============================================================

def generate_draw_shape(
    task
):

    waypoints = (
        generate_shape_waypoints(
            task
        )
    )


    modifiers = task.get(

        "modifiers",

        []

    )


    waypoints = apply_modifiers(

        waypoints,

        modifiers

    )


    return [

        follow_path_command(

            waypoints,

            label=
                (
                    f"Draw "
                    f"{normalize_text(task.get('shape'))}"
                )

        )

    ]


# ============================================================
# MOVE RELATIVE
#
# Bu hâlâ basit Director MOVE_CARTESIAN üretebilir.
#
# ============================================================

def generate_move_relative(
    task
):

    if (
        task.get(
            "vector"
        )
        is not None
    ):

        vector = task[
            "vector"
        ]


        if (
            not isinstance(
                vector,
                (
                    list,
                    tuple
                )
            )
            or
            len(
                vector
            )
            !=
            3
        ):

            raise PathGeneratorError(

                (
                    "MOVE_RELATIVE vector "
                    "[dx,dy,dz] biçiminde olmalı."
                )

            )


        delta = [

            finite_number(
                vector[0],
                "dx"
            ),

            finite_number(
                vector[1],
                "dy"
            ),

            finite_number(
                vector[2],
                "dz"
            )

        ]


    else:

        axis = normalize_axis(

            task.get(
                "axis"
            )

        )


        distance = finite_number(

            task.get(
                "distance"
            ),

            "distance"

        )


        delta = axis_vector(

            axis,

            distance

        )


    return [

        {

            "type":
                "MOVE_CARTESIAN",

            "delta":
                delta,

            "orientation_mode":
                "KEEP",

            "label":
                "Relative Move"

        }

    ]


# ============================================================
# ROTATE RELATIVE
# ============================================================

def generate_rotate_relative(
    task
):

    axis = normalize_axis(

        task.get(
            "axis"
        )

    )


    angle = finite_number(

        task.get(
            "angle"
        ),

        "angle"

    )


    return [

        {

            "type":
                "ROTATE_TCP",

            "axis":
                axis,

            "value":
                angle,

            "label":
                f"Rotate {axis}"

        }

    ]


# ============================================================
# RETURN COMMANDS
# ============================================================

def return_tcp_to_start():

    return {

        "type":
            "RETURN_TCP_TO_START",

        "label":
            "Return TCP to Start"

    }


def return_joints_to_start():

    return {

        "type":
            "RETURN_JOINTS_TO_START",

        "label":
            "Return Joints to Start"

    }


# ============================================================
# SINGLE TASK COMPILER
# ============================================================

def compile_task(
    task
):

    if not isinstance(
        task,
        dict
    ):

        raise PathGeneratorError(

            "Her task dictionary olmalı."

        )


    action = normalize_text(

        task.get(
            "action"
        )

    )


    if not action:

        raise PathGeneratorError(

            "Task action eksik."

        )


    # ========================================================
    # SHAPE
    # ========================================================

    if action == "DRAW_SHAPE":

        return generate_draw_shape(
            task
        )


    # Aliases

    if action == "DRAW_SQUARE":

        task_copy = deepcopy(
            task
        )


        task_copy[
            "shape"
        ] = "SQUARE"


        return generate_draw_shape(
            task_copy
        )


    if action == "DRAW_RECTANGLE":

        task_copy = deepcopy(
            task
        )


        task_copy[
            "shape"
        ] = "RECTANGLE"


        return generate_draw_shape(
            task_copy
        )


    if action == "DRAW_TRIANGLE":

        task_copy = deepcopy(
            task
        )


        task_copy[
            "shape"
        ] = "TRIANGLE"


        return generate_draw_shape(
            task_copy
        )


    if action == "DRAW_CIRCLE":

        task_copy = deepcopy(
            task
        )


        task_copy[
            "shape"
        ] = "CIRCLE"


        return generate_draw_shape(
            task_copy
        )


    if action == "DRAW_LINE":

        task_copy = deepcopy(
            task
        )


        task_copy[
            "shape"
        ] = "LINE"


        return generate_draw_shape(
            task_copy
        )


    # ========================================================
    # BASIC MOVEMENT
    # ========================================================

    if action == "MOVE_RELATIVE":

        return generate_move_relative(
            task
        )


    if action == "ROTATE_RELATIVE":

        return generate_rotate_relative(
            task
        )


    # ========================================================
    # RETURN
    # ========================================================

    if action == "RETURN_TO_START":

        mode = normalize_text(

            task.get(
                "mode",
                "TCP"
            )

        )


        if mode in (

            "TCP",

            "POSITION",

            "POSE"

        ):

            return [
                return_tcp_to_start()
            ]


        if mode in (

            "JOINT",

            "JOINTS",

            "CONFIGURATION"

        ):

            return [
                return_joints_to_start()
            ]


        raise PathGeneratorError(

            "RETURN_TO_START mode TCP veya JOINTS olmalı."

        )


    if action == "RETURN_TCP_TO_START":

        return [
            return_tcp_to_start()
        ]


    if action == "RETURN_JOINTS_TO_START":

        return [
            return_joints_to_start()
        ]


    raise PathGeneratorError(

        (
            "Desteklenmeyen action: "
            f"{action}"
        )

    )


# ============================================================
# MAIN PROGRAM GENERATOR
# ============================================================

def generate_path_program(
    task_ir
):

    if not isinstance(
        task_ir,
        dict
    ):

        raise PathGeneratorError(

            "Task IR dictionary olmalı."

        )


    intent = normalize_text(

        task_ir.get(
            "intent"
        )

    )


    # ========================================================
    # MULTI TASK
    # ========================================================

    if intent == "TASK_SEQUENCE":

        steps = task_ir.get(
            "steps"
        )


        if (
            not isinstance(
                steps,
                list
            )
            or
            len(
                steps
            )
            ==
            0
        ):

            raise PathGeneratorError(

                (
                    "TASK_SEQUENCE en az "
                    "bir step içermeli."
                )

            )


    # ========================================================
    # SINGLE TASK
    # ========================================================

    else:

        if (
            "action"
            in task_ir
        ):

            steps = [
                task_ir
            ]


        elif intent:

            single_task = deepcopy(
                task_ir
            )


            single_task[
                "action"
            ] = intent


            steps = [
                single_task
            ]


        else:

            raise PathGeneratorError(

                (
                    "Task IR içinde intent "
                    "veya action bulunmalı."
                )

            )


    # ========================================================
    # COMPILE ALL TASKS
    # ========================================================

    commands = []


    task_summary = []


    for task_index, task in enumerate(
        steps
    ):

        generated = (
            compile_task(
                task
            )
        )


        commands.extend(
            generated
        )


        task_summary.append({

            "task_index":
                int(
                    task_index
                ),

            "action":
                normalize_text(
                    task.get(
                        "action"
                    )
                ),

            "generated_command_count":
                len(
                    generated
                )

        })


    return {

        "success":
            True,

        "task_count":
            len(
                steps
            ),

        "command_count":
            len(
                commands
            ),

        "tasks":
            task_summary,

        "commands":
            commands

    }