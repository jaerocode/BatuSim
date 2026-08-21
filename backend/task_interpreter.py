import re


# ============================================================
# TASK INTERPRETER
#
# Natural language
#      ↓
# Task IR
#
# Bu dosya şimdilik deterministic parser.
#
# Daha sonra LLM aynı Task IR formatını üretecek.
# ============================================================


class TaskInterpreterError(
    ValueError
):
    pass


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    text
):

    if text is None:

        return ""


    text = str(
        text
    )


    text = (
        text
        .strip()
        .lower()
    )


    replacements = {

        "ç":
            "c",

        "ğ":
            "g",

        "ı":
            "i",

        "ö":
            "o",

        "ş":
            "s",

        "ü":
            "u"

    }


    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )


    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text


# ============================================================
# NUMBER EXTRACTION
# ============================================================

def extract_numbers(
    text
):

    matches = re.findall(

        r"[-+]?\d+(?:[.,]\d+)?",

        text

    )


    numbers = []


    for item in matches:

        item = item.replace(
            ",",
            "."
        )


        try:

            numbers.append(
                float(
                    item
                )
            )

        except ValueError:

            pass


    return numbers


# ============================================================
# UNIT DETECTION
# ============================================================

def detect_length_unit(
    text
):

    text = normalize_text(
        text
    )


    # ========================================================
    # MILLIMETER
    # ========================================================

    if (
        re.search(
            r"\bmm\b",
            text
        )
        or
        "milimetre"
        in text
    ):

        return (
            "mm",
            1.0
        )


    # ========================================================
    # CENTIMETER
    # ========================================================

    if (
        re.search(
            r"\bcm\b",
            text
        )
        or
        "santimetre"
        in text
    ):

        return (
            "cm",
            10.0
        )


    # ========================================================
    # METER
    # ========================================================

    if (
        re.search(
            r"\bm\b",
            text
        )
        or
        "metre"
        in text
    ):

        return (
            "m",
            1000.0
        )


    # ========================================================
    # DEFAULT
    # ========================================================

    return (
        "mm",
        1.0
    )

def convert_length_to_mm(
    value,
    text
):

    _, factor = (
        detect_length_unit(
            text
        )
    )


    return (
        float(
            value
        )
        *
        factor
    )


# ============================================================
# PLANE DETECTION
# ============================================================

def detect_plane(
    text
):

    text = normalize_text(
        text
    )


    patterns = [

        (
            [
                "xy",
                "x-y",
                "x y"
            ],

            "XY"

        ),

        (
            [
                "xz",
                "x-z",
                "x z"
            ],

            "XZ"

        ),

        (
            [
                "yz",
                "y-z",
                "y z"
            ],

            "YZ"

        )

    ]


    for aliases, result in patterns:

        for alias in aliases:

            if alias in text:

                return result


    # Kullanıcı plane söylemediyse
    # ilk sürümde XY varsayıyoruz.
    return "XY"


# ============================================================
# AXIS DETECTION
# ============================================================

def detect_axis(
    text
):

    text = normalize_text(
        text
    )


    if (
        "roll"
        in text
    ):

        return "ROLL"


    if (
        "pitch"
        in text
    ):

        return "PITCH"


    if (
        "yaw"
        in text
    ):

        return "YAW"


    if re.search(
        r"\bx\b",
        text
    ):

        return "X"


    if re.search(
        r"\by\b",
        text
    ):

        return "Y"


    if re.search(
        r"\bz\b",
        text
    ):

        return "Z"


    return None


# ============================================================
# RETURN-TO-START DETECTION
# ============================================================

def contains_return_to_start(
    text
):

    text = normalize_text(
        text
    )


    phrases = [

        "baslangic konumuna geri don",

        "baslangic noktasina geri don",

        "basladigin yere geri don",

        "basladigi yere geri don",

        "ilk konuma geri don",

        "ilk noktaya geri don",

        "geri baslangica don",

        "baslangica geri don",

        "return to start"

    ]


    return any(

        phrase in text

        for phrase in phrases

    )


def detect_return_mode(
    text
):

    text = normalize_text(
        text
    )


    # Explicit joint/configuration wording
    if (
        "joint"
        in text
        or
        "eklem"
        in text
        or
        "baslangic pozu"
        in text
        or
        "ilk poz"
        in text
        or
        "konfigurasyon"
        in text
    ):

        return "JOINTS"


    return "TCP"


# ============================================================
# SHAPE DETECTION
# ============================================================

def detect_shape(
    text
):

    text = normalize_text(
        text
    )


    if (
        "kare"
        in text
        or
        "square"
        in text
    ):

        return "SQUARE"


    if (
        "dikdortgen"
        in text
        or
        "rectangle"
        in text
    ):

        return "RECTANGLE"


    if (
        "ucgen"
        in text
        or
        "triangle"
        in text
    ):

        return "TRIANGLE"


    if (
        "daire"
        in text
        or
        "cember"
        in text
        or
        "circle"
        in text
    ):

        return "CIRCLE"


    if (
        "cizgi"
        in text
        or
        "line"
        in text
    ):

        return "LINE"


    return None


# ============================================================
# SHAPE TASK PARSERS
# ============================================================

def parse_square_task(
    text
):

    numbers = extract_numbers(
        text
    )


    if not numbers:

        raise TaskInterpreterError(
            "Karenin kenar uzunluğu belirtilmeli."
        )


    size = convert_length_to_mm(

        numbers[0],

        text

    )


    return {

        "action":
            "DRAW_SHAPE",

        "shape":
            "SQUARE",

        "plane":
            detect_plane(
                text
            ),

        "size":
            size,

        "reference":
            "CURRENT_TCP",

        "orientation_mode":
            "KEEP"

    }


def parse_rectangle_task(
    text
):

    numbers = extract_numbers(
        text
    )


    if (
        len(
            numbers
        )
        <
        2
    ):

        raise TaskInterpreterError(

            (
                "Dikdörtgen için width ve height "
                "belirtilmeli."
            )

        )


    width = convert_length_to_mm(

        numbers[0],

        text

    )


    height = convert_length_to_mm(

        numbers[1],

        text

    )


    return {

        "action":
            "DRAW_SHAPE",

        "shape":
            "RECTANGLE",

        "plane":
            detect_plane(
                text
            ),

        "width":
            width,

        "height":
            height,

        "reference":
            "CURRENT_TCP",

        "orientation_mode":
            "KEEP"

    }


def parse_triangle_task(
    text
):

    numbers = extract_numbers(
        text
    )


    if not numbers:

        raise TaskInterpreterError(

            "Üçgenin kenar uzunluğu belirtilmeli."

        )


    size = convert_length_to_mm(

        numbers[0],

        text

    )


    return {

        "action":
            "DRAW_SHAPE",

        "shape":
            "TRIANGLE",

        "plane":
            detect_plane(
                text
            ),

        "size":
            size,

        "reference":
            "CURRENT_TCP",

        "orientation_mode":
            "KEEP"

    }


def parse_circle_task(
    text
):

    numbers = extract_numbers(
        text
    )


    if not numbers:

        raise TaskInterpreterError(

            "Dairenin yarıçapı belirtilmeli."

        )


    radius = convert_length_to_mm(

        numbers[0],

        text

    )


    return {

        "action":
            "DRAW_SHAPE",

        "shape":
            "CIRCLE",

        "plane":
            detect_plane(
                text
            ),

        "radius":
            radius,

        "segments":
            48,

        "reference":
            "CURRENT_TCP",

        "orientation_mode":
            "KEEP"

    }


# ============================================================
# MOVE RELATIVE PARSER
# ============================================================

def parse_move_relative_task(
    text
):

    axis = detect_axis(
        text
    )


    if (
        axis
        is None
        or
        axis
        in (
            "ROLL",
            "PITCH",
            "YAW"
        )
    ):

        raise TaskInterpreterError(

            "Lineer hareket için X, Y veya Z ekseni gerekli."

        )


    numbers = extract_numbers(
        text
    )


    if not numbers:

        raise TaskInterpreterError(

            "Hareket mesafesi belirtilmeli."

        )


    distance = convert_length_to_mm(

        numbers[0],

        text

    )


    text_normalized = normalize_text(
        text
    )


    negative_words = [

        "geri",

        "eksi",

        "negatif",

        "azalt",

        "asagi",

        "sola"

    ]


    if any(

        word
        in text_normalized

        for word
        in negative_words

    ):

        distance = (
            -abs(
                distance
            )
        )


    return {

        "action":
            "MOVE_RELATIVE",

        "axis":
            axis,

        "distance":
            distance

    }


# ============================================================
# ROTATE RELATIVE PARSER
# ============================================================

def parse_rotate_relative_task(
    text
):

    axis = detect_axis(
        text
    )


    if (
        axis
        not in (
            "ROLL",
            "PITCH",
            "YAW",
            "X",
            "Y",
            "Z"
        )
    ):

        raise TaskInterpreterError(

            "Dönüş için Roll, Pitch veya Yaw ekseni gerekli."

        )


    numbers = extract_numbers(
        text
    )


    if not numbers:

        raise TaskInterpreterError(

            "Dönüş açısı belirtilmeli."

        )


    angle = float(
        numbers[0]
    )


    text_normalized = normalize_text(
        text
    )


    negative_words = [

        "eksi",

        "negatif",

        "ters",

        "saat yonunun tersine"

    ]


    if any(

        word
        in text_normalized

        for word
        in negative_words

    ):

        angle = (
            -abs(
                angle
            )
        )


    return {

        "action":
            "ROTATE_RELATIVE",

        "axis":
            axis,

        "angle":
            angle

    }


# ============================================================
# SINGLE PRIMARY TASK
# ============================================================

def parse_primary_task(
    text
):

    normalized = normalize_text(
        text
    )


    shape = detect_shape(
        normalized
    )


    # ========================================================
    # SHAPES
    # ========================================================

    if shape == "SQUARE":

        return parse_square_task(
            normalized
        )


    if shape == "RECTANGLE":

        return parse_rectangle_task(
            normalized
        )


    if shape == "TRIANGLE":

        return parse_triangle_task(
            normalized
        )


    if shape == "CIRCLE":

        return parse_circle_task(
            normalized
        )


    # ========================================================
    # ROTATION
    # ========================================================

    if any(

        keyword
        in normalized

        for keyword
        in (
            "roll",
            "pitch",
            "yaw",
            "dondur",
            "don"
        )

    ):

        return parse_rotate_relative_task(
            normalized
        )


    # ========================================================
    # MOVE
    # ========================================================

    if any(

        keyword
        in normalized

        for keyword
        in (
            "git",
            "ilerle",
            "hareket et",
            "kay",
            "cik",
            "in"
        )

    ):

        return parse_move_relative_task(
            normalized
        )


    raise TaskInterpreterError(

        "İstek şu an desteklenen bir robotik göreve çevrilemedi."

    )


# ============================================================
# NATURAL LANGUAGE → TASK IR
# ============================================================

def interpret_task_text(
    text
):

    if (
        text is None
        or
        not str(
            text
        ).strip()
    ):

        raise TaskInterpreterError(

            "Komut metni boş olamaz."

        )


    normalized = normalize_text(
        text
    )


    steps = []


    # ========================================================
    # PRIMARY TASK
    # ========================================================

    primary_task = (
        parse_primary_task(
            normalized
        )
    )


    steps.append(
        primary_task
    )


    # ========================================================
    # RETURN TO START
    # ========================================================

    if contains_return_to_start(
        normalized
    ):

        steps.append({

            "action":
                "RETURN_TO_START",

            "mode":
                detect_return_mode(
                    normalized
                )

        })


    # ========================================================
    # TASK IR
    # ========================================================

    if (
        len(
            steps
        )
        ==
        1
    ):

        return {

            "intent":
                steps[0][
                    "action"
                ],

            **steps[0]

        }


    return {

        "intent":
            "TASK_SEQUENCE",

        "steps":
            steps

    }


# ============================================================
# HIGH-LEVEL RESPONSE
# ============================================================

def interpret_task(
    text
):

    task_ir = (
        interpret_task_text(
            text
        )
    )


    return {

        "success":
            True,

        "input":
            text,

        "task_ir":
            task_ir

    }