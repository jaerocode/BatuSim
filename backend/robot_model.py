import re
import numpy as np
import sympy as sp


# ============================================================
# CONFIG
# ============================================================

MAX_JOINTS = 12
MAX_LINKS = 12

DEFAULT_LINK_VALUE = 50.0
DEFAULT_D_VALUE = 50.0
DEFAULT_JOINT_VALUE = 0.0


# ============================================================
# SYMBOLS
# ============================================================

q_symbols = sp.symbols(
    " ".join(
        f"q{i}"
        for i in range(1, MAX_JOINTS + 1)
    )
)

L_symbols = sp.symbols(
    " ".join(
        f"L{i}"
        for i in range(0, MAX_LINKS + 1)
    )
)

D_symbols = sp.symbols(
    " ".join(
        f"D{i}"
        for i in range(1, MAX_LINKS + 1)
    )
)

LT = sp.Symbol("LT")


# ============================================================
# SYMBOL DICTIONARY
# ============================================================

SYMBOLS = {}


for i, symbol in enumerate(q_symbols, start=1):
    SYMBOLS[f"q{i}"] = symbol


for i, symbol in enumerate(L_symbols):
    SYMBOLS[f"L{i}"] = symbol


for i, symbol in enumerate(D_symbols, start=1):
    SYMBOLS[f"D{i}"] = symbol


SYMBOLS["LT"] = LT


# ============================================================
# PARSER
# ============================================================

def parse_value(value):
    """
    String DH ifadelerini SymPy ifadesine çevirir.

    Örnek:
        "q1"       -> q1
        "L1"       -> L1
        "q2 - 90"  -> q2 - 90
        "L1 + L2"  -> L1 + L2
        90         -> 90
    """

    if isinstance(value, sp.Expr):
        return value

    if isinstance(value, (int, float)):
        return sp.sympify(value)

    if isinstance(value, str):

        value = value.strip()

        if value == "":
            return sp.Integer(0)

        return sp.sympify(
            value,
            locals=SYMBOLS
        )

    return sp.sympify(value)


# ============================================================
# STANDARD DH MATRIX
#
# Schilling / Standard DH:
#
# Rz(theta)
# Tz(d)
# Tx(a)
# Rx(alpha)
#
# Angles are entered in DEGREES.
# ============================================================

def dh_matrix(theta, d, a, alpha):

    theta = parse_value(theta)
    d = parse_value(d)
    a = parse_value(a)
    alpha = parse_value(alpha)

    theta_rad = theta * sp.pi / 180
    alpha_rad = alpha * sp.pi / 180

    c = sp.cos(theta_rad)
    s = sp.sin(theta_rad)

    ca = sp.cos(alpha_rad)
    sa = sp.sin(alpha_rad)

    return sp.Matrix([

        [c, -s * ca,  s * sa, a * c],

        [s,  c * ca, -c * sa, a * s],

        [0,       sa,      ca,     d],

        [0,        0,       0,     1]

    ])


# ============================================================
# FIND SYMBOLS USED BY DH TABLE
# ============================================================

def find_used_symbols(dh_table):
    """
    DH tablosunda gerçekten kullanılan sembolleri bulur.

    Örnek:
        q1, q2, L1, L2, LT
    """

    used = set()

    for row in dh_table:

        for field in (
            "theta",
            "d",
            "a",
            "alpha"
        ):

            expression = parse_value(
                row.get(field, 0)
            )

            used.update(
                expression.free_symbols
            )

    return used


# ============================================================
# DEFAULT VALUES
# ============================================================

def create_default_values(dh_table):
    """
    Kullanıcı henüz değer girmediyse:

        q1, q2... = 0
        L0, L1... = 50 mm
        D1, D2... = 50 mm
        LT         = 50 mm

    Yalnızca DH tablosunda kullanılan semboller oluşturulur.
    """

    used_symbols = find_used_symbols(
        dh_table
    )

    values = {}

    for symbol in used_symbols:

        name = str(symbol)

        if re.fullmatch(r"q\d+", name):

            values[name] = (
                DEFAULT_JOINT_VALUE
            )

        elif re.fullmatch(r"L\d+", name):

            values[name] = (
                DEFAULT_LINK_VALUE
            )

        elif re.fullmatch(r"D\d+", name):

            values[name] = (
                DEFAULT_D_VALUE
            )

        elif name == "LT":

            values[name] = (
                DEFAULT_LINK_VALUE
            )

    return values


# ============================================================
# BUILD SUBSTITUTION DICTIONARY
# ============================================================

def make_substitutions(values):

    substitutions = {}

    for name, value in values.items():

        if name not in SYMBOLS:
            continue

        substitutions[
            SYMBOLS[name]
        ] = float(value)

    return substitutions


# ============================================================
# SYMBOLIC FORWARD KINEMATICS
# ============================================================

def symbolic_forward_kinematics(dh_table):
    """
    T00, T01, T02 ... T0Tool matrislerini üretir.
    """

    T = sp.eye(4)

    frames = [T]

    for row in dh_table:

        A = dh_matrix(
            row.get("theta", 0),
            row.get("d", 0),
            row.get("a", 0),
            row.get("alpha", 0)
        )

        T = sp.simplify(
            T * A
        )

        frames.append(T)

    return frames


# ============================================================
# NUMERIC FORWARD KINEMATICS
# ============================================================

def numeric_forward_kinematics(
    dh_table,
    values=None
):
    """
    DH tablosunu sayısal olarak değerlendirir.

    values verilmezse default değerler kullanılır.

    Kullanıcı sadece bazı değerleri verirse:
        default values
             +
        user values

    birleştirilir.
    """

    symbolic_frames = (
        symbolic_forward_kinematics(
            dh_table
        )
    )

    # --------------------------------------------------------
    # Default değerleri oluştur
    # --------------------------------------------------------

    final_values = create_default_values(
        dh_table
    )

    # --------------------------------------------------------
    # Kullanıcı değerleri defaultların üzerine yazılır
    # --------------------------------------------------------

    if values:

        for name, value in values.items():

            if value is not None:

                final_values[name] = (
                    float(value)
                )

    substitutions = make_substitutions(
        final_values
    )

    numeric_frames = []

    for frame in symbolic_frames:

        evaluated = frame.subs(
            substitutions
        )

        # Hâlâ bilinmeyen sembol varsa hata
        if evaluated.free_symbols:

            raise ValueError(
                "Sayısal değeri bulunamayan "
                f"semboller: {evaluated.free_symbols}"
            )

        numeric_frame = np.array(
            evaluated.evalf(),
            dtype=float
        )

        numeric_frames.append(
            numeric_frame
        )

    return (
        numeric_frames,
        final_values
    )


# ============================================================
# FRAME -> JSON
# ============================================================

def frame_to_dict(T):
    """
    Numpy 4x4 frame'i browser'a gönderilebilir
    JSON yapısına dönüştürür.
    """

    T = np.asarray(
        T,
        dtype=float
    )

    return {

        "matrix":
            T.tolist(),

        "position":
            T[:3, 3].tolist(),

        "rotation":
            T[:3, :3].tolist()

    }


# ============================================================
# ROBOT -> JSON
# ============================================================

def robot_to_dict(
    dh_table,
    values=None
):

    numeric_frames, final_values = (
        numeric_forward_kinematics(
            dh_table,
            values
        )
    )

    frames = [

        frame_to_dict(T)

        for T in numeric_frames

    ]

    return {

        "frames":
            frames,

        "tcp":
            frames[-1],

        "values":
            final_values,

        "frame_count":
            len(frames)

    }