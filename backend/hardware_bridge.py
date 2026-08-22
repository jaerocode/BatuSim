import time

import serial


# ============================================================
# BATUSIM HARDWARE BRIDGE
# ============================================================

DEFAULT_BAUD_RATE = 115200


# ============================================================
# SERVO CALIBRATION
# ============================================================
#
# Şu an fiziksel robot ve BatuSim:
#
# q1 = 90  -> servo J1 = 90
# q2 = 90  -> servo J2 = 90
# q3 = 90  -> servo J3 = 90
#
# Aynı pozu verdiği için:
#
# servo_angle = q
#
# kullanıyoruz.
#
# Eğer ileride herhangi bir servo ters hareket ederse:
#
# offset    = 180
# direction = -1
#
# yapabiliriz.
# ============================================================

SERVO_CONFIG = {

    "q1": {
        "offset": 0.0,
        "direction": 1.0,
        "min": 0.0,
        "max": 180.0,
    },

    "q2": {
        "offset": 0.0,
        "direction": 1.0,
        "min": 0.0,
        "max": 180.0,
    },

    "q3": {
        "offset": 0.0,
        "direction": 1.0,
        "min": 0.0,
        "max": 180.0,
    },

}


# ============================================================
# CLAMP
# ============================================================

def clamp(
    value,
    minimum,
    maximum
):

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


# ============================================================
# ROBOT q -> SERVO ANGLES
# ============================================================

def q_to_servo_angles(
    q_vector
):

    if len(q_vector) != 3:

        raise ValueError(
            "3R robot için q_vector 3 elemanlı olmalıdır."
        )


    servo_angles = []


    for index, q in enumerate(
        q_vector,
        start=1
    ):

        config = SERVO_CONFIG[
            f"q{index}"
        ]


        servo_angle = (

            config["offset"]

            +

            config["direction"]

            *

            float(q)

        )


        servo_angle = clamp(

            servo_angle,

            config["min"],

            config["max"]

        )


        servo_angles.append(

            int(
                round(
                    servo_angle
                )
            )

        )


    return servo_angles


# ============================================================
# HARDWARE BRIDGE
# ============================================================

class ArduinoRobotBridge:

    def __init__(
        self,
        port,
        baud_rate=DEFAULT_BAUD_RATE,
        timeout=2.0
    ):

        self.port = port

        self.baud_rate = baud_rate

        self.timeout = timeout

        self.serial = None


    # ========================================================
    # CONNECT
    # ========================================================

    def connect(
        self
    ):

        if (
            self.serial is not None
            and
            self.serial.is_open
        ):

            return


        print(
            f"Opening Arduino port: {self.port}"
        )


        self.serial = serial.Serial(

            port=self.port,

            baudrate=self.baud_rate,

            timeout=0.2

        )


        # ====================================================
        # Arduino Uno / Nano Serial açılınca reset olabilir.
        #
        # Setup() tekrar çalışır.
        # Servo firmware'imiz BATUSIM_READY gönderebilir.
        # ====================================================

        time.sleep(
            2.0
        )


        print(
            f"Arduino connected: {self.port}"
        )


        # ====================================================
        # BAŞLANGIÇ BUFFER'INI TEMİZLE
        #
        # BATUSIM_READY gibi eski startup mesajları
        # ilk trajectory ACK'i ile karışmasın.
        # ====================================================

        self._drain_startup_messages()


    # ========================================================
    # DRAIN STARTUP MESSAGES
    # ========================================================

    def _drain_startup_messages(
        self
    ):

        if (
            self.serial is None
            or
            not self.serial.is_open
        ):

            return


        time.sleep(
            0.1
        )


        while (
            self.serial.in_waiting
            >
            0
        ):

            line = (

                self.serial
                .readline()
                .decode(
                    "utf-8",
                    errors="ignore"
                )
                .strip()

            )


            if line:

                print(
                    "Arduino startup:",
                    line
                )


        # Ardından kalan byte'ları da temizle.

        self.serial.reset_input_buffer()


    # ========================================================
    # DISCONNECT
    # ========================================================

    def disconnect(
        self
    ):

        if (
            self.serial is not None
            and
            self.serial.is_open
        ):

            self.serial.close()


        self.serial = None


        print(
            "Arduino disconnected."
        )


    # ========================================================
    # WAIT FOR ACK
    # ========================================================

    def _wait_for_ack(
        self
    ):

        start_time = time.time()


        while True:

            elapsed = (

                time.time()

                -
                start_time

            )


            if (
                elapsed
                >
                self.timeout
            ):

                raise RuntimeError(
                    "Arduino ACK timeout."
                )


            line = (

                self.serial
                .readline()
                .decode(
                    "utf-8",
                    errors="ignore"
                )
                .strip()

            )


            if not line:

                continue


            print(
                "Arduino:",
                line
            )


            # =================================================
            # EXPECTED RESPONSE
            # =================================================

            if line.startswith(
                "ACK,"
            ):

                return line


            # =================================================
            # ARDUINO ERROR
            # =================================================

            if line.startswith(
                "ERR,"
            ):

                raise RuntimeError(

                    f"Arduino error: {line}"

                )


            # =================================================
            # OTHER MESSAGES
            #
            # Örneğin:
            #
            # BATUSIM_READY
            #
            # Bunları hata olarak görmüyoruz.
            # =================================================


    # ========================================================
    # SEND SERVO ANGLES
    # ========================================================

    def send_servo_angles(
        self,
        angles
    ):

        if (
            self.serial is None
            or
            not self.serial.is_open
        ):

            raise RuntimeError(
                "Arduino bağlı değil."
            )


        if len(angles) != 3:

            raise ValueError(
                "3 servo açısı gerekli."
            )


        # ====================================================
        # COMMAND
        #
        # Example:
        #
        # Q,90,100,90
        # ====================================================

        command = (

            f"Q,"
            f"{angles[0]},"
            f"{angles[1]},"
            f"{angles[2]}\n"

        )


        # ====================================================
        # SEND
        # ====================================================

        self.serial.write(

            command.encode(
                "utf-8"
            )

        )


        self.serial.flush()


        # ====================================================
        # WAIT FOR CORRECT ACK
        # ====================================================

        response = self._wait_for_ack()


        return response


    # ========================================================
    # SEND ROBOT q
    # ========================================================

    def send_q(
        self,
        q_vector
    ):

        servo_angles = q_to_servo_angles(
            q_vector
        )


        response = self.send_servo_angles(
            servo_angles
        )


        return {

            "q":
                list(
                    q_vector
                ),

            "servo_angles":
                servo_angles,

            "response":
                response,

        }


    # ========================================================
    # STREAM TRAJECTORY
    # ========================================================

    def stream_trajectory(
        self,
        trajectory,
        point_delay=0.04
    ):

        if not trajectory:

            raise ValueError(
                "Trajectory boş."
            )


        print()

        print(
            "========================================"
        )

        print(
            "Trajectory streaming started."
        )

        print(
            "Points:",
            len(
                trajectory
            )
        )

        print(
            "========================================"
        )

        print()


        for index, point in enumerate(
            trajectory
        ):

            # =================================================
            # BATUsim trajectory:
            #
            # {
            #     "q_vector": [...]
            # }
            #
            # veya direkt:
            #
            # [90, 90, 90]
            # =================================================

            if isinstance(
                point,
                dict
            ):

                if (
                    "q_vector"
                    not in point
                ):

                    raise ValueError(

                        (
                            f"Trajectory point {index} "
                            "q_vector içermiyor."
                        )

                    )


                q_vector = point[
                    "q_vector"
                ]


            else:

                q_vector = point


            # =================================================
            # SEND
            # =================================================

            result = self.send_q(
                q_vector
            )


            # =================================================
            # LOG
            # =================================================

            print(

                f"{index + 1:04d}/"
                f"{len(trajectory):04d}",

                "q =",
                result[
                    "q"
                ],

                "servo =",
                result[
                    "servo_angles"
                ],

                "response =",
                result[
                    "response"
                ]

            )


            # =================================================
            # WAIT
            # =================================================

            time.sleep(
                point_delay
            )


        print()

        print(
            "========================================"
        )

        print(
            "Trajectory streaming complete."
        )

        print(
            "========================================"
        )