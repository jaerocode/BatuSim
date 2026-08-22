from backend.hardware_bridge import (
    ArduinoRobotBridge,
)


bridge = ArduinoRobotBridge(
    port="COM4"
)


bridge.connect()


try:

    trajectory = [

        # HOME
        [90, 90, 90],

        # J1 TEST
        [100, 90, 90],
        [90, 90, 90],

        # J2 TEST
        [90, 100, 90],
        [90, 90, 90],

        # J3 TEST
        [90, 90, 100],
        [90, 90, 90],

    ]


    bridge.stream_trajectory(

        trajectory,

        # İlk testte özellikle yavaş.
        point_delay=2.0

    )


finally:

    bridge.disconnect()