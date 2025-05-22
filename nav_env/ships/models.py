"""
Example of existing ships, built by combining actuators, sensors, etc..

There should be two ways for building a ship model: json files or programmaticaly using
Base classes. Example:

my_custom_ship = Ship(
                Actuators=[
                    Thruster(
                        xy=...
                        ),
                    Rudder(
                        config='my_rudder_config.json'
                        ),
                    AzimuthThruster(
                        xy=...
                        )
                    ],
                Sensors=[
                    GNSS(
                        sigma=...
                    ),
                    LiDAR(
                        config='my_lidar_config.json'
                    )
                ]
)

Ship.pose_estimation(Kalman(Ship.GNSS, ...)) --> Pas sûr de ça.......

"""