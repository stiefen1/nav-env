from nav_env.obstacles.obstacles import ObstacleWithKinematics

def get_relative_position_and_speed(own_ship: ObstacleWithKinematics, target_ship: ObstacleWithKinematics) -> tuple:
    """
    Get the relative position and speed of the target ship with respect to the own ship.
    """
    # Get (x, y) speed of own ship and target ship
    vx_os = own_ship.states.x_dot
    vy_os = own_ship.states.y_dot
    vx_ts = target_ship.states.x_dot
    vy_ts = target_ship.states.y_dot

    # Get (x, y) position of own ship and target ship
    px_os = own_ship.states.x
    py_os = own_ship.states.y
    px_ts = target_ship.states.x
    py_ts = target_ship.states.y

    # Relative position
    px_rel = px_ts - px_os
    py_rel = py_ts - py_os

    # Relative speed
    vx_rel = vx_ts - vx_os
    vy_rel = vy_ts - vy_os

    return px_rel, py_rel, vx_rel, vy_rel