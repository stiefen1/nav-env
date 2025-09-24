from nav_env.ships.ship import Ship
from nav_env.ships.states import States3
from nav_env.control.path import Waypoints as WP
from nav_env.control.path import TimeStampedWaypoints as TSWP
from nav_env.control.PID import HeadingAndSpeedController
from nav_env.control.allocation import PowerMinimizerControlAllocation
from nav_env.actuators.collection import ActuatorCollection
from nav_env.actuators.actuators import AzimuthThruster
from nav_env.environment.environment import NavigationEnvironment as NavEnv
import matplotlib.pyplot as plt, numpy as np


LOA = 75
BEAM = 17.2
dt = 1.0
H = 100
lim = ((-H, -H), (H, H))

def speed_controller_final_results() -> None:
    from nav_env.control.guidance import ConstantHeadingAndSpeed

    actuators = ActuatorCollection([
        AzimuthThruster(
            (33, 0), 0, (-90, 90), (0, 300), dt, alpha_rate_max=12, v_rate_max=60
        ),
        AzimuthThruster(
            (-33, 0), 0, (-90, 90), (0, 300), dt, alpha_rate_max=12, v_rate_max=60
        )
    ])

    # Target speed
    target_speed = 5.0  # m/s

    ship = Ship(
        states=States3(),
        guidance=ConstantHeadingAndSpeed(
            desired_heading_deg=0,  # Keep heading constant
            desired_speed=target_speed
        ),
        controller=HeadingAndSpeedController(
            pid_gains_heading=(1e6, 0, 1e7),  # Light heading control to maintain course
            pid_gains_speed=(1e5, 1e4, 1e5),  # Main focus on speed control
            dt=dt,
            allocation=PowerMinimizerControlAllocation(actuators, Q=np.eye(6), W=np.eye(4))
        ),
        actuators=actuators,
        length=LOA, 
        width=BEAM
    )
    
    env = NavEnv(own_ships=[ship])

    # Data storage for simulation
    time_data = []
    actual_speed_data = []
    desired_speed_data = []
    speed_error_data = []
    control_effort_data = []
    position_x_data = []
    position_y_data = []

    tf = 300  # Longer simulation for speed response
    print("Running speed controller simulation...")
    
    # Run the simulation without plotting
    for t in np.linspace(0, tf, int(tf//dt)):
        env.step()
        
        # Get data
        actual_speed = ship.states.speed
        desired_speed = ship._gnc._guidance.desired_speed
        speed_error = desired_speed - actual_speed
        
        # Get control effort (total thrust force)
        control_effort = np.sqrt(ship._gnc._controller.tau_surge**2 + ship._gnc._controller.tau_sway**2) if hasattr(ship._gnc._controller, 'tau_surge') else 0
        
        # Store data
        time_data.append(t)
        actual_speed_data.append(actual_speed)
        desired_speed_data.append(desired_speed)
        speed_error_data.append(speed_error)
        control_effort_data.append(control_effort)
        position_x_data.append(ship.states.x)
        position_y_data.append(ship.states.y)

    print("Simulation complete. Plotting results...")

    # Create final results plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Speed tracking
    ax1.plot(time_data, desired_speed_data, 'r--', label='Desired Speed', linewidth=2)
    ax1.plot(time_data, actual_speed_data, 'b-', label='Actual Speed', linewidth=2)
    ax1.set_title("Speed Tracking Performance")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Speed (m/s)")
    ax1.legend()
    ax1.grid(True)

    # 2. Speed error
    ax2.plot(time_data, speed_error_data, 'g-', linewidth=2)
    ax2.set_title("Speed Error")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error (m/s)")
    ax2.grid(True)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # 3. Ship trajectory
    ax3.plot(position_x_data, position_y_data, 'b-', linewidth=2)
    ax3.plot(position_x_data[0], position_y_data[0], 'go', markersize=10, label='Start')
    ax3.plot(position_x_data[-1], position_y_data[-1], 'ro', markersize=10, label='End')
    ax3.set_title("Ship Trajectory")
    ax3.set_xlabel("X Position (m)")
    ax3.set_ylabel("Y Position (m)")
    ax3.legend()
    ax3.grid(True)
    ax3.set_aspect('equal')

    # 4. Control effort
    ax4.plot(time_data, control_effort_data, 'm-', linewidth=2)
    ax4.set_title("Control Effort (Total Thrust)")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Thrust Force (N)")
    ax4.grid(True)

    plt.tight_layout()

    # Print performance metrics
    final_error = abs(speed_error_data[-1])
    max_error = max(abs(e) for e in speed_error_data)
    
    # Calculate settling time (time to reach within 2% of target speed)
    target_band = 0.1  # m/s (2% of 5 m/s)
    settling_time = None
    steady_state_start = len(time_data) // 2  # Check from halfway point
    
    for i in range(steady_state_start, len(speed_error_data)):
        if abs(speed_error_data[i]) <= target_band:
            settling_time = time_data[i]
            break
    
    # Calculate overshoot
    max_speed = max(actual_speed_data)
    overshoot_percent = ((max_speed - target_speed) / target_speed) * 100 if max_speed > target_speed else 0
    
    # Calculate steady-state performance
    steady_state_speeds = actual_speed_data[-50:]  # Last 50 data points
    steady_state_mean = np.mean(steady_state_speeds)
    steady_state_std = np.std(steady_state_speeds)
    
    print(f"\n=== Speed Controller Performance ===")
    print(f"Target speed: {target_speed:.2f} m/s")
    print(f"Final speed: {actual_speed_data[-1]:.2f} m/s")
    print(f"Final speed error: {final_error:.3f} m/s")
    print(f"Maximum error: {max_error:.3f} m/s")
    print(f"Overshoot: {overshoot_percent:.1f}%")
    print(f"Settling time (±{target_band} m/s): {settling_time:.1f}s" if settling_time else f"Did not settle within ±{target_band} m/s")
    print(f"Steady-state mean: {steady_state_mean:.3f} ± {steady_state_std:.3f} m/s")
    print(f"Distance traveled: {np.sqrt(position_x_data[-1]**2 + position_y_data[-1]**2):.1f} m")
    print(f"PID gains: Kp={1e5:.0e}, Ki={1e4:.0e}, Kd={1e5:.0e}")

    plt.show()

def speed_controller_step_response() -> None:
    """Test speed controller with step changes in desired speed."""
    from nav_env.control.guidance import ConstantHeadingAndSpeed

    actuators = ActuatorCollection([
        AzimuthThruster(
            (33, 0), 0, (-90, 90), (0, 300), dt, alpha_rate_max=12, v_rate_max=60
        ),
        AzimuthThruster(
            (-33, 0), 0, (-90, 90), (0, 300), dt, alpha_rate_max=12, v_rate_max=60
        )
    ])

    ship = Ship(
        states=States3(),
        guidance=ConstantHeadingAndSpeed(
            desired_heading_deg=0,
            desired_speed=0  # Start at rest
        ),
        controller=HeadingAndSpeedController(
            pid_gains_heading=(0, 0, 0),
            pid_gains_speed=(1e7, 1e4, 1e5),
            dt=dt,
            allocation=PowerMinimizerControlAllocation(actuators, Q=np.eye(6), W=np.eye(4))
        ),
        actuators=actuators,
        length=LOA, 
        width=BEAM
    )
    
    env = NavEnv(own_ships=[ship])

    # Data storage
    time_data = []
    actual_speed_data = []
    desired_speed_data = []

    tf = 400
    print("Running step response test...")
    
    # Run simulation with step changes
    for t in np.linspace(0, tf, int(tf//dt)):
        # Change desired speed at different times
        if t < 100:
            desired_speed = 0
        elif t < 200:
            desired_speed = 3.0
        elif t < 300:
            desired_speed = 6.0
        else:
            desired_speed = 1.0
            
        # Update guidance
        ship._gnc._guidance.desired_speed = desired_speed
        
        env.step()
        
        # Store data
        time_data.append(t)
        actual_speed_data.append(ship.states.u)
        desired_speed_data.append(desired_speed)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(time_data, desired_speed_data, 'r--', label='Desired Speed', linewidth=2)
    plt.plot(time_data, actual_speed_data, 'b-', label='Actual Speed', linewidth=2)
    plt.title("Speed Controller Step Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    plt.grid(True)
    
    # Add step annotations
    plt.axvline(x=100, color='k', linestyle=':', alpha=0.5)
    plt.axvline(x=200, color='k', linestyle=':', alpha=0.5)
    plt.axvline(x=300, color='k', linestyle=':', alpha=0.5)
    plt.text(50, 5, 'Rest', ha='center')
    plt.text(150, 5, 'Step to 3 m/s', ha='center')
    plt.text(250, 5, 'Step to 6 m/s', ha='center')
    plt.text(350, 5, 'Step to 1 m/s', ha='center')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # speed_controller_final_results()
    speed_controller_step_response()