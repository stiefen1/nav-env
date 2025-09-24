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
Q_P = np.eye(6)*1e-3
Q_P[0, 0] = 2e1
Q_P[1, 1] = 2e1 # 1e1
W_P = np.eye(4)*1e-2


def heading_controller() -> None:
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
                desired_heading_deg=45,
                desired_speed=0
            ),
            controller=HeadingAndSpeedController(
                pid_gains_heading=(3e7, 0, 5e9), #(1e8, 0, 1e10),
                pid_gains_speed=(0, 0, 0),
                dt=dt,
                allocation=PowerMinimizerControlAllocation(actuators, Q=Q_P, W=W_P) #, Q=Q, W=np.eye(4)*10)
            ),
            actuators=actuators,
            length=LOA, 
            width=BEAM
        )
    

    env = NavEnv(own_ships=[ship])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    env.plot(lim, ax=ax1)
    ax1.set_title("Ship Environment")
    ax1.set_aspect('equal')
    
    # Initialize heading plot
    ax2.set_title("Heading Control")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Heading (deg)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show(block=False)

    # Data storage for plotting
    time_data = []
    actual_heading_data = []
    desired_heading_data = []

    tf = 1000
    for t in np.linspace(0, tf, int(tf//dt)):
        # Update environment
        env.step()
        
        # Get heading data
        actual_heading = ship.states.psi_deg
        desired_heading = ship._gnc._guidance.desired_heading_deg
        
        # Store data
        time_data.append(t)
        actual_heading_data.append(actual_heading)
        desired_heading_data.append(desired_heading)
        
        # Update environment plot
        ax1.cla()
        ax1.set_title(f"Ship Environment - t={t:.1f}s")
        env.plot(lim, ax=ax1)
        ax1.set_aspect('equal')
        
        # Update heading plot
        ax2.cla()
        ax2.plot(time_data, desired_heading_data, 'r--', label='Desired Heading', linewidth=2)
        ax2.plot(time_data, actual_heading_data, 'b-', label='Actual Heading', linewidth=2)
        ax2.set_title(f"Heading Control - t={t:.1f}s")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Heading (deg)")
        ax2.legend()
        ax2.grid(True)
        
        # Set reasonable y-axis limits
        if len(actual_heading_data) > 1:
            y_min = min(min(actual_heading_data), min(desired_heading_data)) - 10
            y_max = max(max(actual_heading_data), max(desired_heading_data)) + 10
            ax2.set_ylim(y_min, y_max)
        
        plt.pause(1e-9)

def heading_controller_final_results() -> None:
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
            desired_heading_deg=5,
            desired_speed=0
        ),
        controller=HeadingAndSpeedController(
            pid_gains_heading=(3e7, 0, 5e9), # (1e8, 0, 2.2e10), # 3e5, 6e4, 0
            pid_gains_speed=(0, 0, 0),
            dt=dt,
            anti_windup=(1e2, float('inf')), # 1e2
            allocation=PowerMinimizerControlAllocation(actuators, Q=Q_P, W=W_P)
        ),
        actuators=actuators,
        length=LOA, 
        width=BEAM
    )
    
    env = NavEnv(own_ships=[ship])

    # Data storage for simulation
    time_data = []
    actual_heading_data = []
    desired_heading_data = []
    heading_error_data = []
    control_effort_data = []

    tf = 5000
    print("Running simulation...")
    
    # Run the simulation without plotting
    for t in np.linspace(0, tf, int(tf//dt)):
        env.step()
        
        # Get data
        actual_heading = ship.states.psi_deg
        desired_heading = ship._gnc._guidance.desired_heading_deg
        heading_error = desired_heading - actual_heading
        
        # Get control effort (torque moment)
        control_effort_single_actuator = ship._gnc._controller.last_commanded_force.tau_z / 33 / 2
        
        # Store data
        time_data.append(t)
        actual_heading_data.append(actual_heading)
        desired_heading_data.append(desired_heading)
        heading_error_data.append(heading_error)
        control_effort_data.append(control_effort_single_actuator)
        print(t)

    print("Simulation complete. Plotting results...")

    # Create final results plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Heading tracking
    ax1.plot(time_data, desired_heading_data, 'r--', label='Desired Heading', linewidth=2)
    ax1.plot(time_data, actual_heading_data, 'b-', label='Actual Heading', linewidth=2)
    ax1.set_title("Heading Tracking Performance")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Heading (deg)")
    ax1.legend()
    ax1.grid(True)

    # 2. Heading error
    ax2.plot(time_data, heading_error_data, 'g-', linewidth=2)
    ax2.set_title("Heading Error")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error (deg)")
    ax2.grid(True)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # 3. Final ship position and trajectory
    ax3.set_title("Final Ship Position and Trajectory")
    env.plot(lim, ax=ax3)
    ax3.set_aspect('equal')

    # 4. Control effort
    ax4.plot(time_data, control_effort_data, 'm-', linewidth=2)
    ax4.set_title("Control Effort (Single Actuator)")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Force (N)")
    ax4.grid(True)

    plt.tight_layout()

    # Print performance metrics
    final_error = abs(heading_error_data[-1])
    max_error = max(abs(e) for e in heading_error_data)
    settling_time = None
    
    # Calculate settling time (time to reach within 2% of final value)
    target_band = 2.0  # degrees
    for i, error in enumerate(heading_error_data):
        if abs(error) <= target_band:
            settling_time = time_data[i]
            break
    
    print(f"\n=== Heading Controller Performance ===")
    print(f"Final heading error: {final_error:.2f} degrees")
    print(f"Maximum error: {max_error:.2f} degrees")
    print(f"Settling time (±{target_band}°): {settling_time:.1f}s" if settling_time else f"Did not settle within ±{target_band}°")
    print(f"PID gains: Kp={1e7:.0e}, Ki={0}, Kd={1e9:.0e}")

    plt.show()

if __name__ == "__main__":
    heading_controller()
    # heading_controller_final_results()
