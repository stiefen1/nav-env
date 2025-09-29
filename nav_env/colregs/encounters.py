from enum import Enum
from typing import Tuple, Literal
import numpy as np
from nav_env.utils.math_functions import wrap_angle_to_pmpi_degrees
import os

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


class Encounter(Enum):
    INVALID = 0
    HEAD_ON = 1
    STARBOARD = 2
    PORT = 3
    OVERTAKING = 4

class Recommendation(Enum):
    INVALID = 0
    TURN_RIGHT = 1
    DO_NOTHING = 2
    TURN_LEFT = 3 # in case of good seamanship when crossing a TSS for instance

CROSSING_ANGLE_DEG = 22.5 # Angle w.r.t horizontal that separate starboard, port from overtaking (degrees)

def get_encounter(
        pose_os:Tuple, 
        heading_os_deg:float,
        pose_ts:Tuple,
        head_on_lim_deg:float=10,
        input_convention:Literal['nav-env', 'blue-boat']='nav-env'
        ) -> Encounter:
    """
    Returns encounter based on OS and TS pose. 
    
    """

    if input_convention == 'nav-env':
        # nav-env convention:
        # pose_os = (east, north)
        # heading is zero when towards north and counter-clockwise positive
        heading_os_deg = -heading_os_deg
    elif input_convention == 'blue-boat':
        pose_os = (pose_os[1], pose_os[0]) # north-east becomes east-north
        pose_ts = (pose_ts[1], pose_ts[0])

    heading_os_rad = np.pi*wrap_angle_to_pmpi_degrees(heading_os_deg)/180
    pose_os = np.array(pose_os)
    pose_ts = np.array(pose_ts)
    pose_rel = pose_ts - pose_os 
    
    encounter_angle_raw_rad = np.sign(pose_rel[0]) * np.arccos(pose_rel[1] / np.linalg.norm(pose_rel)) - heading_os_rad
    encounter_angle_raw_deg = 180 * encounter_angle_raw_rad / np.pi
    encounter_angle_deg = wrap_angle_to_pmpi_degrees(encounter_angle_raw_deg) # [0, 1] @ pose_rel / (norm([0, 1]) * norm(pose_rel)) = cos(a)

    if -head_on_lim_deg <= encounter_angle_deg <= head_on_lim_deg:
        return Encounter.HEAD_ON
    elif -90-CROSSING_ANGLE_DEG <= encounter_angle_deg <= -head_on_lim_deg:
        return Encounter.PORT
    elif head_on_lim_deg <= encounter_angle_deg <= 90+CROSSING_ANGLE_DEG:
        return Encounter.STARBOARD
    elif (-180 <= encounter_angle_deg <= -90-CROSSING_ANGLE_DEG) or (90+CROSSING_ANGLE_DEG <= encounter_angle_deg <= 180):
        return Encounter.OVERTAKING
    else:
        print(f"Invalid value for encounter with relative angle {encounter_angle_deg:.2f} deg")
        return Encounter.INVALID
    
def get_recommendation_from_encounters(ts_wrt_os:Encounter, os_wrt_ts:Encounter, good_seamanship:bool=False, ts_in_TSS:bool=False, os_in_TSS:bool=False) -> Recommendation:
    assert ts_wrt_os != Encounter.INVALID, f"Encounter ts_wrt_os is invalid"
    assert os_wrt_ts != Encounter.INVALID, f"Encounter os_wrt_ts is invalid"
    
    if ts_wrt_os == Encounter.OVERTAKING or os_wrt_ts == Encounter.OVERTAKING:
        return Recommendation.DO_NOTHING
    
    if good_seamanship and ts_in_TSS and (not os_in_TSS):
        match ts_wrt_os:
            case Encounter.HEAD_ON:
                    if os_wrt_ts == Encounter.STARBOARD:
                        return Recommendation.TURN_LEFT
                    else:
                        return Recommendation.TURN_RIGHT
            case Encounter.STARBOARD:
                if os_wrt_ts == Encounter.STARBOARD:
                    return Recommendation.TURN_LEFT
                else:
                    return Recommendation.TURN_RIGHT
            case Encounter.PORT:
                if os_wrt_ts == Encounter.PORT:
                    return Recommendation.TURN_RIGHT
                else:
                    return Recommendation.TURN_LEFT
            case _:
                print(f"Invalid Encounter value {ts_wrt_os}")
                return Recommendation.INVALID
    else:
        match ts_wrt_os:
            case Encounter.HEAD_ON:
                    return Recommendation.TURN_RIGHT
            case Encounter.STARBOARD:
                if os_wrt_ts == Encounter.STARBOARD:
                    return Recommendation.DO_NOTHING
                else:
                    return Recommendation.TURN_RIGHT
            case Encounter.PORT:
                if os_wrt_ts == Encounter.PORT:
                    return Recommendation.DO_NOTHING
                else:
                    return Recommendation.TURN_RIGHT
            case _:
                print(f"Invalid Encounter value {ts_wrt_os}")
                return Recommendation.INVALID
        


# def test_encounter() -> None:
#     from nav_env.ships.ship import Ship, States3
#     from nav_env.ships.moving_ship import MovingShip
#     import matplotlib.pyplot as plt

#     os = Ship(states=States3(5, 3, psi_deg=70), length=10, width=3)
#     ts = MovingShip(states=States3(40, 30, psi_deg=120), length=5, width=2)
#     print(
#         get_encounter(pose_os=os.states.xy, heading_os_deg=os.states.psi_deg, pose_ts=ts.states.xy, input_convention='nav-env'),
#         get_encounter(pose_os=ts.states.xy, heading_os_deg=ts.states.psi_deg, pose_ts=os.states.xy, input_convention='nav-env')
#         )

#     ax = os.plot()
#     ts.plot(ax=ax)
#     ax.set_aspect('equal')
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from nav_env.ships.ship import Ship, States3
from nav_env.ships.moving_ship import MovingShip
from nav_env.colregs.encounters import get_encounter, Encounter

class InteractiveEncounter:
    def __init__(self):
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.3, right=0.7)
        
        # Initial parameters
        self.os_x = 10
        self.os_y = 10
        self.os_heading = 45
        self.ts_x = 30
        self.ts_y = 20
        self.ts_heading = 225
        
        # Create ships
        self.os = Ship(states=States3(self.os_x, self.os_y, psi_deg=self.os_heading), length=15, width=4)
        self.ts = MovingShip(states=States3(self.ts_x, self.ts_y, psi_deg=self.ts_heading), length=10, width=3)
        
        # Create sliders
        self.create_sliders()
        
        # Initial plot
        self.update_plot()
        
    def create_sliders(self):
        """Create all the slider widgets."""
        # Slider positions
        slider_height = 0.03
        slider_spacing = 0.04
        slider_left = 0.15
        slider_width = 0.5
        
        # OS sliders
        ax_os_x = plt.axes([slider_left, 0.25, slider_width, slider_height])
        ax_os_y = plt.axes([slider_left, 0.25 - slider_spacing, slider_width, slider_height])
        ax_os_heading = plt.axes([slider_left, 0.25 - 2*slider_spacing, slider_width, slider_height])
        
        # TS sliders
        ax_ts_x = plt.axes([slider_left, 0.25 - 4*slider_spacing, slider_width, slider_height])
        ax_ts_y = plt.axes([slider_left, 0.25 - 5*slider_spacing, slider_width, slider_height])
        ax_ts_heading = plt.axes([slider_left, 0.25 - 6*slider_spacing, slider_width, slider_height])
        
        # Create sliders
        self.slider_os_x = Slider(ax_os_x, 'OS X', 0, 50, valinit=self.os_x)
        self.slider_os_y = Slider(ax_os_y, 'OS Y', 0, 40, valinit=self.os_y)
        self.slider_os_heading = Slider(ax_os_heading, 'OS Heading', 0, 360, valinit=self.os_heading)
        
        self.slider_ts_x = Slider(ax_ts_x, 'TS X', 0, 50, valinit=self.ts_x)
        self.slider_ts_y = Slider(ax_ts_y, 'TS Y', 0, 40, valinit=self.ts_y)
        self.slider_ts_heading = Slider(ax_ts_heading, 'TS Heading', 0, 360, valinit=self.ts_heading)
        
        # Connect sliders to update function
        self.slider_os_x.on_changed(self.update_parameters)
        self.slider_os_y.on_changed(self.update_parameters)
        self.slider_os_heading.on_changed(self.update_parameters)
        self.slider_ts_x.on_changed(self.update_parameters)
        self.slider_ts_y.on_changed(self.update_parameters)
        self.slider_ts_heading.on_changed(self.update_parameters)
    
    def update_parameters(self, val):
        """Update parameters from sliders and refresh plot."""
        self.os_x = self.slider_os_x.val
        self.os_y = self.slider_os_y.val
        self.os_heading = self.slider_os_heading.val
        self.ts_x = self.slider_ts_x.val
        self.ts_y = self.slider_ts_y.val
        self.ts_heading = self.slider_ts_heading.val
        
        # Update ship states
        self.os = Ship(states=States3(self.os_x, self.os_y, psi_deg=self.os_heading), length=15, width=4)
        self.ts = MovingShip(states=States3(self.ts_x, self.ts_y, psi_deg=self.ts_heading), length=10, width=3)
        
        self.update_plot()
    
    def update_plot(self):
        """Update the plot with current ship positions and encounter analysis."""
        self.ax.clear()
        
        # Plot ships
        self.os.plot(ax=self.ax, c='blue', alpha=0.8)
        self.ts.plot(ax=self.ax, c='red', alpha=0.8)
        
        # Add ship labels
        self.ax.text(self.os_x + 2, self.os_y + 2, 'OS (Own Ship)', fontsize=12, c='blue', fontweight='bold')
        self.ax.text(self.ts_x + 2, self.ts_y + 2, 'TS (Target Ship)', fontsize=12, c='red', fontweight='bold')
        
        # Draw line between ships
        self.ax.plot([self.os_x, self.ts_x], [self.os_y, self.ts_y], 'k--', alpha=0.5, linewidth=1)
        
        # Calculate encounters
        encounter_os_vs_ts = get_encounter(
            pose_os=self.os.states.xy, 
            heading_os_deg=self.os.states.psi_deg, 
            pose_ts=self.ts.states.xy,
            input_convention='nav-env'
        )
        
        encounter_ts_vs_os = get_encounter(
            pose_os=self.ts.states.xy, 
            heading_os_deg=self.ts.states.psi_deg, 
            pose_ts=self.os.states.xy,
            input_convention='nav-env'
        )

        # Recommendation system
        clear_terminal()
        print("COLREGs Recommendation for OS: ", get_recommendation_from_encounters(encounter_os_vs_ts, encounter_ts_vs_os))
        print("COLREGs Recommendation for TS: ", get_recommendation_from_encounters(encounter_ts_vs_os, encounter_os_vs_ts))


        # Calculate distance and bearing
        dx = self.ts_x - self.os_x
        dy = self.ts_y - self.os_y
        distance = np.sqrt(dx**2 + dy**2)
        bearing = np.degrees(np.arctan2(dx, dy))
        
        # Add encounter information as text
        info_text = f"Distance: {distance:.1f} units\n"
        info_text += f"Bearing (OS→TS): {bearing:.1f}°\n\n"
        info_text += f"OS vs TS: {encounter_os_vs_ts.name}\n"
        info_text += f"TS vs OS: {encounter_ts_vs_os.name}\n\n"
        info_text += f"OS Heading: {self.os_heading:.1f}°\n"
        info_text += f"TS Heading: {self.ts_heading:.1f}°"
        
        # Color code the encounter types
        encounter_colors = {
            Encounter.HEAD_ON: 'orange',
            Encounter.STARBOARD: 'green',
            Encounter.PORT: 'purple',
            Encounter.OVERTAKING: 'brown',
            Encounter.INVALID: 'gray'
        }
        
        # Add text box with encounter information
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Add colored indicators for encounter types
        os_color = encounter_colors.get(encounter_os_vs_ts, 'black')
        ts_color = encounter_colors.get(encounter_ts_vs_os, 'black')
        
        self.ax.plot(self.os_x, self.os_y, 'o', c=os_color, markersize=15, alpha=0.7)
        self.ax.plot(self.ts_x, self.ts_y, 'o', c=ts_color, markersize=15, alpha=0.7)
        
        # Set axis properties
        self.ax.set_xlim(-5, 55)
        self.ax.set_ylim(-5, 45)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('East (m)')
        self.ax.set_ylabel('North (m)')
        self.ax.set_title('Interactive COLREGS Encounter Analysis')
        
        # Add legend for encounter types
        legend_elements = []
        for encounter, color in encounter_colors.items():
            if encounter != Encounter.INVALID:
                legend_elements.append(plt.Line2D([0], [0], marker='o', c='w', 
                                                markerfacecolor=color, markersize=10, 
                                                label=encounter.name))
        
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        # Refresh the plot
        self.fig.canvas.draw()
    
    def show(self):
        """Display the interactive figure."""
        plt.show()

def create_interactive_encounter():
    """Create and show the interactive encounter analysis."""
    interactive = InteractiveEncounter()
    
    # Add instructions
    plt.figtext(0.75, 0.8, "Instructions:", fontsize=12, fontweight='bold')
    instructions = """
• Use sliders to change ship positions and headings
• Colored circles around ships indicate encounter type
• OS = Own Ship (blue), TS = Target Ship (red)
• Encounter types are calculated for both perspectives

Encounter Types:
• HEAD_ON: Ships approaching head-on
• STARBOARD: TS on OS's starboard side
• PORT: TS on OS's port side  
• OVERTAKING: One ship overtaking another
    """
    plt.figtext(0.75, 0.2, instructions, fontsize=10, verticalalignment='top')
    
    interactive.show()

if __name__ == "__main__":
    create_interactive_encounter()
