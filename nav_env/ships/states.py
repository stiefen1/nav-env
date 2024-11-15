from dataclasses import dataclass

@dataclass
class ShipStates:
    """Base class for ship states"""
    position: tuple[float, float]
    velocity: tuple[float, float]
    acceleration: tuple[float, float]
    heading: float
    yaw_rate: float
    yaw_acceleration: float
    
    def __add__(self, other:"ShipStates"):
        # print(other.position[0])
        return ShipStates(
            (self.position[0] + other.position[0], self.position[1] + other.position[1]),
            (self.velocity[0] + other.velocity[0], self.velocity[1] + other.velocity[1]),
            (self.acceleration[0] + other.acceleration[0], self.acceleration[1] + other.acceleration[1]),
            self.heading + other.heading,
            self.yaw_rate + other.yaw_rate,
            self.yaw_acceleration + other.yaw_acceleration
            )
    
    def __mul__(self, dt:float):
        return ShipStates(
            (self.velocity[0] * dt, self.velocity[1] * dt),
            (self.acceleration[0] * dt, self.acceleration[1] * dt),
            (0, 0),
            self.yaw_rate * dt,
            self.yaw_acceleration * dt,
            0)
    
    def __rmul__(self, dt:float):
        return self.__mul__(dt)

def test():
    ship_states = ShipStates((0., 0.), (1., 0.), (0., 0.), 0., 0., 0.)
    print(f"Before integration: {ship_states}")
    ds = ship_states * 0.1
    # print((ship_states * 0.1).position)
    print(f"Integration step: {ds}")
    ship_states += ship_states * 0.1
    print(f"After integration: {ship_states}")

if __name__ == "__main__":
    test()  