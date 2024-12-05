"""
Ideal usage:

d1 = Disturbance(WindVector(10, 45))
d2 = Disturbance(WaterVector(1, 90))

coll = DisturbanceCollection([d1, d2])

# In env.step()
for ship in ships:
    ship.step(coll, external_forces=GeneralizedForces()) # Idea is that we can pass both wind / water perturbations and external forces
"""

# from nav_env.wind.wind_vector import WindVector
# from nav_env.water.water_vector import WaterVector

# class DisturbanceCollection:
#     def __init__(self, disturbances: list[WindVector | WaterVector] = None) -> None:
#         self._disturbances = disturbances or []

#     def __iter__(self):
#         for disturbance in self._disturbances:
#             yield disturbance

#     def __str__(self) -> str:
#         return f"{type(self).__name__} Object with {len(self._disturbances)} Disturbance objects"