import os, matplotlib.pyplot as plt
from nav_env.map.map import Map

# Load map
config_path = os.path.join('examples', 'config', 'config_seacharts.yaml')
shore = Map(config_path, depth=0)

# Create figures & ax
fig = plt.figure()
ax = fig.add_subplot()

# Plot complete map
shore.plot(ax=ax)
ax.set_title("Complete Map + Region of Interest")

# Plot region of interest (rectangle)
center = (43150, 6958000)
size = (1500, 850)
region_of_interest = shore.get_obstacle_collection_in_window_from_enc(center, size)
region_of_interest.plot(c='r', ax=ax)

# Tadaaaa
plt.show()
plt.close()

