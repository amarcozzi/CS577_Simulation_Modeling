"""
Based on the paper: A hybrid cellular automata/semi-physical model of fire growth
BY: A.L. Sullivan and I.K. Knight

https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.6604&rep=rep1&type=pdf
"""
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap
from shapely.geometry import LineString, MultiLineString
from matplotlib.collections import LineCollection


class Bubble:

    def __init__(self, pos, vertical_velocity):
        self.pos = pos
        self.vertical_velocity = vertical_velocity
        self.number_time_steps = 1

    def update_position(self, dt, v_w, theta_w):
        """
        Moves the bubble up according to its assigned vertical wind speed, and moves the bubble downwind of the wind
        vector
        """
        x_old, y_old, z_old = self.pos
        # Update new bubble z position
        z_new = z_old + self.vertical_velocity * dt

        # Get the x and y components of the wind vector
        x_vel = v_w * np.cos(theta_w)
        y_vel = v_w * np.sin(theta_w)

        # Update x and y positions of the bubble
        x_new = x_old + x_vel * dt
        y_new = y_old + y_vel * dt

        self.pos = (x_new, y_new, z_new)


class BubbleModel:
    def __init__(self, XB, YB, num_x_cells, num_y_cells, V, theta_w):
        """
        :param tuple XB: Length of the domain in the X direction
        :param tuple YB: Length of the domain in the Y direction
        :param int num_x_cells: Number of cells in the X direction
        :param int num_y_cells: Number of cells in the Y direction
        :param float V: Wind speed
        :param float theta_w: Wind direction in degrees
        """
        # Initialize the world
        # each cell exists in one of three states: unburnt-1, burning-2, burnt-3
        self.world = np.zeros([num_y_cells, num_x_cells], dtype=np.int8)

        # Create our stochastic fuel parameters
        self.fuel_prob = np.random.random([num_y_cells, num_x_cells])
        self.threshold = 1

        # Create a cartesian grid of the world.
        x0, x1 = XB
        x_space = np.linspace(x0, x1, num_x_cells)
        y0, y1 = YB
        y_space = np.linspace(y0, y1, num_y_cells)
        self.xx, self.yy = np.meshgrid(x_space, y_space)
        self.grid = np.zeros([num_y_cells, num_x_cells, 2])
        self.grid[:, :, 0] = self.xx
        self.grid[:, :, 1] = self.yy

        # Also create the grid as a series of lines
        lines = []
        for x in x_space:
            lines.append(((x, y0), (x, y1)))
        for y in y_space:
            lines.append(((x0, y), (x1, y)))
        self.grid_lines = MultiLineString(lines)

        # Plot the line grid
        """self.fig, self.ax = plt.subplots()
        lc = LineCollection(lines, color='gray', lw=1, alpha=0.5)
        self.ax.add_collection(lc)
        plt.xlim([x0 - 0.01, x1 + 0.01])
        plt.ylim([y0 - 0.01, y1 + 0.01])"""

        # Create a grid analogous to the world

        self.bubble_list = []
        self.burning_cells = []
        self.burning_cells_clock = []
        self.center_of_mass = np.zeros(2)
        self.N = 0  # number of burning cells

        # Wind parameters
        self.V = V
        self.wind_dir = np.radians(theta_w)

        # Compute wind as a vector and a unit vector
        x_vel = V * np.cos(self.wind_dir)
        y_vel = V * np.sin(self.wind_dir)
        self.wind_vec = np.array([x_vel, y_vel])
        self.unit_wind_vec = self.wind_vec / np.linalg.norm(self.wind_vec)

        # Keep track of the simulation time
        self.time = 0
        self.dt = 1  # seconds

        # Add some default fuel parameters. We can imagine these changing
        self.burnout_time = 15  # seconds

        self.project = np.array([])

        print('debug_init')

    def initialize_burn(self, rows, cols):
        """ Assigns the (row, col) pairs to state value 3 to represent ignited fuels """
        # Add the burning cells to a list
        com_x = 0
        com_y = 0
        for y, x in zip(rows, cols):
            # Set the positions in the state to burning
            self.world[y, x] = 2

            # Track the burning cells and their burnout time
            self.burning_cells.append((y, x))
            burnout_time = np.random.normal(12, 5)
            self.burning_cells_clock.append(burnout_time)

            # Compute the center of mass of the burning cells
            self._increment_com(y, x)
        self.center_of_mass[0] /= len(self.burning_cells)
        self.center_of_mass[1] /= len(self.burning_cells)

        new_bubble = Bubble((self.center_of_mass[0], self.center_of_mass[1], 0), 5)
        self.bubble_list.append(new_bubble)

    def do_one_time_step(self):
        """
        Performs one time step of the bubble model:
        1) Adds a new bubble to the abstract 3D world
        2) Updates the positions of the other 5 bubbles
        3) Computes vector for fire spread for each cell
        4) Compute maximum distance of fire spread from the magnitude of the fire vector
        5) Ignite new cells along the fire spread vector
        """
        # Add a new bubble to the world
        new_bubble = Bubble((self.center_of_mass[0], self.center_of_mass[1], 0), 5)
        self.bubble_list.append(new_bubble)
        if len(self.bubble_list) > 6:  # Cull bubbles to keep 6 bubbles
            self.bubble_list.pop(0)

        # Update positions of the other 5 bubbles (ignore last bubble since it was just added)
        for bubble in self.bubble_list[:-1]:
            bubble.update_position(self.dt, self.V, self.wind_dir)

        next_burning_cells = list()
        next_burning_cells_clock = list()
        self.center_of_mass = np.zeros(2)

        # Loop over all of the burning cells
        for cell, k in zip(self.burning_cells, range(len(self.burning_cells))):
            # Get row i and column j of the cell
            i, j = cell

            # Recall world coordinate (j, i) = cartesian coordinate (x, y)
            cell_coord = np.array([self.grid[i, j, 0], self.grid[i, j, 1]])

            # Check if the cell should be extinguished. If it is still burning, add it to the next burning cells list
            if self.time < self.burning_cells_clock[k]:
                next_burning_cells.append(cell)
                next_burning_cells_clock.append(self.burning_cells_clock[k])
                self._increment_com(i, j)
            # Extinguish the cell and move on
            else:
                self.world[i, j] = 3
                continue

            # The cell continues to burn, compute the fire spread vector
            # cell_pos =
            fire_wind_vec = np.zeros(3)
            for bubble in self.bubble_list:
                fire_wind_vec += self._compute_fire_wind(cell_coord, bubble.pos)
            fire_wind_vec = fire_wind_vec

            # Sum the fire wind vector with the wind vector to get the resulting fire spread vector
            fire_spread_vec = np.array([fire_wind_vec[0] + self.wind_vec[0], fire_wind_vec[1] + self.wind_vec[1]])

            # Determine which cells lie along the fire spread vector
            cells_on_fire_spread_vec = self._find_cells_on_vector_with_line(cell_coord, fire_spread_vec)

            # Stochastically determine if cells along the fire spread vector ignite
            new_fire_cells, new_fire_clocks = self._ignite_new_cells(cells_on_fire_spread_vec)

            # Add the newly burning cells to our list for the next time step, update their burn clock
            next_burning_cells.extend(new_fire_cells)
            next_burning_cells_clock.extend(new_fire_clocks)

        # Compute the center of mass for the current time step
        self.center_of_mass[0] /= len(next_burning_cells)
        self.center_of_mass[1] /= len(next_burning_cells)

        # Update the burning cells and increment the time step
        self.burning_cells = next_burning_cells
        self.burning_cells_clock = next_burning_cells_clock
        self.time += self.dt
        print(self.center_of_mass)

    def _ignite_new_cells(self, cells):
        """ Ignites cells along the fire spread vector with some probability """
        successfully_ignited = []
        clocks = []
        for cell in cells:
            i, j = cell
            if self.fuel_prob[i, j] <= self.threshold and self.world[i, j] == 0:
                successfully_ignited.append(cell)
                clocks.append(self.burnout_time + self.time)
                self._increment_com(i, j)
                self.world[i, j] = 2
        return successfully_ignited, clocks

    def _find_cells_on_vector_with_line(self, start_pt, vector):
        """ Given a vector in coordinates (x, y), this function returns all of the cells (in world coordinates)
            that lie along the vector, starting from the start_position.
        """
        # Solve for the case when one of the vector components is zero (breaks the line distance function)
        if vector[0] == 0:
            vector[0] += 1e-10
        if vector[1] == 0:
            vector[1] += 1e-10

        end_pt = np.array([start_pt[0] + vector[0], start_pt[1] + vector[1]])
        line = LineString((start_pt, end_pt))

        cells_on_line = []
        try:
            for k, segment in enumerate(line.difference(self.grid_lines)):
                x, y = segment.xy
                i, j = self._cartesian_to_mesh(x[1], y[1])
                cells_on_line.append((i, j))
                # plt.plot(x, y)
                # plt.text(np.mean(x), np.mean(y), str(k))

            # plt.scatter(start_pt[0], start_pt[1], color='red')
            # plt.text(start_pt[0], start_pt[0], f'({start_pt[0]:.1f}, {start_pt[1]:.1f})')
            # plt.show()
        except TypeError:
            i, j = self._cartesian_to_mesh(end_pt[0], end_pt[1])
            cells_on_line.append((i, j))

        return cells_on_line

    @staticmethod
    def _compute_fire_wind(cell_loc, bubble_loc):
        """
        This function computes the fire wind vector from a given cell to a given bubble. According to the paper, the
        magnitude of the vector is based on the inverse square of the distance from the cell to the bubble.
        """
        vec = np.array([bubble_loc[0] - cell_loc[0], bubble_loc[1] - cell_loc[1], bubble_loc[2]])
        r = np.power(np.linalg.norm(vec), 2)
        alpha_abs = np.divide(1, r, out=np.zeros_like(vec), where=r != 0)
        alpha = np.power(np.power(alpha_abs, 2), 1/2)
        vec_cell_to_bubble = alpha * vec

        return vec_cell_to_bubble

    def _cartesian_to_mesh(self, x, y):
        """ Converts cartesian points x and y to row i and column j coordinates of the mesh """
        i = (np.abs(self.grid[:, 0, 1] - y)).argmin()
        j = (np.abs(self.grid[0, :, 0] - x)).argmin()

        return i, j

    def _increment_com(self, i, j):
        self.center_of_mass[0] += self.grid[i, j, 0]
        self.center_of_mass[1] += self.grid[i, j, 1]


# Initialize a bubble model
X_length = 80
Y_length = 40
model = BubbleModel(XB=(-X_length // 2, X_length // 2), YB=(-Y_length // 2, Y_length // 2), num_x_cells=X_length,
                    num_y_cells=Y_length, V=0.5, theta_w=0)

# Start the fire
ignition_line_rows = [i for i in range(Y_length // 4, Y_length - Y_length // 4)]
ignition_line_cols = [0 for j in range(X_length // 4, X_length - X_length // 4)]
model.initialize_burn(ignition_line_rows, ignition_line_cols)

# Plot the fire effects
# Create custom colormaps for burning and for vegetation
color_dict = {1: "white",
              2: "red",
              3: "dimgrey"}

# We create a colormar from our list of colors
cm = ListedColormap([color_dict[x] for x in color_dict.keys()])

# Let's also define the description of each category
labels = np.array(["No Fuel", "Available Fuel", "Burning", "Consumed"])

# Prepare bins for the normalizer
norm_bins = np.sort([*color_dict.keys()]) + 0.5
norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)

# Make normalizer and formatter
fig, ax = plt.subplots()
norm = mpl.colors.BoundaryNorm(norm_bins, len(labels), clip=True)
fmt = mpl.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
fire_state = plt.imshow(model.world, cmap=cm, norm=norm, interpolation='nearest',
                        extent=[-X_length // 2, X_length // 2, -Y_length // 2,
                                Y_length // 2])
diff = norm_bins[1:] - norm_bins[:-1]
tickz = norm_bins[:-1] + diff / 2
cb = fig.colorbar(fire_state, format=fmt, ticks=tickz)

# Plot the center of mass
plt.scatter(model.center_of_mass[0], model.center_of_mass[1], c='g')

print('starting model')
for count in range(100):
    model.do_one_time_step()
    fire_state.set_data(model.world)
    plt.scatter(model.center_of_mass[0], model.center_of_mass[1], c='g')
