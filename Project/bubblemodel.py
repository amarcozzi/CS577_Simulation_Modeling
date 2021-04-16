"""
Based on the paper: A hybrid cellular automata/semi-physical model of fire growth
BY: A.L. Sullivan and I.K. Knight

https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.6604&rep=rep1&type=pdf
"""
import matplotlib.pyplot as plt
import numpy as np
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
    def __init__(self, XB, YB, I, J, V, theta_w):
        """
        :param int X: Length of the domain in the X direction
        :param int Y: Length of the domain in the Y direction
        :param int I: Number of cells in the X direction
        :param int J: Number of cells in the Y direction
        :param float V: Wind speed
        :param float theta_w: Wind direction in degrees
        """
        # Initialize the world
        # each cell exists in one of three states: unburnt-1, burning-2, burnt-3
        self.world = np.zeros([J, I], dtype=np.int8)

        # Create a cartesian grid of the world.
        x0, x1 = XB
        x_space = np.linspace(x0, x1, I)
        y0, y1 = YB
        y_space = np.linspace(y0, y1, J)
        xx, yy = np.meshgrid(x_space, y_space)
        self.grid = np.zeros([J, I, 2])
        self.grid[:, :, 0] = xx
        self.grid[:, :, 1] = yy

        # Also create the grid as a series of lines
        lines = []
        for x in x_space:
            lines.append(((x, y0), (x, y1)))
        for y in y_space:
            lines.append(((x0, y), (x1, y)))
        self.grid_lines = MultiLineString(lines)

        # Plot the line grid
        self.fig, self.ax = plt.subplots()
        lc = LineCollection(lines, color='gray', lw=1, alpha=0.5)
        self.ax.add_collection(lc)
        plt.xlim([x0-0.01, x1+0.01])
        plt.ylim([y0-0.01, y1+0.01])


        self.bubble_list = list()
        self.burning_cells = list()
        self.burning_cells_clock = list()
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

    def ignite_cells(self, rows, cols):
        """ Assigns the (row, col) pairs to state value 3 to represent ignited fuels """
        # Set the positions in the state to burning
        self.world[rows, cols] = 2

        # Add the burning cells to a list
        com_x = 0
        com_y = 0
        for y in rows:
            for x in cols:
                # Track the burning cells and their burnout time
                self.burning_cells.append((x, y))
                burnout_time = np.random.normal(12, 5)
                self.burning_cells_clock.append(burnout_time)
                self.N += 1

                # Compute the center of mass of the burning cells
                com_x += x
                com_y += y
        self.center_of_mass[0] = com_x / self.N
        self.center_of_mass[1] = com_y / self.N

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
        if len(self.bubble_list) > 6:   # Cull bubbles to keep 6 bubbles
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
            j, i = cell

            # Recall world coordinate (j, i) = cartesian coordinate (x, y)
            cell_coord = np.array([self.grid[i, j, 0], self.grid[i, j, 1]])

            # Check if the cell should be extinguished. If it is still burning, add it to the next burning cells list
            if self.time < self.burning_cells_clock[k]:
                next_burning_cells.append(cell)
                next_burning_cells_clock.append(self.burning_cells_clock[k])
            # Extinguish the cell and move on
            else:
                self.world[j, i] = 3
                break

            # The cell continues to burn, compute the fire spread vector
            # cell_pos =
            fire_wind_vec = np.zeros(3)
            for bubble in self.bubble_list:
                fire_wind_vec += self._compute_fire_wind(cell, bubble.pos)
            fire_wind_vec = fire_wind_vec/len(self.bubble_list)

            # Sum the fire wind vector with the wind vector to get the resulting fire spread vector
            fire_spread_vec = np.array([fire_wind_vec[0] + self.wind_vec[0], fire_wind_vec[1] + self.wind_vec[1]])

            # Determine which cells lie along the fire spread vector
            cells_on_fire_spread_vec = self._find_cells_on_vector(cell_coord, fire_spread_vec)
            print(cell, fire_spread_vec)

            # Add the burning cells location to the center of mass
            # self.center_of_mass += []

        # Compute the center of mass for the current time step
        self.center_of_mass[0] /= len(self.burning_cells)
        self.center_of_mass[1] /= len(self.burning_cells)

        # Update the burning cells and increment the time step
        self.burning_cells = next_burning_cells
        self.burning_cells_clock = next_burning_cells_clock
        self.time += self.dt

    def _find_cells_on_vector(self, start_pt, vector):
        """ Given a vector in coordinates (x, y), this function returns all of the cells (in world coordinates)
            that lie along the vector, starting from the start_position.
        """
        vector = [2, 1e-10]
        end_pt = np.array([start_pt[0] + vector[0], start_pt[1] + vector[1]])
        line = LineString((start_pt, end_pt))
        # lc = LineCollection([line], color='red')
        # self.ax.add_collection(lc)
        # plt.show()

        for k, segment in enumerate(line.difference(self.grid_lines)):
            x, y = segment.xy
            i, j = self._cartesian_to_mesh(x[1], y[1])
            plt.plot(x, y)
            plt.text(np.mean(x), np.mean(y), str(k))
        plt.scatter(start_pt[0], start_pt[1], color='red')
        plt.text(start_pt[0], start_pt[0], f'({start_pt[0]:.1f}, {start_pt[1]:.1f})')
        plt.show()

        return True

    def _compute_fire_wind(self, cell_loc, bubble_loc):
        """
        This function computes the fire wind vector from a given cell to a given bubble. According to the paper, the
        magnitude of the vector is based on the inverse square of the distance from the cell to the bubble.
        """
        vec = np.array([bubble_loc[0] - cell_loc[0], bubble_loc[1] - cell_loc[1], bubble_loc[2]])
        comp = np.power(vec, 3)
        alpha = np.divide(1, comp, out=np.zeros_like(vec), where=comp != 0)
        vec_cell_to_bubble = alpha * vec

        return vec_cell_to_bubble

    def _cartesian_to_mesh(self, x, y):
        """ Converts cartesian points x and y to row i and column j coordinates of the mesh """
        i = (np.abs(self.grid[:, 0, 1] - y)).argmin()
        j = (np.abs(self.grid[0,:, 0] - x)).argmin()

        return i, j

# Initialize a bubble model
lattice_size = 10
model = BubbleModel(XB=(-5, 5), YB=(-5, 5), I=10, J=10, V=2, theta_w=0)
model.ignite_cells([5], [1])
for count in range(10):
    model.do_one_time_step()
print('debug_main')