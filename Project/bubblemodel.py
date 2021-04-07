"""
Based on the paper: A hybrid cellular automata/semi-physical model of fire growth
BY: A.L. Sullivan and I.K. Knight

https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.6604&rep=rep1&type=pdf
"""

import numpy as np

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
    def __init__(self, L, V, theta_w):
        self.L = L  # The lattice is size LxL
        # each cell exists in one of three states: unburnt-1, burning-2, burnt-3
        self.world = np.zeros([L, L], dtype=np.int8)
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
        self.elapsed_time = 0
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

        # Loop over all of the burning cells
        for cell, i in zip(self.burning_cells, range(len(self.burning_cells))):
            # Check if the cell should be extinguished. If it is still burning, add it to the next burning cells list
            if self.elapsed_time < self.burning_cells_clock[i]:
                next_burning_cells.append(cell)                                 # Cell burns in next time step
                next_burning_cells_clock.append(self.burning_cells_clock[i])    # how long it takes cell to burn out
            # Extinguish the cell and move on
            else:
                self.world[cell[1], cell[0]] = 3    # Cell is extinguished, and not added to next step
                break

            # The cell continues to burn, compute the fire spread vector
            fire_wind_vec = np.zeros(3)
            for bubble in self.bubble_list:
                vec = np.array([bubble.pos[0] - cell[0], bubble.pos[1] - cell[1], bubble.pos[2]])
                comp = np.power(vec, 3)
                alpha = np.divide(1, comp, out=np.zeros_like(vec), where=comp!=0)
                cell_to_bubble_vec = alpha * vec
                fire_wind_vec += cell_to_bubble_vec
            fire_wind_vec = fire_wind_vec/len(self.bubble_list)
            # Project the 3D vector sum from the cell to the bubbles to 2D
            print(fire_wind_vec)


        self.elapsed_time += self.dt
        print('time-step debug')



# Initialize a bubble model
lattice_size = 10
model = BubbleModel(lattice_size, 0.5, 0)
model.ignite_cells([5], [1])
for i in range(10):
    model.do_one_time_step()
print('debug_main')