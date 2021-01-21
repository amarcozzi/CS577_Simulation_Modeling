import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class DLA:
    def __init__(self, N=80, start_radius=3):
        self.N = N
        self.start_radius = start_radius
        self.ring_size = N // 10
        self.max_radius = self.start_radius + self.ring_size

        self.world = np.zeros((N, N), dtype=np.int8)
        self.num_particles = 0
        self.starting_points = np.zeros((N, N), dtype=np.int8)

    def seed_particle(self, pt_loc=None):
        """ Seeds the world with a single particle in the center"""
        if pt_loc:
            pt = pt_loc
        else:
            pt = (self.N // 2, self.N // 2)
        self.num_particles += 1
        self.world[pt] = self.num_particles

    def simulate(self):
        """ Performs a step in the DLA algorithm """
        while self.start_radius < self.N // 2:
            init_position = self._find_initial_position()
            if not self.random_walk(init_position):
                break

    def random_walk(self, p):
        """ Performs a random walk for a particle randomly placed on the circumference of a circle """
        x, y = p

        while True:
            r_squared = np.square(x - self.N // 2) + np.square(y - self.N // 2)
            r = 1 + np.int(np.sqrt(r_squared))

            # Start new walker
            if r > self.max_radius:
                return True

            # Walk is finished
            neighbor_sum = self._neighbor_sum((x, y))
            if r < self.N // 2 and neighbor_sum > 0:
                self.num_particles += 1
                self.world[y, x] = 1
                if r >= self.start_radius:
                    self.start_radius += 2
                self.max_radius = self.start_radius + self.ring_size
                return False

            # Take a step
            else:
                random_number = np.random.randint(3)
                if random_number == 0 and x in range(2, self.N - 2):
                    x += 1
                elif random_number == 1 and x in range(2, self.N - 2):
                    x -= 1
                elif random_number == 2 and y in range(2, self.N - 2):
                    y += 1
                elif random_number == 3 and y in range(2, self.N - 2):
                    y -= 1

    def _find_initial_position(self):
        """ Finds the random initial position of the new random walker """
        theta = 2 * np.pi * np.random.rand()
        x = self.N // 2 + np.int(self.start_radius * np.cos(theta))
        y = self.N // 2 + np.int(self.start_radius * np.sin(theta))
        self.starting_points[y, x] = 1
        return x, y

    def _neighbor_sum(self, pt):
        """ Sums the 4 nearest neighbors around a point """
        x, y = pt
        right = self.world[y, x + 1]
        left = self.world[y, x - 1]
        up = self.world[y - 1, x]
        down = self.world[y + 1, x]
        s = right + left + up + down
        return s


dla = DLA(N=200, start_radius=5)
dla.seed_particle()

fig, ax = plt.subplots()
im = ax.imshow(dla.world, cmap='gray')
plt.axis('off')

fig2, ax2 = plt.subplots()
im2 = ax2.imshow(dla.starting_points)
plt.axis('off')


def animate(i):
    global dla
    print(dla.start_radius)
    dla.simulate()
    im.set_data(dla.world)
    im2.set_data(dla.starting_points)
    ax.set_title(f'DLA at {i} Iterations')


anim = FuncAnimation(fig, animate, interval=100, frames=1000, repeat=False)
plt.show()
