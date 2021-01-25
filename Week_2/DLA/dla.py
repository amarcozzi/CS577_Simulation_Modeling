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
        self.stop_flag = False

    def seed_particle(self):
        """ Seeds the world with a single particle in the center"""
        pt = (self.N // 2, self.N // 2)
        self.num_particles += 1
        self.world[pt] = self.num_particles

    def simulate(self):
        """ Performs a step in the DLA algorithm """
        # Place particles
        if self.start_radius <= self.N // 2:
            while True:

                init_position = self._find_initial_position()

                # Random walk was a success and a particle was added
                if self.random_walk(init_position):
                    return
                # Restart the random walk with a new initial point
                else:
                    pass
        else:
            self.stop_flag = True

    def random_walk(self, p):
        """ Performs a random walk for a particle randomly placed on the circumference of a circle """
        # Starting point of the particle
        x, y = p

        # Repeat the random walk until
        while True:
            r_squared = np.square(x - self.N // 2) + np.square(y - self.N // 2)
            r = 1 + np.int(np.sqrt(r_squared))

            # If the particle is already in the cluster, try again and release a new particle
            if self.world[y, x] == 1:
                print('Point already in cluster')
                return False

            # Start new walker because it's not going the right direction
            if r > self.max_radius:
                return False

            # The walk has hit the cluster, so add the particle and increment the starting radius
            neighbor_sum = self._neighbor_sum((x, y))
            if r < self.N // 2 and neighbor_sum > 0:
                self.num_particles += 1
                self.world[y, x] = 1
                if r >= self.start_radius:
                    self.start_radius += 2
                self.max_radius = self.start_radius + self.ring_size
                return True

            # Particle still hasn't hit the cluster, take a step
            else:
                random_number = np.random.randint(4)
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


dla = DLA(N=300, start_radius=2)
dla.seed_particle()

fig, ax = plt.subplots()
im = ax.imshow(dla.world, cmap='PuRd')
plt.axis('off')


def animate(i):
    global dla
    if dla.stop_flag:
        return
    dla.simulate()
    im.set_data(dla.world)
    ax.set_title(f'DLA with {dla.num_particles} Particles')


anim = FuncAnimation(fig, animate, interval=20, frames=5000, repeat=False)
plt.show()
