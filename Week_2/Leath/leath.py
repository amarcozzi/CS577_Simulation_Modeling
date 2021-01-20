import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Leath:
    def __init__(self, N=80, p=.6):
        self.N = N
        self.p = p
        self.perimeter = {}
        self.cluster = {}
        self.world = np.zeros((N, N), dtype=np.int8)
        self.stop_flag = False

    def seed_cluster(self, loc=None):
        if loc:
            pt = loc
        else:
            pt = (self.N // 2, self.N // 2)

        self.cluster[pt] = True
        self.add_perimeter(pt)
        self.world[pt] = 1

    def add_perimeter(self, pt):
        """
        Given a point pt (tuple), add to a perimeter all 4 nearest neighbors that are not already in the cluster.
        """
        # Create list of four nearest neighbors
        nn = [(pt[0] + 1, pt[1]), (pt[0] - 1, pt[1]), (pt[0], pt[1] + 1), (pt[0], pt[1] - 1)]

        # Iterate through the points of the nearest neighbors and add them to the perimeter
        for p in nn:
            if p not in self.cluster and p not in self.perimeter:
                self.perimeter[p] = True

    def grow_cluster(self):
        """
        Iterate through each point in perimeter if uniform random [0,1] is less than p add perimeter point to cluster.
        Else mark point as inaccessible.
        Do something to keep cluster from leaving domain.
        """
        # Need a new list to store cluster points
        new_cluster_pts = []

        # Loop through the active points in the perimeter, add the point the cluster with probability p
        active_perimeter = [k for k, v in self.perimeter.items() if bool(v)]
        for pt in active_perimeter:
            if self.p >= np.random.rand():
                self.cluster[pt] = True
                self.perimeter[pt] = False
                new_cluster_pts.append(pt)
            else:
                self.cluster[pt] = False
                self.perimeter[pt] = False

        # Loop through the new cluster points and add their nearest neighbors to the perimeter
        for pt in new_cluster_pts:
            if pt[0] in range(1, self.N - 1) and pt[1] in range(1, self.N - 1):
                self.add_perimeter(pt)

        # Update our world.
        if new_cluster_pts:
            new_cluster_pts_x, new_cluster_pts_y = list(zip(*new_cluster_pts))[0], list(zip(*new_cluster_pts))[1]
            self.world[new_cluster_pts_y, new_cluster_pts_x] = 1
        else:
            self.stop_flag = True


# Initialize a leath cluster
leath = Leath(1000)
leath.seed_cluster()

# matplotlib initialization
fig, ax = plt.subplots()
im = ax.imshow(leath.world, cmap='GnBu')
plt.axis('off')

def animate(i):
    if i == 0:
        leath.seed_cluster()
    elif leath.stop_flag:
        return
    else:
        leath.grow_cluster()
    im.set_data(leath.world)
    ax.set_title(f'Leath Cluster at {i} Iterations')

anim = FuncAnimation(fig, animate, interval=100, frames=1000, repeat=False)

# for i in range(0, 10):
#     leath.grow_cluster()
#     plt.imshow(leath.world, cmap='terrain', vmin=0)
#     plt.show()
#     pass

plt.show()
