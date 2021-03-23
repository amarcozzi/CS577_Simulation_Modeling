import numpy as np

class RabbitModel:

    def __init__(self, V, wind_dir):
        """ Initializes the Rabbit model """

        # Wind parameters
        self.V = V
        self.wind_dir = np.radians(wind_dir)

        # Compute wind as a vector and a unit vector
        x_vel = V * np.cos(self.wind_dir)
        y_vel = V * np.sin(self.wind_dir)
        self.wind_vec = np.array([x_vel, y_vel])
        self.unit_wind_vec = self.wind_vec / np.linalg.norm(self.wind_vec)