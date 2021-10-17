from prm import DistancePRM
import gym
import d4rl # maybe supress import warnings
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection



class AntPRM:
    def __init__(self, env, connect_distance):
        # Record information from environment
        self.env = env
        self.maze_env = env.env.wrapped_env

        self.map = self.maze_env._maze_map
        self.n_rows = len(self.map)
        self.n_cols = len(self.map[0])
        self.max_x, self.max_y = self.maze_env._rowcol_to_xy((self.n_rows - 1,self.n_cols - 1))
        self.min_x, self.min_y = self.maze_env._rowcol_to_xy((0,0))

        # Create PRM Class
        self.connect_distance = connect_distance
        self.prm = DistancePRM(distance=self.distance, collision=self.collision, extend=self.extend, connect_distance=connect_distance) 
    
    def distance(self, q1, q2):
        """
            q1,q2 = (x1, y1), (x2, y2)
            return ||q1-q2||
        """
        return math.sqrt( (q1[0] - q2[0])**2 + (q1[1] - q2[1])**2)

    def collision(self, q):
        """
            q = (x, y)
            returns if q is in an obstacle
        """
        return self.env.env.wrapped_env._is_in_collision(q)
    
    def extend(self, q1, q2, distance_per_check = 2):
        """
        q1,q2 = (x1, y1), (x2, y2)
        returns points on path between q1,q2, used for collision checking
        """
        distance = self.distance(q1,q2)
        checks = math.ceil(distance / distance_per_check) + 2 # also check endpoints
        collision_points = list(zip(np.linspace(q1[0], q2[0], checks),
                               np.linspace(q1[1], q2[1], checks)))
        return collision_points

        # Check for collision at each point
        # for point in collision_points:
        #     if self.collision(point):
        #         return False

        # return True

    def grow_prm(self, n_samples):
        """
            Adds n_samples to prm,
            connect_distance specifies radius to try to connect points
        """
        samples = [self.sample_free_point() for i in range(n_samples)]
        self.prm.grow(samples)

    
    def sample_free_point(self):
        q = self.sample_point_in_map()
        while self.collision(q):
            q = self.sample_point_in_map()
        return q

    def sample_point_in_map(self):
        """
        """
        x = np.random.uniform(self.min_x, self.max_x)
        y = np.random.uniform(self.min_y, self.max_y)
        return (x,y)

    def plot_prm(self):
        # Plot Rectangles
        fig,ax = plt.subplots(1)
        rectangles = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.map[i][j] == 1: # Check for barier
                    side_length = self.maze_env._maze_size_scaling
                    x = j * self.maze_env._maze_size_scaling - self.maze_env._init_torso_x - side_length / 2
                    y = i * self.maze_env._maze_size_scaling - self.maze_env._init_torso_y - side_length / 2
                    rectangle = Rectangle((x,y), side_length, side_length)
                    rectangles.append(rectangle)
        pc = PatchCollection(rectangles, facecolor='brown', alpha=.4, edgecolor='None')
        ax.add_collection(pc)

        # Plot edges
        for edge in self.prm.edges:
            plt.plot([edge.v1.q[0], edge.v2.q[0]], [edge.v1.q[1], edge.v2.q[1]])

        ax.autoscale()
        plt.show()
        return fig, ax




if __name__ == "__main__":
    env = gym.make('antmaze-large-play-v0')
    ant_prm = AntPRM(env, connect_distance=1)
    ant_prm.grow_prm(n_samples=2000)
    #import pdb; pdb.set_trace()

    # Side by side of env
    env.reset()
    env.render()

    env.viewer.cam.elevation = -90
    env.viewer.cam.distance = env.model.stat.extent

    env.reset()
    env.render()

    #Plot PRM
    ant_prm.plot_prm()

        