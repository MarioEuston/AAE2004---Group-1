"""
A* grid planning
author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)
See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)
This is the simple code for path planning class

Modified by: CHAN Pak Lam, CAI Jia Liang

"""


import math

import random

import matplotlib.pyplot as plt

show_animation = True


class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr, fc_x, fc_y, tc_x, tc_y):
        """
        Initialize grid map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution  # get resolution of the grid
        self.rr = rr  # robot radis
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()  # motion model for grid search expansion
        self.calc_obstacle_map(ox, oy)

        self.fc_x = fc_x
        self.fc_y = fc_y
        self.tc_x = tc_x
        self.tc_y = tc_y

        ############you could modify the setup here for different aircraft models (based on the lecture slide) ##########################
        self.C_F = 1
        self.Delta_F = 1
        self.C_T = 2
        self.Delta_T = 5
        self.C_C = 10

        self.Delta_F_A = 2  # additional fuel
        self.Delta_T_A = 5  # additional time

        self.costPerGrid = self.C_F * self.Delta_F + self.C_T * self.Delta_T + self.C_C

        print("PolyU-A380 cost part1-> ", self.C_F *
              (self.Delta_F + self.Delta_F_A))
        print("PolyU-A380 cost part2-> ", self.C_T *
              (self.Delta_T + self.Delta_T_A))
        print("PolyU-A380 cost part3-> ", self.C_C)

    class Node:  # definition of a sinle node
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search
        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),  # calculate the index based on given position
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)  # set cost zero, set parent index -1
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),  # calculate the index based on given position
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        # open_set: node not been tranversed yet. closed_set: node have been tranversed already
        open_set, closed_set = dict(), dict()
        # node index is the grid index
        open_set[self.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(self, goal_node,
                                                                     open_set[
                                                                         o]))  # g(n) and h(n): calculate the distance between the goal node and openset
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            # reaching goal
            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal with cost of -> ", current.cost)
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # print(len(closed_set))

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):  # tranverse the motion matrix
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2] * self.costPerGrid, c_id)

                # add more cost in time-consuming area
                if self.calc_grid_position(node.x, self.min_x) in self.tc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.tc_y:
                        # print("time consuming area!!")
                        node.cost = node.cost + \
                            self.Delta_T_A * self.motion[i][2]

                # add more cost in fuel-consuming area
                if self.calc_grid_position(node.x, self.min_x) in self.fc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.fc_y:
                        # print("fuel consuming area!!")
                        node.cost = node.cost + \
                            self.Delta_F_A * self.motion[i][2]
                    # print()

                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        # print(len(closed_set))
        # print(len(open_set))

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]  # save the goal node as the first point
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(self, n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        d = d * self.costPerGrid
        return d

    def calc_heuristic_maldis(n1, n2):
        w = 1.0  # weight of heuristic
        dx = w * math.abs(n1.x - n2.x)
        dy = w * math.abs(n1.y - n2.y)
        return dx + dy

    def calc_grid_position(self, index, min_position):
        """
        calc grid position
        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]  # allocate memory
        for ix in range(self.x_width):
            # grid position calculation (x,y)
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                # Python’s zip() function creates an iterator that will aggregate elements from two or more iterables.
                for iox, ioy in zip(ox, oy):
                    # The math. hypot() method finds the Euclidean norm
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        # the griid is is occupied by the obstacle
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():  # the cost of the surrounding 8 points
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1]
                #   , # Disable diagonal movements
                #   [-1, -1, math.sqrt(2)],
                #   [-1, 1, math.sqrt(2)],
                #   [1, -1, math.sqrt(2)],
                #   [1, 1, math.sqrt(2)]
                  ]

        return motion


def main():
    print(__file__ + " start the A star algorithm demo !!")  # print simple notes

    # PARAMETERS
    CANVAS_SIZE_X = 70
    CANVAS_SIZE_Y = 70
    CANVAS_START_X = -10
    CANVAS_START_Y = -10
    FUEL_SIZE_X = 30
    FUEL_SIZE_Y = 30
    FUEL_START_X = random.randint(CANVAS_START_X + 1, CANVAS_START_X + CANVAS_SIZE_X - FUEL_SIZE_X - 1)
    FUEL_START_Y = random.randint(CANVAS_START_Y + 1, CANVAS_START_Y + CANVAS_SIZE_Y - FUEL_SIZE_Y - 1)

    sx = gx = random.randint(CANVAS_START_X + 1, CANVAS_START_X + CANVAS_SIZE_X - 1)
    sy = gy = random.randint(CANVAS_START_Y + 1, CANVAS_START_Y + CANVAS_SIZE_Y - 1)
    while (math.dist([sx, sy], [gx, gy]) < 50.0): # generate goal
        gx = random.randint(CANVAS_START_X + 1, CANVAS_START_X + CANVAS_SIZE_X - 1)
        gy = random.randint(CANVAS_START_Y + 1, CANVAS_START_Y + CANVAS_SIZE_Y - 1)

    grid_size = 1  # [m]
    robot_radius = 0.9  # [m] # Travel within one grid size

    ox, oy = [], []
    for i in range(CANVAS_START_X, CANVAS_START_X + CANVAS_SIZE_X):  # draw the bottom border
        ox.append(i)
        oy.append(CANVAS_START_Y)
    for i in range(CANVAS_START_Y, CANVAS_START_Y + CANVAS_SIZE_Y):  # draw the right border
        ox.append(CANVAS_START_X + CANVAS_SIZE_X)
        oy.append(i)
    for i in range(CANVAS_START_X, CANVAS_START_X + CANVAS_SIZE_X):  # draw the top border
        ox.append(i)
        oy.append(CANVAS_START_Y + CANVAS_SIZE_Y)
    for i in range(CANVAS_START_Y, CANVAS_START_Y + CANVAS_SIZE_Y):  # draw the left border
        ox.append(CANVAS_START_X)
        oy.append(i)

    k = round(CANVAS_SIZE_X * CANVAS_SIZE_Y * (0.3 + random.random() * 0.1)) # Tune the value here to adjust density (number of obstacles to generate)
    tempX = random.choices(range(CANVAS_START_X + 1, CANVAS_START_X + CANVAS_SIZE_X), weights=None, cum_weights=None, k=k)
    tempY = random.choices(range(CANVAS_START_Y + 1, CANVAS_START_Y + CANVAS_SIZE_Y), weights=None, cum_weights=None, k=k)
    for i in range(k):
        # If too close to start/goal point, skip this obstacle. Otherwise add to obstacle list
        if (math.dist([sx, sy], [tempX[i], tempY[i]]) <= 3 or math.dist([gx, gy], [tempX[i], tempY[i]]) <= 3):
            continue
        ox.append(tempX[i])
        oy.append(tempY[i])


    # set fuel consuming area
    fc_x, fc_y = [], []
    for i in range(FUEL_START_X, FUEL_START_X + FUEL_SIZE_X):
        for j in range(FUEL_START_Y, FUEL_START_Y + FUEL_SIZE_Y):
            fc_x.append(i)
            fc_y.append(j)

    # set time consuming area
    tc_x, tc_y = [], []
    # for i in range(10, 20):
    #     for j in range(20, 50):
    #         tc_x.append(i)
    #         tc_y.append(j)

    if show_animation:  # pragma: no cover

        plt.plot(fc_x, fc_y, "oy")  # plot the fuel consuming area
        plt.plot(tc_x, tc_y, "or")  # plot the time consuming area
        plt.plot(ox, oy, ".k")  # plot the obstacle
        plt.plot(sx, sy, "og")  # plot the start position
        plt.plot(gx, gy, "xb")  # plot the end position

        plt.grid(True)  # plot the grid to the plot panel
        plt.axis("equal")  # set the same resolution for x and y axis

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius,
                          fc_x, fc_y, tc_x, tc_y)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")  # show the route
        plt.pause(0.001)  # pause 0.001 seconds
        plt.show()  # show the plot


if __name__ == '__main__':
    main()
