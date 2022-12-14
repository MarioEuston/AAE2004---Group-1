"""

Probabilistic Road Map (PRM) Planner

author: Atsushi Sakai (@Atsushi_twi)

modified by: CHAN Pak Lam, CAI Jia Liang
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# parameter
N_SAMPLE = 500  # number of sample_points
N_KNN = 10  # number of edge from one sampled point
MAX_EDGE_LEN = 30.0  # [m] Maximum edge length

show_animation = True


pn = int(input("Passenger Number (per week):"))
max_f = int(input("Maximum Flight (per week):"))
time_cost = str(input("Time Cost (Please enter 'low', 'medium', or 'high'):"))
fuel_cost = eval(input("Fuel cost($/kg):"))



class Node:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," +\
               str(self.cost) + "," + str(self.parent_index)


def prm_planning(start_x, start_y, goal_x, goal_y,
                 obstacle_x_list, obstacle_y_list, robot_radius, *, rng=None):
    """
    Run probabilistic road map planning

    :param start_x: start x position
    :param start_y: start y position
    :param goal_x: goal x position
    :param goal_y: goal y position
    :param obstacle_x_list: obstacle x positions
    :param obstacle_y_list: obstacle y positions
    :param robot_radius: robot radius
    :param rng: (Optional) Random generator
    :return:
    """
    obstacle_kd_tree = KDTree(np.vstack((obstacle_x_list, obstacle_y_list)).T)

    sample_x, sample_y = sample_points(start_x, start_y, goal_x, goal_y,
                                       robot_radius,
                                       obstacle_x_list, obstacle_y_list,
                                       obstacle_kd_tree, rng)
    if show_animation:
        plt.plot(sample_x, sample_y, ".b")

    road_map = generate_road_map(sample_x, sample_y,
                                 robot_radius, obstacle_kd_tree)

    rx, ry = dijkstra_planning(
        start_x, start_y, goal_x, goal_y, road_map, sample_x, sample_y)

    return rx, ry


def is_collision(sx, sy, gx, gy, rr, obstacle_kd_tree):
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)
    d = math.hypot(dx, dy)

    if d >= MAX_EDGE_LEN:
        return True

    D = rr
    n_step = round(d / D)

    for i in range(n_step):
        dist, _ = obstacle_kd_tree.query([x, y])
        if dist <= rr:
            return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    # goal point check
    dist, _ = obstacle_kd_tree.query([gx, gy])
    if dist <= rr:
        return True  # collision

    return False  # OK


def generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree):
    """
    Road map generation

    sample_x: [m] x positions of sampled points
    sample_y: [m] y positions of sampled points
    robot_radius: Robot Radius[m]
    obstacle_kd_tree: KDTree object of obstacles
    """

    road_map = []
    n_sample = len(sample_x)
    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)

    for (i, ix, iy) in zip(range(n_sample), sample_x, sample_y):

        dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)
        edge_id = []

        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]

            if not is_collision(ix, iy, nx, ny, rr, obstacle_kd_tree):
                edge_id.append(indexes[ii])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)

    #  plot_road_map(road_map, sample_x, sample_y)

    return road_map


def dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y):
    """
    s_x: start x position [m]
    s_y: start y position [m]
    goal_x: goal x position [m]
    goal_y: goal y position [m]
    obstacle_x_list: x position list of Obstacles [m]
    obstacle_y_list: y position list of Obstacles [m]
    robot_radius: robot radius [m]
    road_map: ??? [m]
    sample_x: ??? [m]
    sample_y: ??? [m]

    @return: Two lists of path coordinates ([x1, x2, ...], [y1, y2, ...]), empty list when no path was found
    """

    start_node = Node(sx, sy, 0.0, -1)
    goal_node = Node(gx, gy, 0.0, -1)

    open_set, closed_set = dict(), dict()
    open_set[len(road_map) - 2] = start_node

    path_found = True

    while True:
        if not open_set:
            print("Cannot find path")
            path_found = False
            break

        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        # show graph
        if show_animation and len(closed_set.keys()) % 2 == 0:
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("goal is found!")
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            print("Total Trip time required -> ",current.cost)
            

            A321_fc = 54
            A330_fc = 84
            A350_fc = 90
            A321_pc = 200
            A330_pc = 300
            A350_pc = 350
            A321_tc_low = 10
            A330_tc_low = 15
            A350_tc_low = 20
            A321_tc_medium = 15
            A330_tc_medium = 21
            A350_tc_medium = 27
            A321_tc_high = 20
            A330_tc_high = 27
            A350_tc_high = 34
            A321_fix_c = 1800
            A330_fix_c = 2000
            A350_fix_c = 2500
            A321_fn = math.ceil(pn/A321_pc)
            A330_fn = math.ceil(pn/A330_pc)
            A350_fn = math.ceil(pn/A350_pc)
            if time_cost == "low":
                if A321_fn > max_f:
                    cost_A321 = "Not available"
                else:
                    cost_A321 = (A321_fc*goal_node.cost*fuel_cost + A321_tc_low*goal_node.cost + A321_fix_c)*A321_fn
                if A330_fn > max_f:
                    cost_A330 = "Not available"
                else:
                    cost_A330 = (A330_fc*goal_node.cost*fuel_cost + A330_tc_low*goal_node.cost + A330_fix_c)*A330_fn
                if A350_fn > max_f:
                    cost_A350 = "Not available"
                else:
                    cost_A350 = (A350_fc*goal_node.cost*fuel_cost + A350_tc_low*goal_node.cost + A350_fix_c)*A350_fn
                print("A321:{}$\nA330:{}$\nA350:{}$".format(cost_A321,cost_A330,cost_A350))
                break

            elif time_cost == "medium":
                if A321_fn > max_f:
                    cost_A321 = "Not available"
                else:
                    cost_A321 = (A321_fc*goal_node.cost*fuel_cost + A321_tc_medium*goal_node.cost + A321_fix_c)*A321_fn
                if A330_fn > max_f:
                    cost_A330 = "Not available"
                else:
                    cost_A330 = (A330_fc*goal_node.cost*fuel_cost + A330_tc_medium*goal_node.cost + A330_fix_c)*A330_fn
                if A350_fn > max_f:
                    cost_A350 = "Not available"
                else:
                    cost_A350 = (A350_fc*goal_node.cost*fuel_cost + A350_tc_medium*goal_node.cost + A350_fix_c)*A350_fn
                print("A321:{}$\nA330:{}$\nA350:{}$".format(cost_A321,cost_A330,cost_A350))
                break

            elif time_cost == "high":
                if A321_fn > max_f:
                    cost_A321 = "Not available"
                else:
                    cost_A321 = (A321_fc*goal_node.cost*fuel_cost + A321_tc_high*goal_node.cost + A321_fix_c)*A321_fn
                if A330_fn > max_f:
                    cost_A330 = "Not available"
                else:
                    cost_A330 = (A330_fc*goal_node.cost*fuel_cost + A330_tc_high*goal_node.cost + A330_fix_c)*A330_fn
                if A350_fn > max_f:
                    cost_A350 = "Not available"
                else:
                    cost_A350 = (A350_fc*goal_node.cost*fuel_cost + A350_tc_high*goal_node.cost + A350_fix_c)*A350_fn
                print("A321:{}$\nA330:{}$\nA350:{}$".format(cost_A321,cost_A330,cost_A350))
                break
        
            else:
                print("You should input 'low', 'medium', or 'high'in 'Time cost:")
           

            break

        # Remove the item from the open set
        del open_set[c_id]
        # Add it to the closed set
        closed_set[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)
            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id)

            if n_id in closed_set:
                continue
            # Otherwise if it is already in the open set
            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            else:
                open_set[n_id] = node

    if path_found is False:
        return [], []

    # generate final course
    rx, ry = [goal_node.x], [goal_node.y]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index

    return rx, ry


def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover

    for i, _ in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]

            plt.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "-k")


def sample_points(sx, sy, gx, gy, rr, ox, oy, obstacle_kd_tree, rng):
    max_x = max(ox)
    max_y = max(oy)
    min_x = min(ox)
    min_y = min(oy)

    sample_x, sample_y = [], []

    if rng is None:
        rng = np.random.default_rng()

    while len(sample_x) <= N_SAMPLE:
        tx = (rng.random() * (max_x - min_x)) + min_x
        ty = (rng.random() * (max_y - min_y)) + min_y

        dist, index = obstacle_kd_tree.query([tx, ty])

        if dist >= rr:
            sample_x.append(tx)
            sample_y.append(ty)

    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)

    return sample_x, sample_y


def main(rng=None):
    print(__file__ + " start!!")

    # start and goal position
    sx = 0.0  # [m]
    sy = -5.0  # [m]
    gx = 35.0  # [m]
    gy = 45.0  # [m]
    robot_size = 1.0  # [m]

# set obstacle positions for group 1
    ox, oy = [], []
    for i in np.arange(-10, 60, 0.1): # draw the button border 
        ox.append(i)
        oy.append(-10.0)
    for i in np.arange(-10, 60, 0.1): # draw the right border
        ox.append(60.0)
        oy.append(i)
    for i in np.arange(-10, 60, 0.1): # draw the top border
        ox.append(i)
        oy.append(60.0)
    for i in np.arange(-10, 60, 0.1): # draw the left border
        ox.append(-10.0)
        oy.append(i)

    for i in np.arange(-10, 25, 0.1): # draw the free border
        ox.append(10.0)
        oy.append(i)

    for i in np.arange(20, 30, 0.1):
        ox.append(i)
        oy.append(4 * i - 60)

    for i in np.arange(25, 35, 0.1):
        ox.append(i)
        oy.append(3/4*i + 2.5)

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^c")
        plt.grid(True)
        plt.axis("equal")

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    rx, ry = prm_planning(sx, sy, gx, gy, ox, oy, robot_size, rng=rng)

    assert rx, 'Cannot found path'

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()
