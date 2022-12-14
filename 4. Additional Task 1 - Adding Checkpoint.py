"""
A* grid planning
author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)
See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)
This is the simple code for path planning class
"""



import math

import matplotlib.pyplot as plt

import numpy as np

show_animation = True
pn = int(input("passenger number (per week):"))
max_f = int(input("maximum flight:"))
time_cost = str(input("Time cost ('high', 'medium' or 'low'):"))
fuel_cost = eval(input("Fuel cost($/kg):"))
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
cost = 1

class AStarPlanner:
    def __init__(self, ox, oy, resolution, rr, fc_x, fc_y, tc_x, tc_y, jc_x, jc_y):
        """
        Initialize grid map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution # get resolution of the grid
        self.rr = rr # robot radius
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model() # motion model for grid search expansion
        self.calc_obstacle_map(ox, oy)

        self.fc_x = fc_x
        self.fc_y = fc_y
        self.tc_x = tc_x
        self.tc_y = tc_y
        self.jc_x = jc_x
        self.jc_y = jc_y

        self.Delta_C1 = 0.2 # cost intensive area 1 modifier
        self.Delta_C2 = 0.4 # cost intensive area 2 modifier
        self.Delta_C3 = -0.05 # jet stream modifier

        self.costPerGrid = 1 


    class Node: # definition of a sinle node
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        global cost
        """
        A star path search
        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]
        output:
            rx1: x position list of the final path
            ry1: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x), # calculate the index based on given position
                               self.calc_xy_index(sy, self.min_y), 0.0, -1) # set cost zero, set parent index -1
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x), # calculate the index based on given position
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict() 
        # open_set: node not been tranversed yet. closed_set: node have been tranversed already
        open_set[self.calc_grid_index(start_node)] = start_node # node index is the grid index

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(self, goal_node, open_set[o]))
                # g(n) and h(n): calculate the distance between the goal node and openset
            current1 = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current1.x, self.min_x),
                         self.calc_grid_position(current1.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            # reaching goal
            if current1.x == goal_node.x and current1.y == goal_node.y:
                print("First Trip time required -> ",current1.cost )
                cost = current1.cost
                goal_node.parent_index = current1.parent_index
                goal_node.cost = current1.cost
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
            
            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current1

            # print(len(closed_set))

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion): # tranverse the motion matrix
                node = self.Node(current1.x + self.motion[i][0],
                                 current1.y + self.motion[i][1],
                                 current1.cost + self.motion[i][2] * self.costPerGrid, c_id)
                
                ## add more cost in cost intensive area 1
                if self.calc_grid_position(node.x, self.min_x) in self.tc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.tc_y:
                        # print("cost intensive area!!")
                        node.cost = node.cost + self.Delta_C1 * self.motion[i][2]
                
                # add more cost in cost intensive area 2
                if self.calc_grid_position(node.x, self.min_x) in self.fc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.fc_y:
                        # print("cost intensive area!!")
                        node.cost = node.cost + self.Delta_C2 * self.motion[i][2]
                
                # reduce cost in jet stream
                if self.calc_grid_position(node.x, self.min_x) in self.tc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.tc_y:
                        # print("jet stream!!")
                        node.cost = node.cost + self.Delta_C3 * self.motion[i][2]
                
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

        rx1, ry1 = self.calc_final_path(goal_node, closed_set)
        # print(len(closed_set))
        # print(len(open_set))

        return rx1, ry1
        

    def planning(self, gx, gy, gx2, gy2):
        global cost
        """
        A star path search
        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]
        output:
            rx2: x position list of the final path
            ry2: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(gx, self.min_x), # calculate the index based on given position
                               self.calc_xy_index(gy, self.min_y), 0.0, -1) # set cost zero, set parent index -1
        goal_node = self.Node(self.calc_xy_index(gx2, self.min_x), # calculate the index based on given position
                              self.calc_xy_index(gy2, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict() 
        # open_set: node not been tranversed yet. closed_set: node have been tranversed already
        open_set[self.calc_grid_index(start_node)] = start_node # node index is the grid index

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(self, goal_node, open_set[o]))
                # g(n) and h(n): calculate the distance between the goal node and openset
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
                print("Second Trip time required -> ",current.cost )
                cost = cost + current.cost
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
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
            
            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # print(len(closed_set))

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion): # tranverse the motion matrix
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2] * self.costPerGrid, c_id)
                
                ## add more cost in cost intensive area 1
                if self.calc_grid_position(node.x, self.min_x) in self.tc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.tc_y:
                        # print("cost intensive area!!")
                        node.cost = node.cost + self.Delta_C1 * self.motion[i][2]
                
                # add more cost in cost intensive area 2
                if self.calc_grid_position(node.x, self.min_x) in self.fc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.fc_y:
                        # print("cost intensive area!!")
                        node.cost = node.cost + self.Delta_C2 * self.motion[i][2]
                
                # reduce cost in jet stream
                if self.calc_grid_position(node.x, self.min_x) in self.tc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.tc_y:
                        # print("jet stream!!")
                        node.cost = node.cost + self.Delta_C3 * self.motion[i][2]
                
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

        rx2, ry2 = self.calc_final_path(goal_node, closed_set)
        # print(len(closed_set))
        # print(len(open_set))

        return rx2, ry2

    def planning(self, gx2, gy2, gx3, gy3):
        global cost   
        """
        A star path search
        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]
        output:
            rx2: x position list of the final path
            ry2: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(gx2, self.min_x), # calculate the index based on given position
                               self.calc_xy_index(gy2, self.min_y), 0.0, -1) # set cost zero, set parent index -1
        goal_node = self.Node(self.calc_xy_index(gx3, self.min_x), # calculate the index based on given position
                              self.calc_xy_index(gy3, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict() 
        # open_set: node not been tranversed yet. closed_set: node have been tranversed already
        open_set[self.calc_grid_index(start_node)] = start_node # node index is the grid index

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(self, goal_node, open_set[o]))
                # g(n) and h(n): calculate the distance between the goal node and openset
            current3 = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current3.x, self.min_x),
                         self.calc_grid_position(current3.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            # reaching goal
            if current3.x == goal_node.x and current3.y == goal_node.y:
                print("Third Trip time required -> ",current3.cost )
                cost = cost + current3.cost
                goal_node.parent_index = current3.parent_index
                goal_node.cost = current3.cost
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
            
            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current3

            # print(len(closed_set))

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion): # tranverse the motion matrix
                node = self.Node(current3.x + self.motion[i][0],
                                 current3.y + self.motion[i][1],
                                 current3.cost + self.motion[i][2] * self.costPerGrid, c_id)
                
                ## add more cost in cost intensive area 1
                if self.calc_grid_position(node.x, self.min_x) in self.tc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.tc_y:
                        # print("cost intensive area!!")
                        node.cost = node.cost + self.Delta_C1 * self.motion[i][2]
                
                # add more cost in cost intensive area 2
                if self.calc_grid_position(node.x, self.min_x) in self.fc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.fc_y:
                        # print("cost intensive area!!")
                        node.cost = node.cost + self.Delta_C2 * self.motion[i][2]
                
                # reduce cost in jet stream
                if self.calc_grid_position(node.x, self.min_x) in self.tc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.tc_y:
                        # print("jet stream!!")
                        node.cost = node.cost + self.Delta_C3 * self.motion[i][2]
                
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

        rx3, ry3 = self.calc_final_path(goal_node, closed_set)
        # print(len(closed_set))
        # print(len(open_set))

        return rx3, ry3
        

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)] # save the goal node as the first point
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
                             for _ in range(self.x_width)] # allocate memory
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x) # grid position calculation (x,y)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                     # Python???s zip() function creates an iterator that will aggregate elements from two or more iterables. 
                    d = math.hypot(iox - x, ioy - y) # The math. hypot() method finds the Euclidean norm
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True # the griid is is occupied by the obstacle
                        break

    @staticmethod
    def get_motion_model(): # the cost of the surrounding 8 points
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion



def main():
    print(__file__ + " start the A star algorithm demo !!") # print simple notes

    # start and goal position
    sx = 0.0  # [m]
    sy = -5.0  # [m]
    gx = 5.0  # [m]
    gy = 30.0  # [m]
    gx2 = 20.0
    gy2 = 2.0
    gx3 = 35.0
    gy3 = 45.0
    grid_size = 1  # [m]
    robot_radius = 1.0  # [m]

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
        
    

    # set cost intesive area 1
    tc_x, tc_y = [], []
    for i in range(0, 10):
        for j in range(20, 35):
            tc_x.append(i)
            tc_y.append(j)
    
    # set cost intesive area 2
    fc_x, fc_y = [], []
    for i in range(15, 30):
        for j in range(0, 20):
            fc_x.append(i)
            fc_y.append(j)
    
    # set the jet stream
    jc_x, jc_y = [], []
    for i in range(-10, 60):
        for j in range(18, 23):
            jc_x.append(i)
            jc_y.append(j)


    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k") # plot the obstacle        
        plt.plot(fc_x, fc_y, "oy") # plot the cost intensive area 1
        plt.plot(tc_x, tc_y, "or") # plot the cost intensive area 2
        plt.plot(jc_x, jc_y, "og") # plot the jet stream
        plt.plot(sx, sy, "og") # plot the start position 
        plt.plot(gx, gy, "og") # plot the checkpoint1 position
        plt.plot(gx2, gy2, "og") # plot the checkpoint2 position
        plt.plot(gx3, gy3, "og") # plot the end position

        plt.grid(True) # plot the grid to the plot panel
        plt.axis("equal") # set the same resolution for x and y axis 

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()


    a_star = AStarPlanner(ox, oy, grid_size, robot_radius, fc_x, fc_y, tc_x, tc_y, jc_x, jc_y)
    rx1, ry1 = a_star.planning(sx, sy, gx, gy)
    rx2, ry2 = a_star.planning(gx, gy, gx2, gy2)
    rx3, ry3 = a_star.planning(gx2, gy2, gx3, gy3)
    print("Total Trip time required -> ",cost )
    if time_cost == "low":
        if A321_fn > max_f:
            cost_A321 = "Not available"
        else:
            cost_A321 = (A321_fc*cost*fuel_cost + A321_tc_low*cost + A321_fix_c)*A321_fn
        if A330_fn > max_f:
            cost_A330 = "Not available"
        else:
            cost_A330 = (A330_fc*cost*fuel_cost + A330_tc_low*cost + A330_fix_c)*A330_fn
        if A350_fn > max_f:
            cost_A350 = "Not available"
        else:
            cost_A350 = (A350_fc*cost*fuel_cost + A350_tc_low*cost + A350_fix_c)*A350_fn
        print("A321:{}$\nA330:{}$\nA350:{}$".format(cost_A321,cost_A330,cost_A350))
        

    elif time_cost == "medium":
        if A321_fn > max_f:
            cost_A321 = "Not available"
        else:
            cost_A321 = (A321_fc*cost*fuel_cost + A321_tc_medium*cost + A321_fix_c)*A321_fn
        if A330_fn > max_f:
            cost_A330 = "Not available"
        else:
            cost_A330 = (A330_fc*cost*fuel_cost + A330_tc_medium*cost + A330_fix_c)*A330_fn
        if A350_fn > max_f:
            cost_A350 = "Not available"
        else:
            cost_A350 = (A350_fc*cost*fuel_cost + A350_tc_medium*cost + A350_fix_c)*A350_fn
        print("A321:{}$\nA330:{}$\nA350:{}$".format(cost_A321,cost_A330,cost_A350))
        

    elif time_cost == "high":
        if A321_fn > max_f:
            cost_A321 = "Not available"
        else:
            cost_A321 = (A321_fc*cost*fuel_cost + A321_tc_high*cost + A321_fix_c)*A321_fn
        if A330_fn > max_f:
            cost_A330 = "Not available"
        else:
            cost_A330 = (A330_fc*cost*fuel_cost + A330_tc_high*cost + A330_fix_c)*A330_fn
        if A350_fn > max_f:
            cost_A350 = "Not available"
        else:
            cost_A350 = (A350_fc*cost*fuel_cost + A350_tc_high*cost + A350_fix_c)*A350_fn
        print("A321:{}$\nA330:{}$\nA350:{}$".format(cost_A321,cost_A330,cost_A350))
        
            
    else:
        print("You should input 'low', 'medium', or 'high'in 'Time cost:")

    if show_animation:  # pragma: no cover
        plt.plot(rx1, ry1, "-r") # show the route 
        plt.plot(rx2, ry2, "-r")
        plt.plot(rx3, ry3, "-r")
        plt.pause(0.001) # pause 0.001 seconds
        plt.show() # show the plot



if __name__ == '__main__':
    main()
