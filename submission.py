import time

from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
from functools import reduce


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    this_robot = env.robots[robot_id]
    other_robot = env.robots[1 - robot_id]
    package_reward = lambda package: 2*manhattan_distance(package.position, package.destination)
    best_next_package = lambda robot: max([p for p in env.packages if p.on_board], key=lambda p: package_reward(p)*(10-manhattan_distance(p.position, robot.position)))
    next_package_weight = lambda robot, package: ((10-manhattan_distance(package.position, robot.position))**2)*package_reward(package)
    best_charger = lambda robot: max(env.charge_stations, key = lambda cs:(10 - manhattan_distance(cs.position, robot.position)))
    alreadyWon = lambda: this_robot.credit > other_robot.credit and other_robot.battery == 0
    def goodness(robot):
        sum = 1000*(robot.credit + env.num_steps*robot.battery)
        if (robot.package):
            dist = manhattan_distance(robot.position, robot.package.destination)
            sum += 100*package_reward(robot.package)*(10-dist)
        else:
            sum += next_package_weight(robot, best_next_package(robot))
        if robot.battery < 7 and robot.credit != 0 and not alreadyWon():
            sum += 1000*(10-manhattan_distance(best_charger(robot).position, robot.position))
        return sum
    return goodness(this_robot) - goodness(other_robot)
def not_so_smart_heuristic(env: WarehouseEnv, robot_id: int):
    this_robot = env.robots[robot_id]
    value = 0
    value += 1000*this_robot.credit
    if (this_robot.package is not None):
        value += 250
        value -= manhattan_distance(this_robot.position, this_robot.package.destination)
    else:
        value -= closeset_package_dist(env, robot_id)
    return value
def closeset_package_dist(env: WarehouseEnv, robot_id: int):
    distance = float('inf')
    for pack in env.packages:
        distance = min(distance, manhattan_distance(env.robots[robot_id].position, pack.position))
    return distance

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return not_so_smart_heuristic(env, robot_id)
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        initial_time = time.time()
        time_error = 0.1*time_limit
        depth = 1
        next_solution = None
        solution = None
        try:
            while True:
                _, next_solution = self.run_expectimax(env, agent_id, initial_time + time_limit - time_error, depth, agent_id)
                solution = next_solution
                depth += 1
        except Exception as e:
            # print ("minimax took " + str(time.time()-initial_time) + "seconds with a depth of" + str(depth))
            return solution

    def run_expectimax(self, env: WarehouseEnv, agent_id, time_boundary, depth, turn_id):
        curr_time = time.time()
        if(curr_time >= time_boundary):
            raise TimeoutError

        if(env.done() or depth == 0):
            return self.heuristic(env, agent_id), 'park'

        operators, children = self.successors(env,agent_id)
        children_vals = [self.run_expectimax(child, agent_id, time_boundary, depth - 1, 1 - turn_id)[0] for child in
                         children]

        if turn_id == agent_id:
            best_child_index = max(range(len(children_vals)), key=lambda index: children_vals[index])
            return children_vals[best_child_index], operators[best_child_index]

        else:
            worst_child_index = min(range(len(children_vals)), key=lambda index: children_vals[index])
            return children_vals[worst_child_index], operators[worst_child_index]

class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        initial_time = time.time()
        time_error = 0.25*time_limit
        depth = 1
        next_solution = None
        solution = None
        try:
            while True:
                value, next_solution = self.run_expectimax(env, agent_id, initial_time + time_limit - time_error, depth, agent_id)
                solution = next_solution
                depth += 1
        except Exception as e:
            # print ("expectimax took " + str(time.time()-initial_time) + "seconds with a depth of" + str(depth))
            return solution

    def run_expectimax(self, env: WarehouseEnv, agent_id, time_boundary, depth, turn_id):
        curr_time = time.time()
        if(curr_time >= time_boundary):
            raise TimeoutError

        if(env.done() or depth == 0):
            return self.heuristic(env, agent_id), 'park'

        operators, children = self.successors(env,agent_id)
        children_vals = [self.run_expectimax(child, agent_id, time_boundary, depth - 1, 1 - turn_id)[0] for child in
                         children]

        if turn_id == agent_id:
            best_child_index = max(range(len(children_vals)), key=lambda index: children_vals[index])
            return children_vals[best_child_index], operators[best_child_index]

        else:
            regular_probability = self.get_regular_probability(operators)
            expectation = 0
            for index, value in enumerate(children_vals):
                expectation += value*self.probability(regular_probability, operators[index])
            return expectation, None


    def get_regular_probability(self, operators):
        sum = len(operators)
        if 'move east' in operators:
            sum+=1
        if 'pick up' in operators:
            sum+=1
        return 1/sum

    def probability(self, regular_probability, operator):
        if operator == 'move east' or operator == 'pick up':
            return 2*regular_probability
        return regular_probability




# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        # self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
        #                    "move south", "move south", "move south", "move south", "drop_off"]
        self.trajectory = ["move north","move south","move north","move south","move north","move south","move north","move south",
                           "move north","move south","move north","move south","move north","move south","move north","move south",
                           "move north","move south","move north","move south","move north","move south","move north","move south",
                           "move north","move south","move north","move south","move north","move south","move north","move south",
                           "move north","move south","move north","move south","move north","move south","move north","move south",
                           "move north","move south","move north","move south","move north","move south","move north","move south"]
        
    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)