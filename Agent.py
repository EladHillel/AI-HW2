from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


class Agent:
    # returns the next operator to be applied - i.e. takes one turn
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()

    # returns list of legal operators and matching list of states reached by applying them
    def successors(self, env: WarehouseEnv, robot_id: int):
        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(robot_id, op)
        return operators, children

    def heuristic(self, env: WarehouseEnv, robot_id: int):
        robot = env.get_robot(robot_id)
        other_robot = env.get_robot((robot_id + 1) % 2)
        return robot.credit - other_robot.credit


# picks random operators from the legal ones
class AgentRandom(Agent):
    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)
        rp = self.get_regular_probability(operators)
        probs =  [self.probability(rp,op) for op in operators]
        return random.choices(operators, probs)[0]

    def get_regular_probability(self, operators):
        sum = len(operators)
        if 'move east' in operators:
            sum+=1
        if 'pick up' in operators:
            sum+=1
        return 1/sum

    def probability(self, regular_probability, operator):
        if operator == 'move east' or operator == 'pick up':
            return 2* regular_probability
        return regular_probability


class AgentGreedy(Agent):
    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(robot_id, op)
        children_heuristics = [self.heuristic(child, robot_id) for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]


