import random

class Goals:
    def __init__(self, env=None,goal_sets = None):
        self.env = env
        self.goal_sets = goal_sets

    def get_goals(self):
        return self.goals

    def set_goals(self, goals):
        self.goals = goals

    def sample(self):  
        #choose m many goals from the m-th goal set
        goals = []
        for goal_set in self.goal_sets:
            goals.append(random.choice(list(goal_set)))
        return goals

