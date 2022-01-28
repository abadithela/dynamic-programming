# Apurva Badithela
# CDS 112
# apurva@caltech.edu
# 1/26/22

# This script contains classes and utility functions for setting up discrete transition systems
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.text import OffsetFrom
from matplotlib.patches import Arrow
import pdb

Count_FLG = 100 # So while loops dont take forever
class DTA():
    '''
    Class for constructing discrete transition systems

    Constructor needs the following arguments:
    states: finite set of states coomprising the transition system
    actions: finite set of all actions possible by an agent
    prob_transitions: A tuple (u, v, p, act) where u is the initial state, v is the final state reached from u when the action act is taken, and p is the probability with which v is reached when action act is applied at u. For deterministic transitions, p=1.
    goals: set of states that constitute the goal node; can be an empty set
    init_state: Initial state. default=None

    Attributes:
    * T: Transition system

    Functions:
    * setup_transys: constructing a networkx graph representing the transition system
    * cost_to_go: Initializes the cost-to-go functions
    '''
    def __init__(self, states, actions, prob_transitions, goals, init_state=None):
        self.states = states
        self.actions = actions
        self.probT = prob_transitions
        self.goals = goals
        self.init_state = init_state
        self.T = None
        self.Tdict = dict() # Dictionary containing probability and action information for each edge (keys)
        self.setup_transys()

    # self.T is a graph
    def setup_transys(self):
        self.T = nx.DiGraph()
        self.T.add_nodes_from(self.states)
        for ti in self.probT:
            self.T.add_edge(ti[0], ti[1]) # Adding edge
            self.Tdict[(ti[0], ti[1])] = {'p': ti[2], 'act': ti[3]}

    # Get successors of a node:
    def get_successors(self, node):
        successors = [s for s in self.T.successors(node)]
        return successors

    # Get predecessors:
    def get_predecessors(self, node):
        predecessors = [s for s in self.T.predecessors(node)]
        return predecessors

class GridWorld(DTA):
    def __init__(self, nrows, ncols, obs_ratio=0.2):
        self.nrows = nrows
        self.ncols = ncols
        self.actions = ['n','e','s','w']
        self.states = np.linspace(1, nrows*ncols, nrows*ncols)
        self.init_state = nrows*ncols
        self.goal = random.sample(list(self.states[:-1]), 1)[0] # Choose one goal state
        self.obs_ratio = obs_ratio
        self.transitions = self.grid_transitions()
        super().__init__(self.states, self.actions, self.transitions, self.goal, init_state=self.init_state)
        self.obstacles = self.random_gw()

    def grid_transitions(self):
        transitions = [] # List of tuples
        for row in range(1,self.nrows+1):
            for col in range(1, self.ncols+1):
                state = self.ncols*(row - 1) + col
                # Add north action:
                sN = state - self.ncols
                sE = state + 1
                sS = state + self.ncols
                sW = state - 1
                transitions_state = [(state, sN, 1.0, 'n'), (state, sE, 1.0, 'e'), (state, sS, 1.0, 's'), (state, sW, 1.0, 'w')]
                # Remove infeasible transitions on boundaries
                if (col == 1):
                    transitions_state.remove((state, sW, 1.0, 'w'))
                if (col == self.ncols):
                    transitions_state.remove((state, sE, 1.0, 'e'))
                if (row == 1):
                    transitions_state.remove((state, sN, 1.0, 'n'))
                if (row == self.nrows):
                    transitions_state.remove((state, sS, 1.0, 's'))
                transitions.extend(transitions_state)
        return transitions

    # Function to get row-column from state:
    def get_row_col(self, state):
        if state%self.ncols == 0:
            col = self.ncols
            row = state//self.ncols
        else:
            col = state%self.ncols
            row = state//self.ncols + 1
        return col, row # x,y positions on grid

    # Get gridworld position coordinates:
    def get_pos(self, state):
        col, row = self.get_row_col(state)
        # pdb.set_trace()
        return col-1, self.nrows - row
    # Generates random gridworld:
    def random_gw(self):
        remaining_states = np.array([c for c in self.states if (c!=self.goal) and (c!=self.init_state)])
        nobs = int(np.floor(self.nrows*self.ncols*self.obs_ratio)) # rough number of obstacles
        path_not_found = True # Flag to search for feasible grid config
        counter = 1
        while path_not_found:
            obstacles = random.sample(list(remaining_states), nobs)
            gaux = self.T.copy()
            gaux.remove_nodes_from(obstacles)
            # pdb.set_trace()
            if nx.has_path(gaux, self.init_state, self.goal): # Check if there still exists a path
                path_not_found = False
                self.T = gaux.copy()
                return obstacles
            else:
                counter +=1
            if counter > Count_FLG:
                print("No feasible obstacle configuration found")
                break
        return []

    # Other utility plotting functions:
    def color_square(self, ax, state):
        x, y = self.get_pos(state)
        cmap = cm.get_cmap('plasma', 256)
        cNorm  = colors.Normalize(vmin=0, vmax=256)
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
        # print(scalarMap.get_clim())
        if state == self.init_state:
            color = scalarMap.to_rgba(20)
            # pdb.set_trace()
            ax.add_patch(patches.Rectangle((x, y),.99,.99,color=color,alpha=.5))
        if state == self.goal:
            color = scalarMap.to_rgba(200)
            ax.add_patch(patches.Rectangle((x, y),.99,.99,color=color,alpha=.5))

        if state in self.obstacles:
            color = scalarMap.to_rgba(1)
            ax.add_patch(patches.Rectangle((x, y),.99,.99,color=color,alpha=1))
        else:
            color = scalarMap.to_rgba(100)
            ax.add_patch(patches.Rectangle((x,y),.99,.99,color=color,alpha=.5))

     # Plot gridworld:
    def plot_gw(self):
        fig, ax = plt.subplots()
        xticks = range(self.ncols+1)
        yticks = range(self.nrows+1)
        plt.xticks(xticks)
        plt.yticks(yticks, yticks[::-1])
        plt.grid(True)
        self.color_square(ax,self.goal)
        self.color_square(ax,self.init_state)
        for s in self.obstacles:
            self.color_square(ax, s)
        # self.color_square(ax,23.0)
        plt.show()


# Testing the classes:
if __name__ == '__main__':
    nrows = 10
    ncols = 10
    G = GridWorld(nrows, ncols)
    G.plot_gw()
