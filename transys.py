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
import itertools

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
    def __init__(self, nrows, ncols, obs_ratio=0.2, obstacles=None, goal_setting="random", goal=None):
        self.nrows = int(nrows)
        self.ncols = int(ncols)
        self.actions = ['n','e','s','w']
        self.states = np.arange(1, nrows*ncols+1)
        self.init_state = int(nrows*ncols)
        self.current_state = self.init_state # Current state of gridworld
        if goal_setting == "random":
            self.goal = random.sample(list(self.states[:-1]), 1)[0] # Choose one goal state
        else:
            assert(goal is not None)
            self.goal = int(goal)
        self.transitions = self.grid_transitions()
        super().__init__(self.states, self.actions, self.transitions, self.goal, init_state=self.init_state)
        if obs_ratio == "predef":
            assert obstacles is not None
            self.obstacles = obstacles.copy()
            self.obs_ratio = int(len(self.obstacles)/len(self.states))
        else:
            self.obs_ratio = obs_ratio
            self.obstacles = self.random_gw()
        self.prod_curr_state = [[],[]]
        self.sys_agents = dict()
        self.env_agents = dict()
        self.aux_states = [] # Copy of auxiliary states
        self.sys_graph = None
        self.env_graph = None
        self.aux_graph = None

    def grid_transitions(self):
        transitions = [] # List of tuples
        for row in range(1,self.nrows+1):
            for col in range(1, self.ncols+1):
                state = int(self.ncols*(row - 1) + col)
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
            col = int(self.ncols)
            row = int(state//self.ncols)
        else:
            col = int(state%self.ncols)
            row = int(state//self.ncols + 1)
        return col, row # x,y positions on grid

    # Get gridworld position coordinates:
    def get_pos(self, state):
        col, row = self.get_row_col(state)
        # pdb.set_trace()
        return col-1, self.nrows - row

    # Generates random gridworld:
    def random_gw(self, graph):
        remaining_states = np.array([c for c in self.states if (c!=self.goal) and (c!=self.init_state)])
        nobs = int(np.floor(self.nrows*self.ncols*self.obs_ratio)) # rough number of obstacles
        path_not_found = True # Flag to search for feasible grid config
        counter = 1
        while path_not_found:
            obstacles = random.sample(list(remaining_states), nobs)
            gaux = graph.copy()
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
        # if state == self.init_state:
        #     color = scalarMap.to_rgba(0)
        #     # pdb.set_trace()
        #     ax.add_patch(patches.Rectangle((x, y),.99,.99,color=color,alpha=.5))
        if state == self.goal:
            color = scalarMap.to_rgba(200)
            ax.add_patch(patches.Rectangle((x, y),.99,.99,color=color,alpha=.5))

        if state in self.obstacles:
            color = scalarMap.to_rgba(1)
            ax.add_patch(patches.Rectangle((x, y),.99,.99,color=color,alpha=1))
        else:
            color = scalarMap.to_rgba(100)
            ax.add_patch(patches.Rectangle((x,y),.99,.99,color=color,alpha=.5))

    # Take cross product of states:
    def cross_product(self, type, player="sys"):
        if type == "concurrent":
            if player == "sys":
                agents = self.sys_agents.copy()
                agent_state_temp = [str(s) + "_sys_" for s in self.states if s not in self.obstacles]
            else:
                agents = self.env_agents.copy()
                agent_state_temp = [str(s) + "_env_" for s in self.states if s not in self.obstacles]
            # Creating agent states
            all_states = []
            for agent_name in agents.keys():
                agent = agents[agent_name]
                agent_states = [t + str(agent['id']) for t in agent_state_temp]
                all_states.append(agent_states)
            if self.aux_states == []:
                self.aux_states = list(itertools.product(*all_states))
            else:
                self.aux_states.extend(list(itertools.product(*all_states)))
            # pdb.set_trace()
            self.construct_concurrent_transitions(agents,player=player)

        elif type == "turn":
            if self.sys_graph is None:
                self.set_default_graph(player="sys")
            if self.env_graph is None:
                self.set_default_graph(player="env")
            self.construct_turn_transitions()

        else:
            print("Specify turn or concurrent cross product ")

    # Turn based cross product graph:
    def construct_turn_transitions(self):
        sys_nodes = list(self.sys_graph.nodes())
        env_nodes = list(self.env_graph.nodes())
        self.aux_graph = nx.DiGraph()
        cross_edges = []
        for s in sys_nodes:
            if len((self.sys_agents.keys())) > 1:
                snum = [int(si.split('_')[0]) for si in s]
                s_succ = list(self.sys_graph.successors(s))
                s_succ_num = [[int(ui.split('_')[0]) for ui in u] for u in s_succ]
                # s_succ_num = *s_succ_num.copy()
            else:
                snum = int(s.split('_')[0])
                s_succ = list(self.sys_graph.successors(s))
                s_succ_num = [int(u.split('_')[0]) for u in s_succ]

            for e in env_nodes:
                if len((self.env_agents.keys())) > 1:
                    pdb.set_trace()
                    enum = [int(ei.split('_')[0]) for ei in e]
                    e_succ = list(self.env_graph.successors(e))

                    e_succ_num = [[int(ui.split('_')[0]) for ui in u] for u in e_succ]
                    # e_succ_num = *e_succ_num.copy()
                else:
                    enum = int(e.split('_')[0])
                    e_succ = list(self.env_graph.successors(e))
                    e_succ_num = [int(u.split('_')[0]) for u in e_succ]

                if enum!=snum:
                    start_node = (s,e)
                    for i in range(len(s_succ)):
                        if type(s_succ_num[i]) == list:
                            if enum not in s_succ_num[i]:
                                cross_edges.append((start_node, (s_succ[i], e)))
                        else:
                            if enum != s_succ_num[i]:
                                cross_edges.append((start_node, (s_succ[i], e)))


                    for i in range(len(e_succ)):
                        if type(e_succ_num[i]) == list:
                            if snum not in e_succ_num[i]:
                                cross_edges.append((start_node, (s, e_succ[i])))
                        else:
                            if snum != e_succ_num[i]:
                                cross_edges.append((start_node, (s, e_succ[i])))
        self.aux_graph.add_edges_from(cross_edges)

    # Construct default graph:
    # If no multiple players for each node
    def set_default_graph(self, player):
        suff = "_"+player
        nodes = []
        edges = []
        for s in self.states:
            if s not in self.obstacles:
                node = str(s)+suff
                nodes.append(node)
                successors = self.get_successors(s)
                succ_nodes = [str(v)+suff for v in successors]
                for snode in succ_nodes:
                    edges.append((node, snode))

        if player == "sys":
            self.sys_graph = nx.DiGraph()
            self.sys_graph.add_edges_from(edges)
        else:
            self.env_graph = nx.DiGraph()
            self.env_graph.add_edges_from(edges)

    # Constructing concurrent transitions of one agent:
    def construct_concurrent_transitions(self, agents, player="sys"):
        G = nx.DiGraph()
        cross_edges = [] # List
        for st in self.aux_states:
            st_num = [int(si.split('_')[0]) for si in st]
            st_suffix = ["_"+si.split('_')[1] + "_"+si.split('_')[2] for si in st]
            if len(st_num) == len(set(st_num)):   # No two agents starting in same state
                all_successors_num = [self.get_successors(s) for s in st_num]
                successors_cross = itertools.product(*all_successors_num)
                for succ_candidate in successors_cross:
                    if len(list(succ_candidate)) == len(set(succ_candidate)): # no collision
                        successor = [str(succ_candidate[i]) + st_suffix[i] for i in range(len(succ_candidate))] # Creating successor edge
                        cross_edges.append((st, tuple(successor)))
        # pdb.set_trace()
        G.add_edges_from(cross_edges)
        if player == "sys":
            self.sys_graph = G.copy()
        else:
            self.env_graph = G.copy()

    def add_agents(self, agent):
        '''
        Function for adding agents in the gridworld
        agent: ['name', init_state, actions, type=sys/env]
        '''
        if agent[3] =="sys":
            nagent = len(list(self.sys_agents.keys())) + 1
            self.sys_agents[agent[0]] = dict()
            self.sys_agents[agent[0]]['id'] = nagent
            self.sys_agents[agent[0]]['x0'] = agent[1]
            self.sys_agents[agent[0]]['act'] = agent[2]
            self.prod_curr_state[0].append(agent[1])
        else:
            nagent = len(list(self.env_agents.keys())) + 1
            self.env_agents[agent[0]] = dict()
            self.env_agents[agent[0]]['id'] = nagent
            self.env_agents[agent[0]]['x0'] = agent[1]
            self.env_agents[agent[0]]['act'] = agent[2]
            self.prod_curr_state[1].append(agent[1])

    def compute_conncurrent_product_transitions(self):
        if len(list(self.sys_agents.keys())) > 1:
            self.cross_product("concurrent", player="sys") # Increase the number of states in game graph
        if len(list(self.env_agents.keys())) > 1:
            self.cross_product("concurrent", player="env") # Increase the number of states in game graph

    def construct_game_graph(self):
        if len(list(self.env_agents.keys())) > 0:
            self.cross_product("turn")
        else:
            self.aux_graph = self.sys_graph.copy()

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
        # pdb.set_trace()
        for s in self.obstacles:
            self.color_square(ax, s)
        # self.color_square(ax,23.0)
        return fig, ax

    # Plot current state of gridworld:
    def plot_agents(self):
        fig, ax = self.plot_gw()
        for si in self.sys_agents.keys():
            agent = self.sys_agents[si]
            agent_id = agent['id']
            sys_state = self.prod_curr_state[0]
            agent_state = sys_state[agent_id-1]
            # pdb.set_trace()
            agent_x, agent_y = self.get_pos(agent_state)
            circle = plt.Circle((agent_x + 0.5, agent_y + 0.5),0.4, fc='tab:orange',ec="tab:orange")
            ax.add_patch(circle)
            plt.text(agent_x+0.25, agent_y+0.25, "S"+str(agent_id))

        for si in self.env_agents.keys():
            agent = self.env_agents[si]
            agent_id = agent['id']
            env_state = self.prod_curr_state[1]
            # pdb.set_trace()
            if type(env_state) == list:
                agent_state = env_state[agent_id-1]
            else:
                agent_state = env_state
            agent_x, agent_y = self.get_pos(agent_state)
            circle = plt.Circle((agent_x + 0.5, agent_y + 0.5),0.4, fc='firebrick',ec="firebrick")
            ax.add_patch(circle)
            plt.text(agent_x+0.25, agent_y+0.25, "E"+str(agent_id))

    def create_players(self, player_list):
        assert player_list != []
        for p in player_list:
            self.add_agents(p)

    # Function to construct all the product graphs()
    def construct_product_graphs(self):
        self.compute_conncurrent_product_transitions()
        print("============= Constructed concurrent product transitions ============")
        self.construct_game_graph()
        print("============= Constructed turn product transitions ============")

# Testing the classes:
if __name__ == '__main__':
    nrows = 10
    ncols = 10
    obstacles = [15,16,17,25,26,27,35,36,37,45,46,47,66,67,68,76,77,78,86,87,88]
    G = GridWorld(nrows, ncols, obs_ratio="predef", obstacles=obstacles, goal_setting="predef", goal=41)

    # Adding agents:
    sys1 = ['sys1', 70, ['n','e','w','s'], "sys"]
    sys2 = ['sys2', 100, ['n','e','w','s'], "sys"]
    env1 = ['env1', 3, ['n','e','w','s'], "env"]
    G.create_players([sys1, sys2, env1])
    G.construct_product_graphs()
    G.plot_agents()
    # fig, ax = G.plot_gw()
    plt.show()
