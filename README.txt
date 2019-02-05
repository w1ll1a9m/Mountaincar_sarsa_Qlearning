Reinforcement learning mountain car problem.
William Lopez

Requirements:

python: 3.6
GYM
NUMPY
matplotlib.pyplot

The following code uses reinforcement learning to solve the mountain car problem. For this particular case 
The state space is discretized, the number of "bins" or resolution can be adjusted by the variable number_states.
The variable mode controls whether SARSA or Q learning will be used to update the q-values. pick 1 for SARSA 0 for q-learning.
The agent will be trained by 10000 episode, this can be changed in the for loop, each episode lasts until the agent achieves the goal.
At the end the resulting q-values, the number of steps per episode along the different episodes will be ploted along with a graphical representation of the best policy accquired.
