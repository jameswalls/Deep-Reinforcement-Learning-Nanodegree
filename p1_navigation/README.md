# Udacity: DRL Project 1 - Navigation

### Overview
Train an agent capable of getting an average reward of 13 over 100 episodes of the given environment. You can watch the end result [here!](https://youtu.be/wGq79WkGFRg).

### Environment Description
Conceptually, the environment consists of a square space in which yellow and blue bananas randomly spawn. To an agent the environment is part of a 37 dimensional vector that includes ray signals of "viewing" bananas, it also includes information about its velocity. The agent is, uh, something that picks a bananas if it moves over it, recieving a reward of +1 if it picks a yellow one and -1 if it picks a blue one. No reward, 0, is recieved if no banana is picked when an action is taken.

The avaliable actions for the agent are:
- ```0```: forward
- ```1```: move backward
- ```2```: turn left
- ```3```: turn right

### Gerring Started
**Set-up the environment**
1. First follow the **Dependencies** on [THIS](https://github.com/udacity/deep-reinforcement-learning#dependencies) to generate a python environment.
2. Then donwload the [Bananna.app](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation) and place it on your working directory for the project.

**Run the environment!**
Open the Report.ipynb notebook and input the path to the Banana.app on the second cell of code. To train the agent simply run the remaining cells upto and including the call to train_agent. By running the code the agent should train until it reaches the above 13 average score.
A plot of the training is showed and then you can run the environment again to watch the trained agent in action!


The notebook orchestrates the proper functions generated in the 3 '.py' files:
- **dqn_agent.py**: the Agent class which is used to interact with the environment and learn from it.
- **train_agent.py:**: the train function to carry out deep Q-learning.
- **model.py**: the Q-neural network model used by the agent to take desicions.

For further details on the impolementation look into the notebook.