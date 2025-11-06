# Flappy Bird Evolutionary AI

This is a very simple evolutionary neural network that learned to play a flappy bird clone through a simulation.

Using batches of 100 agents, it simulates games for 20 generations. For each generation, after all the agents have died, it scores them and chooses the best agents to reproduce with minor changes to its network, thus evolving the agents to become better at the game as time goes on

## Initial Network
The starting network is a very simple 3 layer (input, 1 hidden, output) network that uses ReLU activation for the input and hidden layers and Sigmoid on the output. 
```
class FlappyNet(nn.Module):
    def __init__(self, input_dim):
        super(FlappyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
```

## Training
For training I created a simulation of my clone with no graphics to improve efficiency while training. Then for 20 generations I created 100 agents per generation to train the network. At the end of each generation we take the 2 best networks and create 50 children of each with slight changes to their parameters, then repeat.
```
for _ in range(len(old_agents)):
        # Choose one of the top two networks as the parent (50/50 chance)
        parent = old_networks[best1] if random.random() < 0.5 else old_networks[best2]
        # Create a new child network
        child = FlappyNet(input_dim)
        # Copy parent's weights
        child.load_state_dict(parent.state_dict())
        # Optionally, mutate the child's weights a bit:
        for param in child.parameters():
            param.data += 0.1 * torch.randn_like(param.data)
        new_networks.append(child)
        new_agents.append(Agent(50, 300))
    return new_agents, new_networks
```
These mutations as we call them are what drive the evolutionary process.

## The Final Network
After all 20 generations have run, we save the best parameters to a pickle file and then test it out in the real clone with full graphics so we can see it in action.
