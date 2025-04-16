import random
import torch
import numpy as np
import pickle
from Network import FlappyNet


SCREEN_WIDTH, SCREEN_HEIGHT = 400, 600

class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = 0
        self.width = 20
        self.height = 20
        self.gravity = 0.5
        self.alive = True
        self.score = 0

    def update(self):
        if self.alive:
            self.velocity += self.gravity
            self.y += self.velocity

    def flap(self):
        if self.alive:
            self.velocity = -8
            print(f"Agent at x={self.x} flapped!")


class Pipe:
    def __init__(self, x, gap_y, gap_height):
        self.x = x
        self.gap_y = gap_y
        self.width = 50
        self.gap_height = gap_height
        self.passed = False

    def update(self, speed=5):
        self.x -= speed

    def off_screen(self, screen_width):
        return self.x < -50


class Game:
    def __init__(self, agents):
        self.agents = agents
        self.pipes = [Pipe(400, random.randint(100, 300), 200)]
        self.screen_height = SCREEN_HEIGHT
        self.game_over = False

    def update(self):
        # Update agents
        for agent in self.agents:
            if agent.alive:
                agent.update()

        # Update pipes and check collisions
        for pipe in self.pipes:
            pipe.update()
            for agent in self.agents:
                if agent.alive and self.check_collision(agent, pipe):
                    agent.alive = False  # Mark agent as dead if collision occurs

        # Check if any pipe has been passed and is off screen, then update scores
        new_pipes = []
        for pipe in self.pipes:
            if pipe.off_screen(SCREEN_WIDTH):
                if not pipe.passed:
                    pipe.passed = True
                    # Increase score for each alive agent that passed this pipe
                    for agent in self.agents:
                        if agent.alive:
                            agent.score += 1
            else:
                new_pipes.append(pipe)
        self.pipes = new_pipes

        if self.pipes and self.pipes[-1].x < 250:
            self.pipes.append(Pipe(400, random.randint(100, 300), random.randint(150, 250)))

        # End game when all agents are dead
        if not any(agent.alive for agent in self.agents):
            self.game_over = True

    def check_collision(self, agent, pipe):
        if agent.x + agent.width > pipe.x and agent.x < pipe.x + pipe.width:
            if agent.y < pipe.gap_y or agent.y + agent.height > pipe.gap_y + pipe.gap_height:
                return True
        if agent.y < 0 or agent.y + agent.height > SCREEN_HEIGHT:
            return True
        return False


def get_network_inputs(agent, pipes):
    """
    Network inputs
      1. Agent's normalized y position (0-1)
      2. Agent's normalized velocity (scaled)
      3. Horizontal distance to the nearest pipe (normalized)
      4. Nearest pipe's normalized gap_y (top of gap)
      5. Nearest pipe's normalized gap_height
      6. Difference between agent's y and gap center (normalized)
    """
    # Find the nearest pipe ahead of the agent
    nearest_pipe = None
    for pipe in pipes:
        if pipe.x + pipe.width >= agent.x:
            nearest_pipe = pipe
            break
    if nearest_pipe is None:
        nearest_pipe = pipes[0]

    horizontal_distance = nearest_pipe.x - agent.x
    gap_center = nearest_pipe.gap_y + nearest_pipe.gap_height / 2
    vertical_difference = agent.y - gap_center

    inputs = [
        agent.y / SCREEN_HEIGHT,           # Normalize agent's y position
        agent.velocity / 10,                 # Scale agent's velocity
        horizontal_distance / SCREEN_WIDTH,  # Normalize horizontal distance
        nearest_pipe.gap_y / SCREEN_HEIGHT,  # Normalize pipe's gap y position
        nearest_pipe.gap_height / SCREEN_HEIGHT,  # Normalize pipe's gap height
        vertical_difference / SCREEN_HEIGHT  # Normalize difference between agent y and gap center
    ]
    return np.array(inputs, dtype=np.float32)


input_dim = 6
num_agents = 100


# Run simulation for one generation and then return agents and networks
def run_generation(agents, networks):
    game = Game(agents)
    while not game.game_over:
        for _ in range(5):
            for i, agent in enumerate(game.agents):
                if agent.alive:
                    inputs = get_network_inputs(agent, game.pipes)
                    tensor_inputs = torch.tensor(inputs).unsqueeze(0)
                    output = networks[i](tensor_inputs)
                    if output[0, 0].item() > 0.5:
                        agent.flap()
            game.update()

        # Debug output: Print each agent's status (position, velocity, and score)
        print("Agents:")
        for idx, agent in enumerate(game.agents):
            status = "Alive" if agent.alive else "Dead"
            print(f"  Agent {idx}: Y = {agent.y:.2f}, Velocity = {agent.velocity:.2f}, Score = {agent.score} ({status})")
        print("Pipes:")
        for pipe in game.pipes:
            print(f"  Pipe: X = {pipe.x:.2f}, Gap Y = {pipe.gap_y}, Gap Height = {pipe.gap_height}")
        print("-" * 40)

    print("Generation Over! All agents are dead.")
    return game.agents, networks


# Function to create the next generation using the two best agents' networks.
def next_generation(old_agents, old_networks):
    # Sort agents by score in descending order; get their indices.
    indices = sorted(range(len(old_agents)), key=lambda i: old_agents[i].score, reverse=True)
    best1, best2 = indices[0], indices[1]
    print(f"Top agents: {best1} (score {old_agents[best1].score}) and {best2} (score {old_agents[best2].score})")

    # Print out the weights and biases for the top two networks
    print("\nTop 1 Network Weights and Biases:")
    for key, value in old_networks[best1].state_dict().items():
        print(f"{key}: {value}")

    print("\nTop 2 Network Weights and Biases:")
    for key, value in old_networks[best2].state_dict().items():
        print(f"{key}: {value}")

    new_networks = []
    new_agents = []
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


# Initialize the first generation
networks = [FlappyNet(input_dim) for _ in range(num_agents)]
agents = [Agent(50, 300) for _ in range(num_agents)]

# Run through multiple generations (evolve N-1 generations)
num_generations = 20  # Total number of generations you want
for generation in range(num_generations - 1):
    print(f"\n--- Generation {generation} ---")
    agents, networks = run_generation(agents, networks)
    agents, networks = next_generation(agents, networks)

# Run one final generation without evolving afterward to capture performance.
print(f"\n--- Final Evaluation Generation ---")
agents, networks = run_generation(agents, networks)

# Now select the best network from the final generation based on scores.
best_index, best_agent = max(enumerate(agents), key=lambda tup: tup[1].score)
best_network = networks[best_index]
print(f"\nBest network from final evaluation is Agent {best_index} with score {best_agent.score}")

# Save the best network to a pickle file
with open('best_network.pkl', 'wb') as f:
    pickle.dump(best_network.state_dict(), f)
print("Best network saved to 'best_network.pkl'.")

