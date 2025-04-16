import pygame
import pickle
import torch
import numpy as np
import random
from Network import FlappyNet

# Load the saved network state
with open('best_network.pkl', 'rb') as f:
    state_dict = pickle.load(f)

input_dim = 6
# Create a new network instance and load the state dict
best_net = FlappyNet(input_dim)
best_net.load_state_dict(state_dict)
best_net.eval()  # Set network to evaluation mode

# Pygame initialization
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird AI")
clock = pygame.time.Clock()

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE  = (0, 0, 255)
GREEN = (0, 255, 0)
RED   = (255, 0, 0)

# Agent class for the Pygame game
class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 20
        self.height = 20
        self.velocity = 0
        self.gravity = 0.5
        self.alive = True

    def update(self):
        if self.alive:
            self.velocity += self.gravity
            self.y += self.velocity

    def flap(self):
        if self.alive:
            self.velocity = -8

    def draw(self, surface):
        pygame.draw.rect(surface, BLUE, (self.x, self.y, self.width, self.height))

# Pipe class for the Pygame game
class Pipe:
    def __init__(self, x, gap_y, gap_height):
        self.x = x
        self.width = 50
        self.gap_y = gap_y
        self.gap_height = gap_height
        self.passed = False

    def update(self, speed=5):
        self.x -= speed

    def draw(self, surface):
        # Draw the top pipe and bottom pipe as rectangles.
        top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y)
        bottom_rect = pygame.Rect(self.x, self.gap_y + self.gap_height, self.width, SCREEN_HEIGHT - (self.gap_y + self.gap_height))
        pygame.draw.rect(surface, GREEN, top_rect)
        pygame.draw.rect(surface, GREEN, bottom_rect)

    def off_screen(self):
        return self.x < -self.width

# Function to create the network input for the agent
def get_network_inputs(agent, pipes):
    # Find the nearest pipe ahead of the agent.
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

# Collision detection: returns True if the agent collides with a pipe or goes off screen.
def check_collision(agent, pipe):
    if agent.x + agent.width > pipe.x and agent.x < pipe.x + pipe.width:
        if agent.y < pipe.gap_y or agent.y + agent.height > pipe.gap_y + pipe.gap_height:
            return True
    if agent.y < 0 or agent.y + agent.height > SCREEN_HEIGHT:
        return True
    return False

def main():
    # Create an agent controlled by the network
    agent = Agent(50, 300)
    # Create initial pipe
    pipes = [Pipe(400, random.randint(100, 300), random.randint(150, 250))]
    score = 0
    font = pygame.font.SysFont(None, 36)

    running = True
    while running:
        clock.tick(60)  # Run at 60 frames per second
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Use the loaded network to decide whether to flap.
        if agent.alive:
            inputs = get_network_inputs(agent, pipes)
            tensor_inputs = torch.tensor(inputs).unsqueeze(0)
            output = best_net(tensor_inputs)
            if output[0, 0].item() > 0.5:
                agent.flap()

        # Update the agent
        agent.update()

        # Update pipes, check for collisions, and update score
        for pipe in pipes:
            pipe.update()
            if not pipe.passed and pipe.x + pipe.width < agent.x:
                pipe.passed = True
                score += 1
            if check_collision(agent, pipe):
                agent.alive = False

        # Remove pipes that have gone off screen
        pipes = [pipe for pipe in pipes if not pipe.off_screen()]
        # Add a new pipe if needed (if the last pipe is far enough left)
        if len(pipes) == 0 or pipes[-1].x < 250:
            new_gap_y = random.randint(100, 300)
            new_gap_height = random.randint(150, 250)
            pipes.append(Pipe(SCREEN_WIDTH, new_gap_y, new_gap_height))

        # Rendering
        screen.fill(WHITE)
        # Draw agent
        agent.draw(screen)
        # Draw pipes
        for pipe in pipes:
            pipe.draw(screen)
        # Draw the score on the screen
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (10, 10))
        pygame.display.flip()

        # If the agent is dead, end the game loop
        if not agent.alive:
            running = False

    pygame.quit()

if __name__ == "__main__":
    main()
