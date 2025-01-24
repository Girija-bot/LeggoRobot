import mujoco
import mujoco_viewer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NeuralController(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralController, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.init_weights()

    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def lyapunov_function(state):
    position = state[:model.nq]  # Position
    velocity = state[model.nq:]   # Velocity
    return torch.sum(position**2) + torch.sum(velocity**2)

def lyapunov_derivative(state, next_state):
    V = lyapunov_function(state)
    V_next = lyapunov_function(next_state)
    return V_next - V

def barrier_function(state):
    return torch.sum(1.0 / (state**2 + 1e-5))

# Load the humanoid model
model = mujoco.MjModel.from_xml_path('humanoid.xml')
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)

# Initialize the neural controller
controller = NeuralController(input_size=model.nq + model.nv, output_size=model.nu)
optimizer = optim.Adam(controller.parameters(), lr=0.001)

# Simulation parameters
num_episodes = 5
stability_threshold = 0.01
height_threshold = 0.5
max_steps_per_episode = 500
stable_episodes = 0

# Exploration parameters
epsilon_start = 1.0  # Initial exploration rate
epsilon_end = 0.1    # Final exploration rate
epsilon_decay = 0.995  # Decay rate for exploration

previous_episode_reward = 0
epsilon = epsilon_start

# Main training loop
for episode in range(num_episodes):
    mujoco.mj_resetData(model, data)
    state = torch.FloatTensor(data.qpos.tolist() + data.qvel.tolist())
    done = False
    episode_reward = 0
    step = 0
    prev_x_position = data.qpos[0]

    while not done and step < max_steps_per_episode:
        if not viewer.is_alive:
            print("Viewer window closed, ending simulation.")
            done = True
            break

        # Epsilon-greedy action selection
        action = controller(state)
        action = action.detach().numpy()
        action = 0.02 * action.clip(model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        
        # Add exploration noise based on epsilon
        if np.random.rand() < epsilon:
            noise = np.random.normal(0, 0.01, size=action.shape)  # Gaussian noise
            action += noise
        
        data.ctrl[:] = action

        # Step the simulation and render
        mujoco.mj_step(model, data)
        viewer.render()  # Ensure viewer renders

        next_state = torch.FloatTensor(data.qpos.tolist() + data.qvel.tolist())

        if data.qpos[2] < height_threshold:
            print("Robot has fallen!")
            done = True
            episode_reward -= 50  # Gradual penalty
        else:
            current_x_position = data.qpos[0]
            distance_moved = current_x_position - prev_x_position
            prev_x_position = current_x_position
            
            V_dot = lyapunov_derivative(state, next_state)
            is_stable = (V_dot <= stability_threshold)
            stability_reward = 5.0 if is_stable else -0.1  # More gradual reward/penalty
            
            forward_reward = distance_moved * 2  # Scale distance reward
            
            action_smoothness_penalty = -0.01 * torch.sum((torch.tensor(action) - torch.tensor(data.ctrl))**2).item()
            B_value = barrier_function(state)
            safety_penalty = -5 if B_value < 0.1 else 0
            
            # Total reward calculation
            reward = stability_reward + forward_reward + action_smoothness_penalty + safety_penalty
            
            # Log step details for debugging
            print(f"Episode: {episode + 1}, Step: {step}, X Position: {current_x_position}, Reward: {reward}, B_value: {B_value}")

            # Adjust rewards for episode improvement
            if episode > 0 and episode_reward > previous_episode_reward:
                reward += 1.0  # Small bonus for improvement
            elif episode > 0 and episode_reward < previous_episode_reward:
                reward -= 1.0  # Small penalty for regression

            previous_episode_reward = episode_reward

            optimizer.zero_grad()
            reward_tensor = torch.tensor(reward, requires_grad=True)  # Avoid warnings
            loss = -reward_tensor
            loss.backward()
            optimizer.step()

            state = next_state
            episode_reward += reward
            step += 1

    if not done:
        episode_reward += 5  # Small bonus if the episode was successful

    print(f'Episode {episode + 1}, Total Reward: {episode_reward}, Balanced Steps: {step}')

    if step == max_steps_per_episode and episode_reward >= 0:
        stable_episodes += 1

    # Update epsilon for exploration
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

print(f'Out of {num_episodes} episodes, {stable_episodes} were fully balanced.')
viewer.close()
