import matplotlib.pyplot as plt
import numpy as np
import time
import os
from homework2 import Hw2Env
from dqn import DQN
from config import Config
from dqn import get_target, init_target
import csv
import pandas as pd

def train(config: Config, dqn: DQN):
    env = Hw2Env(n_actions=config.action_dim, render_mode="offscreen")
    rewards_history = []
    rps_history = []
    model_path = "dqn_model.pth"
    history_path = "training_history.csv"
    epsilon_path = "epsilon.txt"
    action_reward_path = "action_reward.csv"

    # Load existing epsilon if available
    if os.path.exists(epsilon_path):
        with open(epsilon_path, "r") as f:
            config.epsilon = float(f.read().strip())
        print(f"Loaded existing epsilon: {config.epsilon:.4f}")

    # Load existing history if available
    if os.path.exists(history_path):
        with open(history_path, "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                rewards_history.append(float(row[1]))  # Cumulative reward
                rps_history.append(float(row[2]))      # RPS

    # Open existing action reward history if available
    if os.path.exists(action_reward_path):
        action_reward_file = open(action_reward_path, "a", newline="")
        action_reward_writer = csv.writer(action_reward_file)
    else:
        action_reward_file = open(action_reward_path, "w", newline="")
        action_reward_writer = csv.writer(action_reward_file)
        action_reward_writer.writerow(["Episode", "Step", "Action", "Reward"])
    

    for episode in range(config.num_episodes):
        env.reset()
        state = env.high_level_state()
        prev_state = state[:4]
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        start_time = time.time()

        while not done:

            concatenated_state = np.concatenate((state, prev_state), axis=None)
            action = dqn.act(concatenated_state)

            next_state, reward, is_terminal, is_truncated = env.step(action)
    
            done = is_terminal or is_truncated
            cumulative_reward += reward
            episode_steps += 1
            concatenated_next_state = np.concatenate((next_state, state[:4]), axis=None)
            
            dqn.replay_buffer.add(concatenated_state, action, reward, concatenated_next_state, done)
            dqn.update()

            prev_state = state[:4]
            state = next_state

            # Save action reward history
            action_reward_writer.writerow([episode, episode_steps, action, reward])
        
        end_time = time.time()
        rewards_history.append(cumulative_reward)
        rps_history.append(cumulative_reward / max(episode_steps, 1))

        # Save history to file
        with open(history_path, "a", newline="") as file:
            writer = csv.writer(file)
            if episode == 0 and not os.path.exists(history_path):  # Add header if new file
                writer.writerow(["Episode", "Cumulative Reward", "RPS"])
            writer.writerow([episode, cumulative_reward, rps_history[-1]])

        # Epsilon Decay
        config.epsilon = max(config.epsilon * config.epsilon_decay, config.epsilon_min)
        with open(epsilon_path, "w") as f:
            f.write(str(config.epsilon))

        # Logging
        avg_reward = np.mean(rewards_history[-100:])
        print(f"Episode={episode}, Reward={cumulative_reward:.4f}, "
              f"AvgReward={avg_reward:.4f}, RPS={rps_history[-1]:.4f}, "
              f"Time={end_time - start_time:.2f}s, Epsilon={config.epsilon:.4f}")

        # Save Model Periodically
        if episode % config.save_interval == 0:
            dqn.save(model_path)
            print(f"Model saved at episode {episode}")

    # Save final model
    dqn.save(model_path)
    print("Final model saved.")

    # Plot results
    plot_results(rewards_history, rps_history)

def plot_results(rewards, rps):
    window_size_reward = 400
    window_size_rps = 1200

    rewards = rewards[1000:]  # Skip first 1000 episodes for better visualization
    rps = rps[1000:]

    plt.figure(figsize=(24, 8))

    # Convert to pandas Series for rolling average
    rewards_series = pd.Series(rewards)
    rps_series = pd.Series(rps)

    # Compute rolling averages
    rewards_smoothed = rewards_series.rolling(window=window_size_reward, min_periods=1).mean()
    rps_smoothed = rps_series.rolling(window=window_size_rps, min_periods=1).mean()

    # Cumulative Reward Plot
    plt.subplot(1, 2, 1)
    plt.plot(rewards_series, alpha=0.3, label="Raw Cumulative Reward", color="gray")
    plt.plot(rewards_smoothed, label=f"Smoothed (window={window_size_reward})", color="blue")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title("Training Progress - Reward")
    plt.ylim(-0.25, 3)
    plt.legend()

    # Rewards Per Step (RPS) Plot
    plt.subplot(1, 2, 2)
    plt.plot(rps_series, alpha=0.3, label="Raw RPS", color="gray")
    plt.plot(rps_smoothed, label=f"Smoothed (window={window_size_rps})", color="green")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards Per Step")
    plt.title("Training Progress - RPS")
    plt.ylim(-0.01, 0.2)
    plt.legend()

    plt.savefig("training_progress.png")
    plt.show()

if __name__ == "__main__":
    config = Config()

    # Load model if exists
    model_path = "dqn_model.pth"
    dqn = DQN(config)
    target_dqn = DQN(config) 
    target_dqn.load_state_dict(dqn.state_dict()) 
    target_dqn.eval()
    init_target(target_dqn)

    if os.path.exists(model_path):
        dqn.load(model_path)
        print("Loaded existing model for continued training.")

    train(config, dqn)
