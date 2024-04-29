import gym
from stable_baselines3 import PPO

# Load the pre-trained model checkpoint
def load_from_hub(repo_id, filename):
    print(f"Loading model from {repo_id}...")
    # Code to load model from the hub
    print("Model loaded successfully.")
    return PPO.load(filename)

# Load the pre-trained model checkpoint
repo_id = "igpaub/ppo-CarRacing-v2"
filename = "ppo-CarRacing-v2.zip"  # Replace {MODEL FILENAME} with the actual filename
model = load_from_hub(repo_id, filename)

# Instantiate the CarRacing environment
env = gym.make("CarRacing-v2", render_mode='human')

# Run the model in the environment
total_reward = 0
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()

# Close the environment
env.close()
print(total_reward)
