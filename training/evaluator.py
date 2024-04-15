import gym
import numpy as np
import torch
import wandb
from copy import deepcopy

class Evaluator:
    def __init__(self, dataset, model, config):
        """
        Initialize the evaluator with a specified environment and a model.
        Args:
            model (torch.nn.Module): The model to be evaluated.
            config (dict): Configuration dictionary.
        """
        self.env_name = config["dataset"]["env_name"]
        self.num_eval = config["evaluating"]["num_eval"]
        self.max_episode_length = config["evaluating"]["max_episode_length"]


        self.envs = [gym.make(self.env_name) for _ in range(self.num_eval)]
        self.model = model.to(config["training"]["device"])
        self.model.eval()  # Ensure the model is in evaluation mode
        self.device = config["training"]["device"]
        self.use_wandb = config["wandb"]["log_to_wandb"]
        self.dataset = dataset

    def evaluate(self, current_epoch):
        """
        Evaluate the model across all environments for a given number of episodes.
        Args:
            num_episodes (int): Number of episodes to run for each environment.
        """
        total_rewards = []
        obs_list = [env.reset()[None] for env in self.envs]
        
        #TODO: Init Actions and Reward-to-gos
        
        obs = np.concatenate(obs_list, axis=0)
        recorded_obs = [deepcopy(obs[:, None])]
        dones = [0 for _ in range(self.num_envs)]
        episode_rewards = [0 for _ in range(self.num_envs)]
        episode_steps = [0 for _ in range(self.num_envs)]

        #TODO: Is the stop condition correct ??? 
        while sum(dones) <  self.num_eval:
            if all(episode_steps[i] >= self.max_episode_length for i in range(self.num_eval) if dones[i] == 0):
                # Stop evaluation if all active episodes have reached the horizon limit
                break
            total_reward = 0

            obs = self.dataset.normalizer.normalize(obs, 'observations')
            
            #TODO: Compute Conditions 
            conditions = {}
            samples = self.model.conditional_sample(conditions, returns=returns)
            #TODO: extract actions 
            actions = []
            actions = self.dataset.normalizer.unnormalize(actions, 'actions')


            obs_list = []
            for i in range(self.num_envs):
                if dones[i] == 1:# or episode_steps[i] >= Config.horizon:
                    # Skip evaluation for done episodes or those that reached the horizon
                    obs_list.append(self.envs[i].reset()[None])  # Reset for consistency in shape
                    continue

                this_obs, this_reward, this_done, _ = self.envs[i].step(actions[i])
                obs_list.append(this_obs[None])
                episode_steps[i] += 1  # Increment step count for this episode

                if this_done:# or episode_steps[i] >= Config.horizon:
                    dones[i] = 1
                    episode_rewards[i] += this_reward
                    # logger.print(f"Episode ({i}): {episode_rewards[i]}", color='green')
                else:
                    episode_rewards[i] += this_reward

            obs = np.concatenate(obs_list, axis=0)
            recorded_obs.append(deepcopy(obs[:, None]))            
            

        episode_rewards = np.array(episode_rewards)
        average_ep_reward = np.mean(episode_rewards)
        std_ep_reward = np.std(episode_rewards)
        if self.use_wandb:
            # Log the evaluation metrics to wandb
            wandb.log({"average_ep_reward": average_ep_reward, 
                    "std_ep_reward": std_ep_reward, 
                    "epoch": current_epoch})

    def model_predict(self, obs):
        """
        Predict the action for a given observation using the model.
        Args:
            obs (numpy.ndarray): Observations from the environment.
        """
        # Ensure observation tensor is on the correct device
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            return self.model(obs_tensor)
