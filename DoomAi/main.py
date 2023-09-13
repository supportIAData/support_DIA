import os
import cv2
import time
import random
import numpy as np
from gym import Env
from vizdoom import *
from matplotlib import pyplot as plt
from gym.spaces import Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


# SCENARIO_CONFIG = 'scenarii/basic.cfg'  # Basic scenario for first step of training.
# SCENARIO_CONFIG = 'scenarii/defend_the_center.cfg'  # Scenario with many monsters + 100% death.
# SCENARIO_CONFIG = 'scenarii/deadly_corridor_doom_skill_1.cfg'  # Curriculum Learning Step 1.
# SCENARIO_CONFIG = 'scenarii/deadly_corridor_doom_skill_2.cfg'  # Curriculum Learning Step 2.
SCENARIO_CONFIG = 'scenarii/deadly_corridor_doom_skill_3.cfg'  # Curriculum Learning Step 3.
# SCENARIO_CONFIG = 'scenarii/deadly_corridor_doom_skill_4.cfg'  # Curriculum Learning Step 4.
# SCENARIO_CONFIG = 'scenarii/deadly_corridor.cfg'  # Curriculum Learning Step 5.
NUMBER_OF_EPISODES = 5


# VizDoom integration in GYM OpenAI.
class VizDoomGym(Env):
    # Env starting method
    def __init__(self, render=False, config_path=SCENARIO_CONFIG):
        super().__init__()  # Inheritate from mother Class.
        self.game = DoomGame()
        self.game.load_config(config_path)

        # Setting render to False makes calculus way faster.
        if not render:
            self.game.set_window_visible(False)

        # The game has to be initiated after the rendering has been designed.
        self.game.init()

        # Defining observation_space params.
        self.observation_space = Box(
            low=0, high=255, shape=(100, 160, 1), dtype=np.uint8
        )

        # On définit l'espace d'action
        # self.action_space = Discrete(3)  # Configuration for most scenarii.
        self.action_space = Discrete(7)   # Configuration for the deadly_corridor scenario. Else, 3.

        # Games variables for the Curriculum learning => deadly_corridor scenario
        self.damage_taken = 0
        self.damage_count = 0
        self.ammo = 52

    # comment on fait un "step" dans l'environnement
    def step(self, action):
        # Spécifie les actions possibles et fait une action
        # actions = np.identity(3, dtype=np.uint8)  # Configuration for most scenarii.
        actions = np.identity(7, dtype=np.uint8)  # Configuration for the deadly_corridor scenario. Else, 3.
        movement_reward = self.game.make_action(actions[action], 4)
        reward = 0

        # Récupère les informations à retourner
        # Gère le cas ou le niveau est fini en retournant une image par défaut et l'info à 0
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)

            # if (state)

            # Reward shaping
            game_variables = self.game.get_state().game_variables
            health, damage_taken, damage_count, ammo = game_variables
            # print(health, damage_taken, damage_count, ammo)

            # Calculate reward deltas
            damage_taken_delta = self.damage_taken - damage_taken
            self.damage_taken = damage_taken
            damage_count_delta = damage_count - self.damage_count
            self.damage_count = damage_count
            ammo_delta = ammo - self.ammo
            self.ammo = ammo

            # movement_reward * 1.5, * 1.25, * 1.05, * 1.01 tested
            reward = movement_reward + damage_taken_delta*10 + damage_count_delta*200 + ammo_delta*5

            info = self.game.get_state().game_variables[0]  # ammo
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0

        info = {"info": info}
        done = self.game.is_episode_finished()

        return state, reward, done, info

    # Method defining the behavior when we reset the game
    def reset(self):
        # Reset des valeurs des variables
        self.damage_taken = 0
        self.damage_count = 0
        self.ammo = 52

        # Définition du state
        state = self.game.new_episode().get_state().screen_buffer
        state = self.grayscale(state)

        return state

    # permet de passer le jeu en echelle de gris et de le resizer pour avoir moins de pixels à processer
    @staticmethod
    def grayscale(observation):
        gray = cv2.cvtColor(
            np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY
        )  # on change l'axe de notre image pour que cv2 focntionne correctement puis on la fait en gris
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))

        return state

    # Method to close the game
    def close(self):
        self.game.close()


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self): 
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


# CHECKPOINT_DIR = 'train/train_basic'
# LOG_DIR = 'logs/log_basic'
# CHECKPOINT_DIR = 'train/train_defend'
# LOG_DIR = 'logs/log_defend'
# CHECKPOINT_DIR = 'train/train_corridor/curriculum_2'
# LOG_DIR = 'logs/log_corridor/curriculum_2'
CHECKPOINT_DIR = 'train/train_corridor/curriculum_3'
LOG_DIR = 'logs/log_corridor/curriculum_3'
# CHECKPOINT_DIR = './train/train_corridor/curriculum_4/test_3'
# LOG_DIR = 'logs/log_corridor/curriculum_4/test_3'
# CHECKPOINT_DIR = './train/train_corridor/curriculum_5/base'
# LOG_DIR = 'logs/log_corridor/curriculum_5/base'

# models_path = 'train/train_corridor/curriculum_4/test_3/'
# last_model_path = str(os.listdir(models_path)[1])  # 1 => best_model_100000.zip for train_defend
# last_model_path = str(os.listdir(models_path)[34])  # 34 => best_model_400000.zip for curriculum_1 & 4
# last_model_path = str(os.listdir(models_path)[30])  # 30 => best_model_370000.zip for curriculum_2 & 3
# print(last_model_path)

# callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# env = VizDoomGym(config_path=SCENARIO_CONFIG)
# learning_model = PPO(
#     'CnnPolicy',
#     env,
#     tensorboard_log=LOG_DIR,
#     verbose=1,
#     learning_rate=0.00001,
#     n_steps=8192,
#     clip_range=.1,
#     gamma=.95,
#     gae_lambda=.9)

# learning_model.load(models_path + last_model_path)
# learning_model.set_env(env)
# learning_model.learn(total_timesteps=400000, callback=callback)

# import model to evaluate

# models_path = 'train/train_basic/'
# models_path = 'train/train_defend/'
# models_path = 'train/train_corridor/'
models_path = 'train/train_corridor/curriculum_4/test/'
print(os.listdir(models_path)[1])  # 100000 is in 2nd position of the array
last_model_path = str(os.listdir(models_path)[34])  # 34 == 400 000 // 6 == 300 000 et qqch
print(last_model_path)
model = PPO.load(models_path + last_model_path)
print(models_path + last_model_path)
env = VizDoomGym(render=True)
# mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)


for episode in range(NUMBER_OF_EPISODES):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        time.sleep(0.05)
        total_reward += reward
    print('Total reward for the episode {} is {}'.format(episode, total_reward))
    time.sleep(2)
