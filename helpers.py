##########################################
######### Record & Display Video #########
##########################################
import imageio
import time
import numpy as np
import base64
import IPython
import gym
import PIL.Image
import pyvirtualdisplay
import gym.wrappers
import re

# Video 
from pathlib import Path
from IPython import display as ipythondisplay

import os
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'

######### Hyperparameters #########
env_id = 'procgen:procgen-fruitbot-v0'
logs_base_dir = './runs' # Log DIR

# Record video
def record_video(env_id, model, video_length=500, video_folder='./videos'):
  """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """
  eval_env = gym.make(env_id, distribution_mode='easy', render_mode = 'rgb_array')
  eval_env.metadata["render.modes"] = ["human", "rgb_array"]
  eval_env = gym.wrappers.Monitor(env=eval_env, video_callable=lambda episode_id: episode_id == 1, directory=video_folder, force=True)
  _ = eval_env.reset()

  obs = eval_env.reset()
  for _ in range(video_length):
    action, _ = model.predict(obs)
    obs, _, done, _ = eval_env.step(action)
    if done:
      obs = eval_env.reset()
      done = False
  name = eval_env.videos[len(eval_env.videos) - 1][0]
  # Close the video recorder
  eval_env.close()
  
  name = re.findall("openai.*", name)[0]
  return name


## Display video
def show_videos(video_path='', prefix=''):
  html = []
  mp4 = Path(video_path + '/' + prefix)
  video_b64 = base64.b64encode(mp4.read_bytes())
  html.append('''<video alt="{}" 
              loop controls style="height: 400px;">
              <source src="data:video/mp4;base64,{}" type="video/mp4" />
              </video>'''.format(mp4, video_b64.decode('ascii')))
  ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))

def record(model, length=1500):
  name = record_video(env_id, model, video_length=length)
  name = name
  show_videos('videos', prefix=name)

