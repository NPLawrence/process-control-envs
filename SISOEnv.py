"""
A simple Gym environment designed for PI control of single-input single-output (SISO) systems
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import control
import matplotlib.pyplot as plt

class SISOEnv(gym.Env):

  metadata = {'render.modes': ['human']}

  def __init__(self, timestep=0.1, setpoint = 1.0, render_env=False):
    self.num = [0, 1] #numerator of continuous-time transfer function
    self.den = [1, 2, 1] #denominator
    self.ts = timestep
    self.sp = setpoint
    self.render_env = render_env
    
    # we specify a continuous-time system above for ease, but are controlling a
    # discrete-time system: self.sys
    self.sysc = control.tf( self.num, self.den )
    self.sys = control.c2d(self.sysc, self.ts, method='zoh')
    self.ss_sys = control.tf2ss(self.sys)
    self.x = np.zeros(np.transpose(self.ss_sys.C).shape)

    self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))

  def reward(self):

    # return -np.abs(self.y - self.sp)**2 - 0.01*np.abs(self.u)
    return np.abs(self.y - self.sp)

  def step(self, action):

    self.t += 1

    self.action = action
    self.u = self.action
    
    self.x = np.dot(self.ss_sys.A, self.x) + np.dot(self.ss_sys.B, self.u)
    self.y = np.dot(self.ss_sys.C, self.x).item() + np.random.normal(0,0.01)

    self.error = self.sp - self.y
    self.int += self.ts*self.error

    self.state = np.array([self.error, self.int])

    self.error_hist = np.append(self.error_hist, self.error)

    terminated = False
    if self.t>=20:
      self.error_hist = np.delete(self.error_hist, 0)
      if np.max(np.abs(self.error_hist)) < 0.1:
        terminated = True
        

    return self.state, self.reward(), terminated, False,  {}

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    self.x = np.random.normal(0,0.1,size = np.transpose(self.ss_sys.C).shape)
    # self.action = 0
    self.y = np.dot(self.ss_sys.C, self.x).item()
    self.int = 0

    self.error = self.sp - self.y

    self.state = np.array([self.error, self.int])

    self.error_hist = np.zeros([])
    self.t = 0

    if self.render_env:
      self._create_plot()

    return self.state
  
  def _create_plot(self):

    #initialize the figure info
    with plt.ion():
      self.fig, (self.ax1, self.ax2) = plt.subplots(2,1,sharex=True, layout='constrained')
      self.xdata1, self.ydata1 = [], []
      self.xdata_sp, self.ydata_sp = [], []
      self.xdata2, self.ydata2 = [], []
      self.ax1.set_xlim(0, 10)
      self.ax1_min, self.ax1_max, self.ax2_min, self.ax2_max = self.sp - 1, self.sp + 1, -5, 5
      self.ax1.set_ylim(self.ax1_min, self.ax1_max)
      self.ax2.set_ylim(self.ax2_min, self.ax2_max)
      self.ax1.set_ylabel("Output ($y_t$)")
      self.ax2.set_ylabel("Input ($u_t$)")
      self.ax2.set_xlabel(r"Time step")
      self.ln1, = self.ax1.plot([], [], label = 'output', color = 'tab:blue', linestyle='-', drawstyle='steps-post')
      self.ln_sp, = self.ax1.plot([], [], label = 'setpoint', color = 'darkorange', linestyle='--')
      self.ln2, = self.ax2.plot([], [], color = 'tab:blue', linestyle='-', drawstyle='steps-post')


  def render(self):

    with plt.ion():
      self.xdata1.append(self.t)
      self.ydata1.append(self.y)
      self.xdata_sp.append(self.t)
      self.ydata_sp.append(self.sp)

      if self.t > 0:

        self.xdata2.append(self.t)
        self.ydata2.append(self.u)

        #setting the axis limits in terms of latest data
        self.ax1.set_xlim(0, self.t + 1)

        a, b, c, d = np.max((self.ax1_min, np.min(self.ydata1))),\
          np.min((self.ax1_max, np.max(self.ydata1))), np.max((self.ax2_min, np.min(self.ydata2))), np.min((self.ax2_max, np.max(self.ydata2)))
        ax1_min, ax1_max, ax2_min, ax2_max = a*(1 - 0.10*np.sign(a)), b*(1 + 0.10*np.sign(b)), c*(1 - 0.10*np.sign(c)), d*(1 + 0.10*np.sign(d))
        self.ax1.set_ylim(ax1_min, ax1_max)
        self.ax2.set_ylim(ax2_min, ax2_max)

      self.ln1.set_data(self.xdata1, self.ydata1)
      self.ln_sp.set_data(self.xdata_sp, self.ydata_sp)
      self.ln2.set_data(self.xdata2, self.ydata2)

    plt.pause(1e-6)

    plt.draw()
    return plt 

  def close(self):
    print('close')



if __name__ == '__main__':
  print("Plotting environment")
  from gymnasium.wrappers import TimeLimit

  max_episode_steps = 100
  env = TimeLimit(SISOEnv(render_env=True), max_episode_steps=max_episode_steps)
  observation = env.reset(seed=123)
  # env.render()

  for _ in range(max_episode_steps):

    # action = np.random.choice(np.array([-1.0, 1.0]))
    action = 2.0*observation[0] + 1.0*observation[1]

    observation, reward, terminated, truncated, info = env.step(action)
    plt = env.render()

  plt.show()

    



