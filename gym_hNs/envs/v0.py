import gym
from gym import error, spaces, utils
import numpy as np
from gym.utils import seeding

class V0(gym.Env):
  metadata = {'render.modes': ['human'],
              'video.frames_per_second': 50
    }

  def __init__(self, n_teams = 2, n_agents = [1,1]):
    self.inertia = False
    self.agent_mass = 1.0
    self.delta_time = 1/60  # 1/fps?
    self.full_observable = True
    self.delay_attack = 0
    self.cooldown_attack = 1  # seconds?
    self.cooldown_speed_penalty = 1
    self.speed_penalty_multiplier = 0.5
    self.damage = 0.5  # global for now

    self.n_teams = n_teams
    self.n_agents = n_agents
    assert (len(self.n_agents) == self.n_teams)

    self.agents = []
    for i in range(self.n_teams):
        for j in range(self.n_agents[i]):
            self.agents.append(Agent(team = i, Id=str(i)+str(j),position = [50,50],direction=0))

    # action space: movement xy (2) (2), rotation (2), attack (1) . + NOOP
    self.action_space = spaces.Tuple([spaces.MultiDiscrete([3, 3, 3, 2])
                                      ]*len(self.n_agents)) #for _ in range(len(agents))]


     # observation space: myHealth, attack cooldown, other agents [type (ally/enemy),position]
    self.observation_space =  [spaces.Dict({"myHealth": spaces.Box(low=0,high=1,shape=(1,)),
                                            "cooldown_timer": spaces.Box(low=0, high = self.cooldown_attack,shape=(1,)),
                                            "agents": spaces.Tuple([spaces.Dict({"type":spaces.Discrete(1),
                                                                    "position":spaces.Box(np.array([0,-np.pi]),np.array([1000,np.pi]))
                                                                    })
                                                       ]*(len(self.n_agents)-1))})
                               ]*len(self.agents)

    self.viewer = None

  def step(self, action_n):
      obs_n = []
      reward_n = []
      done_n = []
      #info_n = {'n': []}

      # advance world state
      for i, agent in enumerate(self.agents):
          assert self.action_space[i].contains(action_n[i])
          if agent.cooldown_attack > 0:
              agent.cooldown_attack -= self.delta_time
          if agent.cooldown_attack > 0:
              agent.cooldown_speed_penalty -= self.delta_time

          else:
              if action_n[i][3] == 1: # if attack
                  # ... apply damage
                  agent.cooldown_attack = self.cooldown_attack
                  agent.cooldown_speed_penalty = self.cooldown_speed_penalty

      # rotate
      for i, agent in enumerate(self.agents):
          # 0,1,2 = CW, NOOP, CCW
          agent.direction = (action_n[i][2]-1) * agent.rotation_speed * self.delta_time

      # move
      # TODO: agents must stay in an area
      for i, agent in enumerate(self.agents):
          # 0,1,2 = East, NOOP, West
          agent.position[0] = np.cos(agent.direction) * (action_n[i][0]-1) * agent.speed * self.delta_time
          # 0,1,2 = North, NOOP, South
          agent.position[1] = np.sin(agent.direction) * (action_n[i][1] - 1) * agent.speed * self.delta_time

      # TODO: implement obs, reward, done
      # record observation for each agent
      # for agent in self.agents:
      #     obs_n.append(self._get_obs(agent))
      #     reward_n.append(self._get_reward(agent))
      #     done_n.append(self._get_done(agent))

      return self.agents

  def reset(self):
    # TODO: reset
    pass
    
  def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = screen_width
        scale = screen_width/world_width
        agent_radius = 10

        if self.viewer is None:
            from gym_hNs import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # self.trans = []
            for agent in self.agents:
                agent_trans = rendering.Transform()
                body = rendering.make_circle(agent_radius)
                body.add_attr(agent_trans)
                body.set_color(agent.team,0.5,(1-agent.team)) # assumes 2 teams
                self.viewer.add_geom(body)
                # self.bodies.append(body)
                agent.body = body
                agent.trans = agent_trans



        # if self.state is None:
        #     return None

        for agent in self.agents:
            agent.trans.set_translation(agent.position[0],agent.position[1])
            agent.trans.set_rotation(agent.direction)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


  def close(self):
      if self.viewer:
          self.viewer.close()
          self.viewer = None

class Agent():

    def __init__(self, team, Id, position, direction, health = 1, speed_penalty_mult = 0.5):
        self.team = team
        self.id = Id
        self.alive = True
        self.health = health
        # TODO: make pos and dir spaces.Box, assert
        self.position = position  # initialized by env
        self._direction = direction  # dir=0: looking east -->
        self.base_speed = 10
        self.rotation_speed = 0.1 * (2*np.pi)
        self.speed_penalty_multiplier = speed_penalty_mult
        self.cooldown_speed_penalty = 0
        self.cooldown_attack = 0
        self.body = None
        self.trans = None

    @property
    def speed(self):
        if self.cooldown_speed_penalty > 0:
            return self.base_speed * self.speed_penalty_multiplier
        else:
            return self.base_speed

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
          if np.abs(self.direction) > np.pi:
              self.direction -= np.sign(self.direction)*(2*np.pi)

class Projectile():
    def __init__(self, position, direction, speed, range):
        self.position = position
        # ...
