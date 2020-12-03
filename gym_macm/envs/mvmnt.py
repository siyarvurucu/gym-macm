import gym
from gym import spaces
from gym_macm.settings import fwSettings
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef,
                   b2PolygonShape, b2DistanceSquared, b2Color, b2Vec2)
import random
import numpy as np

class Agent:
    def __init__(self, ID, actor=None):
        self.id = ID
        self.actor = actor
        self.rotation_speed = 0.8 * (2 * np.pi)
        self.force = 20 # walk force

    @property
    def color(self):
        return b2Color(1, 0.2, 0.2)

class Flock(gym.Env):
    name = "Flock"
    description = ("Flock")
    """
    Agents moving to a point as a group. If an agent crashes with 
    another object, it gets negative reward
    """

    def __init__(self, render = False, n_agents = 10, actors = None, time_limit = 60):
        # super(World, self).__init__()
        self.settings = fwSettings
        if render:
            from gym_macm.backends.pyglet_framework import PygletFramework as framework
        else:
            from gym_macm.backends.no_render import NoRender as framework
        self.framework = framework(fwSettings)
        self.framework.env = self
        # Framework.__init__(self)
        self.done = False
        self.n_agents = n_agents

        self.world_width = 30
        self.world_height = 30
        self.start_spread = 20
        self.start_point = [0, 0]

        self.time_limit = time_limit # in seconds
        self.time_passed = 0 # time passed in the env
        self.target = b2Vec2((random.random()-0.5)*100, (random.random()-0.5)*100)
        # game settings
        circle = b2FixtureDef(
            shape=b2CircleShape(radius=0.5
                                ),
            density=1,
            friction=0.3
            )
        self.cooldown_atk = 1  # minimum time between consecutive attacks
        self.cooldown_mov_penalty = 0.5 # movement speed penalty. Maybe due to attacking or getting hit

        # Creating agents
        self.agents = []
        for i in range(self.n_agents[0]):
            x = self.start_spread * (random.random() - 0.5) + self.start_point[0]
            y = self.start_spread * (random.random() - 0.5) + self.start_point[1]
            angle = random.uniform(-1,1) * np.pi
            agent = Agent(ID = str(i), actor = actors[i])
            agent.body = self.framework.world.CreateDynamicBody(
                fixtures=circle,
                position=(x, y),
                angle=angle,
                userData=agent,
                linearDamping=5,
                fixedRotation=True
            )
            self.agents.append(agent)
        self.create_space()
        self.create_space_flag = False

    def step(self, actions=None):
        # print(settings)
        if self.done: # "Episode is finished."
            print("Episode is finished.")
            self.quit() # TODO: is quit() necessary?

        obs = self.get_obs()  # obs of t+1
        if actions==None:
            actions = {}
            for agent in self.agents:
                act = agent.actor({agent.id:obs[agent.id]})
                actions[agent.id] = act

        assert self.action_space.contains(actions)

        # Agent actions
        for agent in self.agents:
                # 0,1,2 = CW, NOOP, CCW
            # if actions[agent.id][2] != 1:
            #     print(agent.direction)

            # Rotation
            agent.body.angle = (agent.body.angle + (actions[agent.id][2]-1) *
                                agent.rotation_speed * (1/self.settings.hz))
            if np.abs(agent.body.angle) > np.pi:
                agent.body.angle -= np.sign(agent.body.angle) * (2 * np.pi)

            # Movement
            angle = agent.body.angle
            c = 1 / np.sqrt(2) if ((actions[agent.id][0] != 1) and (actions[agent.id][1] != 1)) else 1
            x_force = (np.cos(angle)            * (actions[agent.id][0]-1) +
                      np.cos(angle + np.pi / 2) * (actions[agent.id][1] - 1)) * c * agent.force #* (1/self.settings.hz)
            y_force = (np.sin(angle)            * (actions[agent.id][0]-1) +
                      np.sin(angle + np.pi / 2) * (actions[agent.id][1] - 1)) * c * agent.force #* (1/self.settings.hz)

            agent.body.ApplyForce(force=(x_force, y_force), point=agent.body.position, wake=True)


        # super(World, self).Step(self.settings)
        self.framework.Step(self.settings)
        
        rewards = self.get_rewards()
        self.time_passed += (1/self.settings.hz)
        if  (self.time_passed > self.time_limit):
            self.done = True
            print("Time is up")
        return obs, actions, rewards

    def create_space(self):
        self.action_space = spaces.Dict({agent.id: spaces.MultiDiscrete([3, 3, 3])
                                         for agent in self.agents})

        self.observation_space = spaces.Dict(
            {agent.id: spaces.Dict({"nodes": spaces.Tuple([spaces.Dict({"type": spaces.Discrete(1),
                                                                        "id": spaces.Discrete(1),
                                                                        "position": spaces.Box(
                                                                             np.array([0, -np.pi]),
                                                                             np.array([np.inf, np.pi]))
                                                                         })
                                                            ] * (len(self.agents)+1))})
             for agent in self.agents})

    def get_rewards(self):
        rewards = {}
        for contact in self.framework.world.contacts:
            rewards[contact.fixtureA.body.userData.id] = -1
            rewards[contact.fixtureB.body.userData.id] = -1
        for agent in self.agents:
            if agent.id not in rewards:
                r = np.sqrt(b2DistanceSquared(self.target, agent.body.position))
                rewards[agent.id] = (-r/100) + 1
        return rewards

    def get_obs(self):
        obs = {}
        for agent in self.agents:
            obs[agent.id] = {"nodes": []}
            for other_agent in self.agents:
                if agent == other_agent:
                    continue
                rel_position = other_agent.body.position - agent.body.position
                r = np.sqrt(b2DistanceSquared(other_agent.body.position, agent.body.position))
                t = np.arctan2(rel_position[1], rel_position[0]) - agent.body.angle
                t = t - np.sign(t) * 2 * np.pi if np.abs(t) > np.pi else t
                obs[agent.id]["nodes"].append({"type": 0,
                                               "id": other_agent.id,
                                               "position": np.array([r, t])})
            rel_position = self.target - agent.body.position
            r = np.sqrt(b2DistanceSquared(self.target, agent.body.position))
            t = np.arctan2(rel_position[1], rel_position[0]) - agent.body.angle
            t = t - np.sign(t) * 2 * np.pi if np.abs(t) > np.pi else t
            obs[agent.id]["nodes"].append({"type": 1,
                                           "id": str(len(self.agents)),
                                           "position": np.array([r, t])})

        return obs

    def reset(self):
        self.done = False
        for agent in self.agents:
            agent.reset()
            x = self.start_spread*(random.random()-0.5) + self.start_point[0]
            y = self.start_spread*(random.random()-0.5) + self.start_point[1]
            angle = random.uniform(-1, 1) * np.pi
            agent.body.position = (x, y)
            agent.body.angle = angle
        self.create_space()


    def ShapeDestroyed(self, shape):
        """
        Callback indicating 'shape' has been destroyed.
        """
        pass

    def JointDestroyed(self, joint):
        """
        The joint passed in was removed.
        """
        pass

    def BeginContact(self, agent1, agent2):
        pass

    def CheckKeys(self):
        pass

    def MouseDown(self, p):
        """
        Mouse moved to point p, in world coordinates.
        """
        self.target = p

        self.framework.gui_objects["target"] = {'shape':'circle',
                                                'values': [p,1,b2Color(1,1,1)]}

    def quit(self):
        self.framework.contactListener = None
        self.framework.destructionListener = None
        self.framework.renderer = None
        return


if __name__ == "__main__":

    import time
    import gym_macm.envs.bots as bots
    n_agents = 10
    actors = [bots.flock] * n_agents
    render = True
    s = time.time()
    world = Flock(render = render, n_agents = n_agents, actors=actors)

    if render:
        world.framework.run()

    else:
        while not world.done:
            # actions = world.action_space.sample()
            world.step()



    print(world.time_passed)
    print(time.time() - s)