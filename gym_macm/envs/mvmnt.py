import gym
from gym import spaces
from gym_macm.settings import flockSettings
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef,
                   b2PolygonShape, b2DistanceSquared, b2Color, b2Vec2)
from gym_macm.backends.no_render import NoRender
try:
    from gym_macm.backends.pyglet_framework import PygletFramework
except:
    pass

import random
import numpy as np

class Agent:
    def __init__(self, settings, ID, actor=None):
        self.id = ID
        self.actor = actor
        self.rotation_speed = settings.agent_rotation_speed
        self.force = settings.agent_force
        self.color = b2Color(0.4, 0.4, 0.6)

class Flock(gym.Env):
    name = "Flock v0"
    description = ("Flock")
    """
    Agents moving to a point as a group. If an agent crashes with 
    another object, it gets negative reward
    """

    def __init__(self, n_agents = 10, actors = None, colors = None,
                **kwargs):
        # super(World, self).__init__()
        self.settings = flockSettings(**kwargs)
        self.framework = PygletFramework(self.settings) if self.settings.render else NoRender(self.settings)
        self.framework.env = self
        # Framework.__init__(self)
        self.done = False
        self.n_agents = n_agents
        self.time_passed = 0 # time passed in the env

        self.target = b2Vec2((random.random()-0.5)*100, (random.random()-0.5)*100)
        if self.settings.render:
            self.framework.gui_objects["target"] = {'shape': 'circle',
                                                    'values': [self.target, self.settings.reward_tol, b2Color(1, 1, 1)]}

        self.agents = []
        for i in range(self.n_agents[0]):
            x = self.settings.start_spread * (random.random() - 0.5) + self.settings.start_point[0]
            y = self.settings.start_spread * (random.random() - 0.5) + self.settings.start_point[1]
            angle = random.uniform(-1,1) * np.pi
            agent = Agent(self.settings, ID = i)
            if actors:
                agent.actor = actors[i]
            if colors:
                agent.color = colors[i]
            agent.body = self.framework.world.CreateDynamicBody(
                **self.settings.bodySettings,
                position=(x, y),
                angle=angle,
                userData=agent
            )
            self.agents.append(agent)
        self.create_space()
        self.create_space_flag = False

        self.obs = self.get_obs()

    def step(self, actions=None):

        if self.done: # "Episode is finished."
            # print("Episode is finished.")
            self.quit() # TODO: is quit() necessary?

        if actions==None:
            # If actions are not provided, collect actions from
            # actor of each agent
            actions = {}
            for agent in self.agents:
                act = agent.actor({agent.id:self.obs[agent.id]})
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
                      np.cos(angle + np.pi / 2) * (actions[agent.id][1] - 1)) * c * agent.force
            y_force = (np.sin(angle)            * (actions[agent.id][0]-1) +
                      np.sin(angle + np.pi / 2) * (actions[agent.id][1] - 1)) * c * agent.force

            agent.body.ApplyForce(force=(x_force, y_force), point=agent.body.position, wake=True)


        # super(World, self).Step(self.settings)
        self.framework.Step(self.settings)
        
        rewards = self.get_rewards()
        self.time_passed += (1/self.settings.hz)
        if  (self.time_passed > self.settings.time_limit):
            self.done = True

        self.obs = self.get_obs()

        return self.obs, rewards # s(t+1), r(t)

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
            if self.settings.render:
                contact.fixtureA.body.userData.color = b2Color(1, 0.2, 0.2)
                contact.fixtureB.body.userData.color = b2Color(1, 0.2, 0.2)
        for agent in self.agents:
            if agent.id not in rewards:
                d = np.sqrt(b2DistanceSquared(self.target, agent.body.position))
                if self.settings.reward_mode=="linear":
                    rewards[agent.id] = (-d/35) + 1
                    if self.settings.render:
                        agent.color = self.color = b2Color(0.5, 0.5, 0.8)
                if self.settings.reward_mode=="binary":
                    r = int(d < self.settings.reward_tol)
                    rewards[agent.id] = r
                    if self.settings.render:
                        agent.color = self.color = b2Color(0.2, 0.2+0.8*r, 1-0.8*r)
        return rewards

    def get_obs(self):
        obs = {}
        for agent in self.agents:
            obs[agent.id] = {"nodes": []}
            closest_dist = np.inf
            closest_agent = None
            for other_agent in self.agents:
                if agent == other_agent:
                    continue
                rel_position = other_agent.body.position - agent.body.position
                r = np.sqrt(b2DistanceSquared(other_agent.body.position, agent.body.position))
                if r < closest_dist:
                    closest_agent = other_agent
                    closest_dist = r
            rel_position = closest_agent.body.position - agent.body.position
            t = np.arctan2(rel_position[1], rel_position[0]) - agent.body.angle
            t = t - np.sign(t) * 2 * np.pi if np.abs(t) > np.pi else t
            if self.settings.coord=="polar":
                position = np.array([closest_dist,t])
            if self.settings.coord=="cartesian":
                position = np.array([r,np.cos(t),np.sin(t)])
            obs[agent.id]["nodes"].append({"type": 0,
                                           "id": closest_agent.id,
                                           "position": position})

            rel_position = self.target - agent.body.position
            r = np.sqrt(b2DistanceSquared(self.target, agent.body.position))
            t = np.arctan2(rel_position[1], rel_position[0]) - agent.body.angle
            t = t - np.sign(t) * 2 * np.pi if np.abs(t) > np.pi else t
            if self.settings.coord == "polar":
                position = np.array([r,t])
            if self.settings.coord == "cartesian":
                position = np.array([r,np.cos(t),np.sin(t)])

            obs[agent.id]["nodes"].append({"type": 1,
                                           "id": len(self.agents),
                                           "position": position})

        return obs

    def reset(self):
        self.done = False
        for agent in self.agents:
            agent.reset()
            x = self.settings.start_spread*(random.random()-0.5) + self.settings.start_point[0]
            y = self.settings.start_spread*(random.random()-0.5) + self.settings.start_point[1]
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
                                                'values': [p,self.settings.reward_tol,b2Color(1,1,1)]}

    def quit(self):
        self.framework.quit()
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