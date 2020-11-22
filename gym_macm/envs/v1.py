
import gym
from gym import spaces
from gym_macm.cm_framework import Keys
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef,
                   b2PolygonShape, b2DistanceSquared, b2Color)

# from Box2D import b2Body
from gym_macm.settings import fwSettings

import random
import numpy as np

class Agent:
    def __init__(self, ID, team=0, actor=None):
        self.init_health = 1
        self.team = team
        self.id = ID
        self.actor = actor

        self.rotation_speed = 0.8 * (2 * np.pi)
        self._force = 20 # walk force
        self.melee_range = 2
        self.melee_dmg = 0.25
        self.percent_mov_penalty = 0.2

        self.health = self.init_health
        self.alive = True
        self.cooldown_atk = 0
        self.cooldown_mov_penalty = 0

    def reset(self):
        self.health = self.init_health
        self.alive = True
        self.cooldown_atk = 0
        self.cooldown_mov_penalty = 0

    @property
    def color(self):
        if self.team == 0:
            return b2Color(0.2, 0.2, 1)
        if self.team == 1:
            return b2Color(1, 0.2, 0.2)
        if self.team == 2:
            return b2Color(0.2, 1, 0.2)

    @property
    def force(self):
        return self._force * (1- self.percent_mov_penalty *
                              int(self.cooldown_mov_penalty > 0))

    def reset(self):
        self.health = self.init_health
        self.cooldown_attack = 0
        self.cooldown_mov_penalty = 0

class TDM(gym.Env):
    name = "Team Deathmatch"
    description = ("TDM on an empty world")

    def __init__(self, framework, n_agents = [1,1], actors = None, time_limit = 60):
        # super(World, self).__init__()
        self.settings = fwSettings
        self.framework = framework(fwSettings)
        self.framework.env = self
        # Framework.__init__(self)
        self.done = False
        self.n_agents = n_agents
        self.world_width = 30
        self.world_height = 30
        self.time_limit = time_limit # in seconds
        self.time_passed = 0 # time passed in the env

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
        for i in range(len(self.n_agents)):
            for j in range(self.n_agents[i]):
                x = random.random() * (i + self.world_width / 2)
                y =  random.random() * self.world_height
                angle = random.uniform(-1,1) * np.pi
                agent = Agent(team=i, ID = str(i) + str(j), actor = actors[i][j])
                agent.body = self.framework.world.CreateDynamicBody(
                    fixtures=circle,
                    position=(x, y),
                    angle=angle,
                    userData=agent,
                    linearDamping=5,
                    fixedRotation=True
                )
                self.agents.append(agent)
        self.n_alive = self.n_agents.copy()
        self.create_space()
        self.create_space_flag = False
        self.obs = self.get_obs()

    def step(self, actions=None):
        # print(settings)
        if self.done: # "Episode is finished."
            print("Episode is finished.")
            self.quit() # TODO: is quit() necessary?


        if actions==None:
            actions = {}
            for agent in self.agents:
                if not agent.alive:
                    continue
                act = agent.actor(self.obs[agent.id])
                actions[agent.id] = act

        assert self.action_space.contains(actions)

        # Agent actions
        for agent in self.agents:
            if not agent.alive:
                continue
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

            # Attack
            if agent.cooldown_atk <= 0:
                if actions[agent.id][3]:
                    point1 = agent.body.position
                    d = (agent.melee_range * np.cos(agent.body.angle), agent.melee_range * np.sin(agent.body.angle))
                    point2 = point1 + d
                    self.framework.world.RayCast(self.framework.raycastListener, point1, point2)
                    if self.framework.renderer:
                        self.framework.renderer.DrawSegment(point1, point2, b2Color(0.8, 0.8, 0.8))
                    agent.cooldown_atk = self.cooldown_atk
                    agent.cooldown_mov_penalty = self.cooldown_mov_penalty
                    if self.framework.raycastListener.hit:
                        self.framework.raycastListener.fixture.body.userData.health -= agent.melee_dmg
            else:
                agent.cooldown_atk -= (1/self.settings.hz)
        # Agent deaths
        for agent in self.agents:
            if not agent.alive:
                continue
            if agent.health <= 0:
                agent.alive = False
                agent.body.active = False
                self.n_alive[agent.team] -= 1
                # We recreate action and obs space when n of agents change
                self.create_space()

        # super(World, self).Step(self.settings)
        self.framework.Step(self.settings)
        self.obs = self.get_obs()
        rewards, done = None, None
        self.time_passed += (1/self.settings.hz)
        alive_teams = [i for i, e in enumerate(self.n_alive) if e != 0]
        if  (self.time_passed > self.time_limit):
            self.done = True
            print("Time is up")
        if len(alive_teams) < 2:
            self.done = True
            print("winner: team %s" % alive_teams[0])


        return self.obs, rewards, done

    def create_space(self):
        self.action_space = spaces.Dict({agent.id: spaces.MultiDiscrete([3, 3, 3, 2])
                                         for agent in self.agents if agent.alive})

        self.observation_space = spaces.Dict(
            {agent.id: spaces.Dict({"myHealth": spaces.Box(low=0, high=1, shape=(1,)),
                                    "myTeam": spaces.Discrete(1),
                                    "agents": spaces.Tuple([spaces.Dict({"type": spaces.Discrete(1),
                                                                         "position": spaces.Box(
                                                                             np.array([0, -np.pi, -np.pi]),
                                                                             # TODO: world size
                                                                             np.array([np.inf, np.pi, np.pi]))
                                                                         })
                                                            ] * (len(
                                        [agent for agent in self.agents if agent.alive]) - 1))})
             for agent in self.agents if agent.alive})

    def get_rewards(self):
        pass

    def get_obs(self):
        obs = {}
        for agent in self.agents:
            if not agent.alive:
                continue
            obs[agent.id] = {"myHealth": np.array([agent.health]),
                             "myTeam": agent.team,
                             "agents": []}
            for other_agent in self.agents:
                if agent == other_agent or not other_agent.alive:
                    continue
                rel_position = other_agent.body.position - agent.body.position
                r = np.sqrt(b2DistanceSquared(other_agent.body.position,agent.body.position))
                t = np.arctan2(rel_position[1], rel_position[0]) - agent.body.angle
                t = t - np.sign(t) * 2 * np.pi if np.abs(t) > np.pi else t
                p = other_agent.body.angle - agent.body.angle
                p = p - np.sign(p) * 2 * np.pi if np.abs(p) > np.pi else p
                is_ally = int(agent.team == other_agent.team)
                obs[agent.id]["agents"].append({"type": is_ally,
                                                "position": np.array([r, t, p])})
        return obs

    def reset(self):
        self.done = False
        for agent in self.agents:
            agent.reset()
            x = random.random() * (agent.team + self.world_width / 2)
            y = random.random() * self.world_height
            angle = random.uniform(-1, 1) * np.pi
            agent.body.position = (x, y)
            agent.body.angle = angle
        self.create_space()
        self.n_alive = self.n_agents.copy()


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

    def quit(self):
        self.framework.contactListener = None
        self.framework.destructionListener = None
        self.framework.renderer = None
        return

class ControlledTDM(TDM):

    """
    press H to type agent id you want to control in terminal
    W, A, S, D: movement, Mouse: Direction
    """

    def __init__(self, **kwargs):
        super(ControlledTDM, self).__init__(**kwargs)
        # TODO: pick by ID, not list idx
        self.idx = 0 # controlled agent idx
        # actions
        self.longitudinal = 1
        self.lateral = 1
        self.rotation = 1
        self.attack = 0

        # TODO: changing the controlled agent. Put the original actor back
        self.controlled_agent = self.agents[0]
        self.controlled_agent.actor = self.ControlledActor

    def step(self):
        super(ControlledTDM, self).step()
        self.resetActor()
        self.framework.viewCenter = self.controlled_agent.body.position

    def resetActor(self):
        self.longitudinal = 1
        self.lateral = 1
        self.attack = 0

        p = self.framework.mouseWorld
        if p==None:
            self.rotation = 1
        else:
            body = self.controlled_agent.body
            t = np.arctan2(p[1] - body.position[1], p[0] - body.position[0]) \
                - body.angle
            t = t - np.sign(t) * 2 * np.pi if np.abs(t) > np.pi else t
            self.rotation = np.sign(t) + 1

    def ControlledActor(self, obs=None):
        return np.array([self.longitudinal, self.lateral, self.rotation, self.attack])

    def CheckKeys(self):
        """
        Check the keys that are evaluated on every main loop iteration.
        I.e., they aren't just evaluated when first pressed down
        """
        # force = self.force
        body = self.controlled_agent.body
        keys = self.framework.keys
        if keys[Keys.K_w]:
            # body.linearVelocity += (2,0)
            self.longitudinal = 2
            # body.ApplyForce(force=(0, force), point=body.position, wake=True)
        if keys[Keys.K_s]:
            # body.linearVelocity +=3 (-2,0)
            # print(body.linearVelocity)
            self.longitudinal = 0
            # body.ApplyForce(force=(0,-force), point=body.position, wake=True)
        if keys[Keys.K_d]:
            # body.linearVelocity += (2,0)
            self.lateral = 0
            # body.ApplyForce(force=(force, 0), point=body.position, wake=True)
        if keys[Keys.K_a]:
            # body.linearVelocity += (-2,0)
            # print(body.position)
            self.lateral = 2
            # body.ApplyForce(force=(-force, 0), point=body.position, wake=True)

    def MouseMove(self, p):
        """
        yey
        Mouse moved to point p, in world coordinates.
        """
        # body.angle = np.arctan2(p[1] - body.position[1], p[0] - body.position[0])
        # body.angle += (2,0)
        pass

    def MouseDown(self, p):
        """
        yey
        Mouse moved to point p, in world coordinates.
        """
        self.attack = 1

    def Keyboard(self, key):
        if key == Keys.K_h:
            print("Enter agent ID")
            agentid = input()
            try:
                self.controlled_agent = self.agents[int(agentid)]
            except:
                print("Invalid agent ID")


class Flock(gym.Env):
    name = "Team Deathmatch"
    description = ("TDM on an empty world")

    def __init__(self, framework, n_agents = [1,1], actors = None, time_limit = 60):
        # super(World, self).__init__()
        self.settings = fwSettings
        self.framework = framework(fwSettings)
        self.framework.env = self
        # Framework.__init__(self)
        self.done = False
        self.n_agents = n_agents
        self.world_width = 30
        self.world_height = 30
        self.time_limit = time_limit # in seconds
        self.time_passed = 0 # time passed in the env

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
        for i in range(len(self.n_agents)):
            for j in range(self.n_agents[i]):
                x = random.random() * (i + self.world_width / 2)
                y =  random.random() * self.world_height
                angle = random.uniform(-1,1) * np.pi
                agent = Agent(team=i, ID = str(i) + str(j), actor = actors[i][j])
                agent.body = self.framework.world.CreateDynamicBody(
                    fixtures=circle,
                    position=(x, y),
                    angle=angle,
                    userData=agent,
                    linearDamping=5,
                    fixedRotation=True
                )
                self.agents.append(agent)
        self.n_alive = self.n_agents.copy()
        self.create_space()
        self.create_space_flag = False
        self.obs = self.get_obs()

    def step(self, actions=None):
        # print(settings)
        if self.done: # "Episode is finished."
            print("Episode is finished.")
            self.quit() # TODO: is quit() necessary?


        if actions==None:
            actions = {}
            for agent in self.agents:
                if not agent.alive:
                    continue
                act = agent.actor(self.obs[agent.id])
                actions[agent.id] = act

        assert self.action_space.contains(actions)

        # Agent actions
        for agent in self.agents:
            if not agent.alive:
                continue
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

            # Attack
            if agent.cooldown_atk <= 0:
                if actions[agent.id][3]:
                    point1 = agent.body.position
                    d = (agent.melee_range * np.cos(agent.body.angle), agent.melee_range * np.sin(agent.body.angle))
                    point2 = point1 + d
                    self.framework.world.RayCast(self.framework.raycastListener, point1, point2)
                    if self.framework.renderer:
                        self.framework.renderer.DrawSegment(point1, point2, b2Color(0.8, 0.8, 0.8))
                    agent.cooldown_atk = self.cooldown_atk
                    agent.cooldown_mov_penalty = self.cooldown_mov_penalty
                    if self.framework.raycastListener.hit:
                        self.framework.raycastListener.fixture.body.userData.health -= agent.melee_dmg
            else:
                agent.cooldown_atk -= (1/self.settings.hz)
        # Agent deaths
        for agent in self.agents:
            if not agent.alive:
                continue
            if agent.health <= 0:
                agent.alive = False
                agent.body.active = False
                self.n_alive[agent.team] -= 1
                # We recreate action and obs space when n of agents change
                self.create_space()

        # super(World, self).Step(self.settings)
        self.framework.Step(self.settings)
        self.obs = self.get_obs()
        rewards, done = None, None
        self.time_passed += (1/self.settings.hz)
        alive_teams = [i for i, e in enumerate(self.n_alive) if e != 0]
        if  (self.time_passed > self.time_limit):
            self.done = True
            print("Time is up")
        if len(alive_teams) < 2:
            self.done = True
            print("winner: team %s" % alive_teams[0])


        return self.obs, rewards, done

    def create_space(self):
        self.action_space = spaces.Dict({agent.id: spaces.MultiDiscrete([3, 3, 3, 2])
                                         for agent in self.agents if agent.alive})

        self.observation_space = spaces.Dict(
            {agent.id: spaces.Dict({"myHealth": spaces.Box(low=0, high=1, shape=(1,)),
                                    "agents": spaces.Tuple([spaces.Dict({"type": spaces.Discrete(1),
                                                                         "position": spaces.Box(
                                                                             np.array([0, -np.pi, -np.pi]),
                                                                             # TODO: world size
                                                                             np.array([np.inf, np.pi, np.pi]))
                                                                         })
                                                            ] * (len(
                                        [agent for agent in self.agents if agent.alive]) - 1))})
             for agent in self.agents if agent.alive})

    def get_rewards(self):
        pass

    def get_obs(self):
        obs = {}
        for agent in self.agents:
            if not agent.alive:
                continue
            obs[agent.id] = {"myHealth": np.array([agent.health]),
                             "agents": []}
            for other_agent in self.agents:
                if agent == other_agent or not other_agent.alive:
                    continue
                rel_position = other_agent.body.position - agent.body.position
                r = np.sqrt(b2DistanceSquared(other_agent.body.position,agent.body.position))
                t = np.arctan2(rel_position[1], rel_position[0]) - agent.body.angle
                t = t - np.sign(t) * 2 * np.pi if np.abs(t) > np.pi else t
                p = other_agent.body.angle - agent.body.angle
                p = p - np.sign(p) * 2 * np.pi if np.abs(p) > np.pi else p
                is_ally = int(agent.team == other_agent.team)
                obs[agent.id]["agents"].append({"type": is_ally,
                                                "position": np.array([r, t, p])})
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

    def quit(self):
        self.framework.contactListener = None
        self.framework.destructionListener = None
        self.framework.renderer = None
        return



if __name__ == "__main__":
    # Problem: Cannot run in real time while reading states in a loop
    # Possibilites:
    # 1. Record the non-rendered loop. Run a recording:
    # initial state and actions will be read from a file.
    # Disadv1: physics wil be simulated twice.
    # 2. Provide the actors to the world, so they will be a part of the system.
    #  2^ makes the controlled world possible?
    #

    if fwSettings.backend=='pyglet':
        from gym_macm.backends.pyglet_framework import PygletFramework as Framework
    elif fwSettings.backend=='no_render':
        from gym_macm.backends.no_render import NoRender as Framework
    # framework = Framework(fwSettings)
    framework = Framework
    import time
    import gym_macm.envs.bots as bots
    n_agents = [10, 10, 10]
    actors = [[bots.bot0] * n_agents[1]] * len(n_agents)

    s = time.time()
    if fwSettings.backend == 'no_render':
        world = TDM(framework = framework, n_agents = n_agents, actors=actors)

        while not world.done:
            # actions = world.action_space.sample()
            world.step()
    else:
        world = ControlledTDM(framework = framework, n_agents = n_agents, actors=actors)
        world.framework.run()
    print(world.time_passed)
    print(time.time() - s)