from gym.envs.registration import register

register(
    id='hNs-v1',
    entry_point='gym_hNs.envs:TDM',
)

register(
    id='hNs-v2',
    entry_point='gym_hNs.envs:ControlledTDM',
)
