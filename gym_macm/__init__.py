from gym.envs.registration import register

register(
    id='cm-v1',
    entry_point='gym_macm.envs:TDM',
)

register(
    id='cm-v2',
    entry_point='gym_macm.envs:ControlledTDM',
)
