from gym.envs.registration import register

register(
    id='cm-tdm-v0',
    entry_point='gym_macm.envs:TDM',
)

register(
    id='cm-ctdm-v0',
    entry_point='gym_macm.envs:ControlledTDM',
)

register(
    id='cm-flock-v0',
    entry_point='gym_macm.envs:Flock',
)
