from gym.envs.registration import register

register(
    id='cm-tdm',
    entry_point='gym_macm.envs:TDM',
)

register(
    id='cm-ctdm',
    entry_point='gym_macm.envs:ControlledTDM',
)

register(
    id='cm-flock',
    entry_point='gym_macm.envs:Flock',
)
