from gym.envs.registration import register

register(
    id='selfdrive-v0',
    entry_point='gym_self_drive.envs:SelfDriveEnv',
)
