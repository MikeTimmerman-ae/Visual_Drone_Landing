from gymnasium.envs.registration import register

register(
    id="environment/FlightArena",
    entry_point="environment.envs:FlightEnv",
)
