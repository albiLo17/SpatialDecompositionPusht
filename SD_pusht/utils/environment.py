"""Environment utilities for PushT."""

try:
    from gym_pusht.envs import PushTEnv
    GYM_PUSHT_AVAILABLE = True
except ImportError:
    GYM_PUSHT_AVAILABLE = False
    PushTEnv = None


def make_env(render_mode="rgb_array", legacy=True, seed=None):
    """Create a factory function for PushT environments.
    
    Args:
        render_mode: Rendering mode for the environment.
        legacy: Whether to use legacy mode.
        seed: Random seed for the environment.
        
    Returns:
        A function that creates and returns a PushT environment.
    """
    if not GYM_PUSHT_AVAILABLE:
        raise ImportError("gym_pusht is not installed. Install it to use PushT environments.")
    
    def _thunk():
        env = PushTEnv(render_mode=render_mode, legacy=legacy)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _thunk


def apply_legacy_state(env, state):
    """Apply a legacy state to the PushT environment.
    
    Args:
        env: PushTEnv instance.
        state: State array [agent_x, agent_y, block_x, block_y, block_angle].
    """
    if not GYM_PUSHT_AVAILABLE:
        raise ImportError("gym_pusht is not installed. Install it to use PushT environments.")
    
    # state: [agent_x, agent_y, block_x, block_y, block_angle]
    agent_pos = list(state[:2])
    block_pos = list(state[2:4])
    block_angle = float(state[4])

    # ensure env has been setup (creates space, bodies)
    # if you already called env.reset(), you can skip env._setup()
    # but do not call env._set_state(...) because it uses the other ordering
    env.agent.position = agent_pos
    # legacy ordering: position first, then angle
    env.block.position = block_pos
    env.block.angle = block_angle

    # step one physics tick so internal caches update
    # use env.dt or env.sim_hz-derived dt
    dt = getattr(env, "dt", 1.0 / getattr(env, "sim_hz", 100))
    env.space.step(dt)

