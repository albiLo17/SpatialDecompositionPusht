import os
import time
import csv
import numpy as np
import pygame
import gymnasium as gym

# ---- tiny helpers ------------------------------------------------------------

def clamp01(x, lo=0.0, hi=512.0):
    return np.array([np.clip(x[0], lo, hi), np.clip(x[1], lo, hi)], dtype=np.float32)

def mouse_to_env_coords(mouse_xy, win_w=512, win_h=512, flip_y=False):
    """
    Convert pygame mouse (x,y) to environment coordinates.
    pygame: origin top-left (y increases downward).
    Some envs expect y-up (origin bottom-left) — set flip_y=True in that case.
    Default: no flip (flip_y=False) because this repo's env uses y-down.
    """
    x, y = mouse_xy
    if flip_y:
        y = win_h - y
    return np.array([x, y], dtype=np.float32)

class EpisodeWriter:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.out_dir, "manifest.csv")
        if not os.path.exists(self.manifest_path):
            with open(self.manifest_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["ep_id","n_steps","success","coverage_final","reward_sum","obs_type","time_start","time_end"]
                )

    def save(self, ep_id, traj, obs_type, success, coverage, reward_sum, t0, t1):
        path = os.path.join(self.out_dir, f"episode_{ep_id:05d}.npz")
        np.savez_compressed(
            path,
            # arrays
            observations=np.array(traj["observations"], dtype=object if isinstance(traj["observations"][0], dict) else None),
            actions=np.asarray(traj["actions"], dtype=np.float32),
            rewards=np.asarray(traj["rewards"], dtype=np.float32),
            dones=np.asarray(traj["dones"], dtype=np.bool_),
            infos=np.array(traj["infos"], dtype=object),
            frames=np.asarray(traj["frames"]) if len(traj["frames"]) > 0 else np.empty((0,), dtype=np.uint8),
        )
        with open(self.manifest_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [ep_id, len(traj["rewards"]), int(success), float(coverage), float(reward_sum),
                 obs_type, int(t0), int(t1)]
            )

def init_traj():
    return dict(observations=[], actions=[], rewards=[], dones=[], infos=[], frames=[])

# ---- TELEOP COLLECTOR --------------------------------------------------------

def teleop_collect(
    make_env,
    out_dir="pusht_demos",
    num_episodes=10,
    max_steps=1500,
    smooth=0.35,            # 0=no smoothing, 1=follow mouse immediately
    record_pixels=True,     # will store frames from env._render(visualize=True)
    seed=None,
):
    """
    make_env: a callable returning the PushT env (so we can reset cleanly if needed).
              Example: lambda: gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="human")
    """
    writer = EpisodeWriter(out_dir)
    env = make_env()
    assert env.render_mode == "human", "Use render_mode='human' for interactive teleop."

    ep_id = 0
    clock = pygame.time.Clock()
    running = True

    while running and ep_id < num_episodes:
        obs, info = env.reset(seed=seed)
        obs_type = "dict" if isinstance(obs, dict) else "array"
        traj = init_traj()
        reward_sum = 0.0
        t0 = time.time()

        # current action starts at the agent position (or center if not available yet)
        try:
            agent_pos = np.array(info.get("pos_agent", [256.0, 256.0]), dtype=np.float32)
        except Exception:
            agent_pos = np.array([256.0, 256.0], dtype=np.float32)
        action = agent_pos.copy()

        mouse_down = False
        step_i = 0

        # First render to open the window
        env.render()

        while running and step_i < max_steps:
            # ---- input handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r:  # manual reset
                        step_i = max_steps  # force episode end
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_down = True
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    mouse_down = False

            # ---- build action from mouse
            if mouse_down:
                target = mouse_to_env_coords(pygame.mouse.get_pos(), 512, 512, flip_y=False)
                target = clamp01(target, 0.0, 512.0)
                action = (1.0 - smooth) * action + smooth * target  # low-pass to make pushing stable

            # ---- env step
            next_obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
            reward_sum += float(reward)

            # draw with red marker on last action in visualize mode, and also update the window
            frame_vis = None
            if record_pixels:
                be = env.unwrapped  # unwrap TimeLimit/OrderEnforcing/etc.
                screen = be._draw()  # pygame Surface of the current scene
                frame_vis = be._get_img(
                    screen,
                    width=be.visualization_width,
                    height=be.visualization_height,
                    render_action=True,
                )
            env.render()

            # ---- record
            traj["observations"].append(obs)
            traj["actions"].append(action.copy())
            traj["rewards"].append(reward)
            traj["dones"].append(terminated or truncated)
            traj["infos"].append(info)
            if record_pixels and frame_vis is not None:
                traj["frames"].append(frame_vis)

            obs = next_obs
            step_i += 1
            clock.tick(env.metadata.get("render_fps", 10))  # ~10 Hz control (env internally integrates physics)

            if terminated or truncated:
                break

        # ---- end episode + save
        success = bool(info.get("is_success", False))
        coverage = float(info.get("coverage", 0.0))
        t1 = time.time()
        writer.save(ep_id, traj, obs_type, success, coverage, reward_sum, t0, t1)
        print(f"[teleop] saved episode {ep_id} | steps={len(traj['rewards'])} | success={success} | coverage={coverage:.3f}")
        ep_id += 1

    env.close()
    print(f"[teleop] finished. wrote {ep_id} episodes to: {os.path.abspath(out_dir)}")


# ---- example usage -----------------------------------------------------------
if __name__ == "__main__":
    import gym_pusht  # your package providing PushTEnv

    def make_env():
        # Pick your preferred observation; pixels_agent_pos is nice for IL
        return gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",   # or "state", "pixels", "environment_state_agent_pos"
            render_mode="human",
        )

    teleop_collect(
        make_env=make_env,
        out_dir="pusht_demos_npz",
        num_episodes=25,
        max_steps=2000,
        smooth=0.35,
        record_pixels=True,
        seed=42,
    )
