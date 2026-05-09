"""Replay a yoked action sequence in the CornerMazeEnv with pygame rendering.

Reads a yoked parquet file from data/yoked/, auto-configures the environment
from stored metadata (session type, cue/goal orientation, trial configs),
and steps through the action sequence with play/pause/step controls.

Controls:
    - Click play/pause button or press SPACE to toggle playback
    - Click step buttons or press LEFT/RIGHT arrows to step when paused
    - Press Q or close window to quit

Usage:
    python yoking/replay_session.py data/yoked/CM016_5.parquet --delay 33
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from corner_maze_rl.env.corner_maze_env import CornerMazeEnv
from corner_maze_rl.env.constants import STATE_TRIAL, STATE_ITI, STATE_PRETRIAL

_GOAL_IDX_TO_NAME = {0: 'NE', 1: 'SE', 2: 'SW', 3: 'NW'}

# ── Colors ────────────────────────────────────
_BG = (255, 255, 255)
_BTN_ACTIVE = (70, 130, 180)
_BTN_HOVER = (100, 160, 210)
_BTN_DISABLED = (180, 180, 180)
_BTN_TEXT = (255, 255, 255)
_BTN_TEXT_DIS = (220, 220, 220)
_INFO_FG = (0, 0, 0)


def _inject_trial_configs(env, trial_configs):
    """Build full pretrial/trial/ITI sequence and configure for replay."""
    grid_configuration_sequence = []
    num_trials = len(trial_configs)

    for i, cfg in enumerate(trial_configs):
        start_arm, cue, goal = cfg[0], cfg[1], cfg[2]
        grid_configuration_sequence.append(env.maze_config_pre_list[start_arm][cue])
        grid_configuration_sequence.append(env.maze_config_trl_list[start_arm][cue][goal])
        if i < num_trials - 1:
            next_start_arm = trial_configs[i + 1][0]
            grid_configuration_sequence.append(env.maze_config_iti_list[next_start_arm])

    sequence_labels = []
    for entry in grid_configuration_sequence:
        if isinstance(entry, tuple):
            sequence_labels.append(env.layout_name_lookup.get(entry, ''))
        elif isinstance(entry, list):
            sequence_labels.append([env.layout_name_lookup.get(sub, '') for sub in entry])

    trial_tags = [cfg[3] for cfg in trial_configs]
    env.grid_configuration_sequence = grid_configuration_sequence
    env.grid_configuration_len = len(grid_configuration_sequence)
    env.session_num_trials = num_trials
    env.trial_tags = trial_tags
    env.trial_configs = [list(cfg) for cfg in trial_configs]
    env.sequence_labels = sequence_labels
    env.sequence_count = 0

    env.update_grid_configuration(grid_configuration_sequence[0])
    env.session_phase = STATE_PRETRIAL
    env.pretrial_step_count = 0

    env.agent_pos, env.agent_dir = env.gen_start_pose()
    env.agent_pose = (*env.agent_pos, env.agent_dir)
    env.agent_start_pos = env.agent_pos
    env.fwd_pos = env.front_pos
    env.fwd_cell = env.grid.get(*env.fwd_pos)
    env.cur_cell = type(env.grid.get(*env.agent_pos)).__name__


def load_yoked_parquet(path):
    """Load a yoked parquet file and return (actions_df, metadata_dict)."""
    pf = pq.read_table(path)
    meta_raw = pf.schema.metadata or {}
    meta = {k.decode(): v.decode() for k, v in meta_raw.items()
            if k != b'pandas' and k != b'ARROW:schema'}
    actions_df = pf.to_pandas()
    return actions_df, meta


class ReplayController:
    """Manages env state and supports forward/backward stepping."""

    def __init__(self, env, actions, trial_configs, is_exposure, init_dir=None, init_pos=None):
        self.env = env
        self.actions = actions
        self.trial_configs = trial_configs
        self.is_exposure = is_exposure
        self.init_dir = init_dir
        self.init_pos = init_pos
        self.current_step = -1  # before first action
        self.total_steps = len(actions)
        self.terminated = False

    def _reset_env(self):
        """Reset env to initial state."""
        self.env.reset()
        if self.trial_configs:
            _inject_trial_configs(self.env, self.trial_configs)
        elif self.init_dir is not None:
            if self.init_pos:
                self.env.agent_pos = self.init_pos
            self.env.agent_dir = self.init_dir
            self.env.agent_pose = (*self.env.agent_pos, self.init_dir)
            self.env.fwd_pos = self.env.front_pos
            self.env.fwd_cell = self.env.grid.get(*self.env.fwd_pos)
        self.terminated = False

    def step_forward(self):
        """Advance one step. Returns True if action was applied."""
        if self.terminated or self.current_step + 1 >= self.total_steps:
            return False
        self.current_step += 1
        action = int(self.actions[self.current_step])
        _, _, terminated, truncated, _ = self.env.step(action)
        if terminated or truncated:
            self.terminated = True
        return True

    def step_backward(self):
        """Go back one step by replaying from scratch."""
        if self.current_step <= 0:
            return False
        target = self.current_step - 1
        self._reset_env()
        self.current_step = -1
        for _ in range(target + 1):
            self.current_step += 1
            action = int(self.actions[self.current_step])
            _, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                self.terminated = True
                break
        return True

    def get_frame(self):
        """Get the current grid frame as an RGB array."""
        return self.env.get_frame(
            self.env.highlight, self.env.tile_size, self.env.agent_pov
        )

    def get_eye_images(self):
        """Get left/right eye images if available. Returns (left, right) or None."""
        if not hasattr(self.env, '_pose_to_left_eye'):
            return None
        pose_label = self.env._get_pose_label()
        left = self.env._pose_to_left_eye.get(pose_label, self.env._zero_eye)
        right = self.env._pose_to_right_eye.get(pose_label, self.env._zero_eye)
        return left, right


def main():
    parser = argparse.ArgumentParser(
        description='Replay a yoked action sequence with play/pause controls.',
    )
    parser.add_argument('parquet', type=str, help='Path to a yoked parquet file.')
    parser.add_argument('--delay', type=int, default=50,
                        help='Delay between actions in ms during playback (default: 50).')
    args = parser.parse_args()

    # Load data
    actions_df, meta = load_yoked_parquet(args.parquet)
    subject = meta.get('subject', '?')
    session_number = meta.get('session_number', '?')
    session_phase = meta.get('session_phase', '')
    session_type = meta.get('session_type', '')
    orientation = meta.get('cue_goal_orientation', 'N/NE')

    is_exposure = session_phase == 'Exposure'
    trial_configs = None
    if 'trial_configs' in meta:
        trial_configs = json.loads(meta['trial_configs'])

    goal_location = 'NE'
    if trial_configs:
        goal_location = _GOAL_IDX_TO_NAME.get(trial_configs[0][2], 'NE')

    session_number = meta.get('session_number', '')
    n_rewards_meta = int(meta.get('n_rewards', '0'))
    if is_exposure:
        env_session_type = 'exposure_b' if (session_number == '2e' and n_rewards_meta > 0) else 'exposure'
    else:
        env_session_type = 'PI+VC f2 acquisition'
    max_steps = max(len(actions_df) * 2, 10000)

    print(f'Replaying {subject} session {session_number}')
    print(f'  Type: {session_type} ({session_phase})')
    print(f'  Orientation: {orientation}, Goal: {goal_location}')
    print(f'  Actions: {len(actions_df)}')
    if trial_configs:
        print(f'  Trials: {len(trial_configs)}')

    # Create env (rgb_array mode — we handle pygame ourselves)
    env = CornerMazeEnv(
        render_mode='rgb_array',
        max_steps=max_steps,
        session_type=env_session_type,
        agent_cue_goal_orientation=orientation,
        start_goal_location=goal_location,
    )
    env.reset()
    if trial_configs:
        _inject_trial_configs(env, trial_configs)
    else:
        # Exposure: set initial position and direction from action data
        init_pos = (int(actions_df['grid_x'].iloc[0]), int(actions_df['grid_y'].iloc[0]))
        init_dir = int(actions_df['direction'].iloc[0])
        env.agent_pos = init_pos
        env.agent_dir = init_dir
        env.agent_pose = (*init_pos, init_dir)
        env.fwd_pos = env.front_pos
        env.fwd_cell = env.grid.get(*env.fwd_pos)

    if not trial_configs:
        init_dir = int(actions_df['direction'].iloc[0])
        init_pos = (int(actions_df['grid_x'].iloc[0]), int(actions_df['grid_y'].iloc[0]))
    else:
        init_dir = None
        init_pos = None
    controller = ReplayController(
        env, actions_df['action'].values, trial_configs, is_exposure, init_dir, init_pos,
    )

    # ── Pygame setup ──────────────────────────────
    import pygame
    import pygame.freetype
    pygame.init()

    # Get initial frame to determine sizes
    frame = controller.get_frame()
    frame = np.transpose(frame, (1, 0, 2))
    grid_w, grid_h = frame.shape[0], frame.shape[1]

    # Eye images
    eyes = controller.get_eye_images()
    has_eyes = eyes is not None
    gap = int(grid_w * 0.05)
    eye_size = grid_h if has_eyes else 0

    # Layout dimensions
    content_w = grid_w + (gap + eye_size + gap + eye_size if has_eyes else 0)
    padding = int(grid_w * 0.03)
    font_size = 16
    info_h = int(font_size * 1.8) * 3  # 3 rows of info text
    btn_h = 36
    btn_w = 60
    btn_gap = 12
    btn_bar_h = btn_h + 16

    window_w = content_w + padding * 2
    window_h = grid_h + info_h + btn_bar_h + padding * 2

    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption(f'Replay: {subject} {session_number}')
    clock = pygame.time.Clock()
    font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)

    # Button positions (centered under grid)
    btn_total_w = btn_w * 3 + btn_gap * 2
    btn_x_start = padding + (content_w - btn_total_w) // 2
    btn_y = padding + grid_h + info_h + 8

    btn_back = pygame.Rect(btn_x_start, btn_y, btn_w, btn_h)
    btn_play = pygame.Rect(btn_x_start + btn_w + btn_gap, btn_y, btn_w, btn_h)
    btn_fwd = pygame.Rect(btn_x_start + 2 * (btn_w + btn_gap), btn_y, btn_w, btn_h)

    # Compute yoked phase labels from action patterns + trial configs.
    # Each trial cycle: pretrial (synthetic F,TT,F,F,TT = 7 actions)
    # → trial (until PICKUP with rewarded=1) → ITI (until next pretrial).
    yoked_phases = []
    if trial_configs:
        tc_idx = 0
        phase_label = 'PRE'
        pretrial_steps_left = 7
        for si in range(len(actions_df)):
            yoked_phases.append(phase_label)
            act_val = int(actions_df['action'].iloc[si])
            rwd_val = int(actions_df['rewarded'].iloc[si])

            if phase_label == 'PRE':
                pretrial_steps_left -= 1
                if pretrial_steps_left <= 0:
                    phase_label = 'TRIAL'
            elif phase_label == 'TRIAL':
                if act_val == 3 and rwd_val == 1:  # rewarded PICKUP
                    tc_idx += 1
                    if tc_idx < len(trial_configs):
                        phase_label = 'ITI'
                    # else: last trial, stay in TRIAL
            elif phase_label == 'ITI':
                # Detect pretrial start: F at an arm start pose
                # Simplified: when we see the pattern matching pretrial
                # location, switch to PRE. Use a step counter heuristic:
                # pretrial starts are spaced by the trial+ITI actions.
                # For now, detect the pretrial by checking if the position
                # matches an arm start pose for the next trial.
                if tc_idx < len(trial_configs):
                    next_arm = trial_configs[tc_idx][0]
                    arm_pos = {0: (6, 3), 1: (9, 6), 2: (6, 9), 3: (3, 6)}
                    pos_si = (int(actions_df['grid_x'].iloc[si]),
                              int(actions_df['grid_y'].iloc[si]))
                    if pos_si == arm_pos[next_arm]:
                        phase_label = 'PRE'
                        pretrial_steps_left = 6  # already consumed 1
    else:
        yoked_phases = ['EXP'] * len(actions_df)

    # State
    playing = False
    last_step_time = 0
    delay_ms = args.delay
    action_names = {0: 'L', 1: 'R', 2: 'F', 3: 'PICKUP', 4: 'PAUSE'}
    phase_names = {1: 'EXPA', 2: 'EXPB', 3: 'PRE', 4: 'TRIAL', 5: 'ITI'}

    def draw_button(rect, label, enabled, hovered):
        if enabled:
            color = _BTN_HOVER if hovered else _BTN_ACTIVE
            text_color = _BTN_TEXT
        else:
            color = _BTN_DISABLED
            text_color = _BTN_TEXT_DIS
        pygame.draw.rect(screen, color, rect, border_radius=6)
        text_rect = font.get_rect(label, size=font_size)
        tx = rect.centerx - text_rect.width // 2
        ty = rect.centery - text_rect.height // 2
        font.render_to(screen, (tx, ty), label, size=font_size, fgcolor=text_color)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_RIGHT and not playing:
                    controller.step_forward()
                elif event.key == pygame.K_LEFT and not playing:
                    controller.step_backward()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_play.collidepoint(mouse_pos):
                    playing = not playing
                elif btn_back.collidepoint(mouse_pos) and not playing:
                    controller.step_backward()
                elif btn_fwd.collidepoint(mouse_pos) and not playing:
                    controller.step_forward()

        # Auto-advance in play mode
        if playing and not controller.terminated:
            now = pygame.time.get_ticks()
            if now - last_step_time >= delay_ms:
                if not controller.step_forward():
                    playing = False
                last_step_time = now

        # ── Draw ──────────────────────────────────
        screen.fill(_BG)

        # Grid
        frame = controller.get_frame()
        frame = np.transpose(frame, (1, 0, 2))
        grid_surf = pygame.surfarray.make_surface(frame)
        screen.blit(grid_surf, (padding, padding))

        # Eye images
        if has_eyes:
            eyes = controller.get_eye_images()
            if eyes:
                left_eye, right_eye = eyes
                left_rgb = np.repeat(
                    (left_eye * 255).astype(np.uint8)[:, :, np.newaxis], 3, axis=2)
                right_rgb = np.repeat(
                    (right_eye * 255).astype(np.uint8)[:, :, np.newaxis], 3, axis=2)
                left_surf = pygame.surfarray.make_surface(
                    np.transpose(left_rgb, (1, 0, 2)))
                left_surf = pygame.transform.smoothscale(left_surf, (eye_size, eye_size))
                right_surf = pygame.surfarray.make_surface(
                    np.transpose(right_rgb, (1, 0, 2)))
                right_surf = pygame.transform.smoothscale(right_surf, (eye_size, eye_size))
                ex = padding + grid_w + gap
                screen.blit(left_surf, (ex, padding))
                screen.blit(right_surf, (ex + eye_size + gap, padding))

        # Info text (3 rows)
        step = controller.current_step
        info_y = padding + grid_h + 4
        line_h = int(font_size * 1.8)
        dir_names = {0: 'E', 1: 'S', 2: 'W', 3: 'N'}

        if step >= 0:
            act = int(controller.actions[step])
            phase = phase_names.get(env.session_phase, '?')
            rewarded = int(actions_df['rewarded'].iloc[step])
            rwd_tag = '  RWD' if rewarded else ''

            yk_phase = yoked_phases[step] if step < len(yoked_phases) else '?'
            line1 = (f'Step {step + 1}/{controller.total_steps}'
                     f'  act={action_names.get(act, "?")}'
                     f'  env={phase}  yoked={yk_phase}'
                     f'  tc={env.trial_count}{rwd_tag}')

            # Post-step state comparison: env vs yoked next-step
            env_x, env_y = int(env.agent_pos[0]), int(env.agent_pos[1])
            env_d = dir_names.get(int(env.agent_dir), '?')
            line2 = f'  env:   ({env_x:2d},{env_y:2d}) {env_d}'

            next_step = step + 1
            if next_step < controller.total_steps:
                yk_x = int(actions_df['grid_x'].iloc[next_step])
                yk_y = int(actions_df['grid_y'].iloc[next_step])
                yk_d = dir_names.get(int(actions_df['direction'].iloc[next_step]), '?')
                yk_act = action_names.get(int(actions_df['action'].iloc[next_step]), '?')
                line3 = f'  yoked: ({yk_x:2d},{yk_y:2d}) {yk_d}  next_act={yk_act}'
                if (env_x, env_y) != (yk_x, yk_y) or env_d != yk_d:
                    line3 += '  *** DIVERGE ***'
            else:
                line3 = '  yoked: (end of data)'
        else:
            line1 = f'Step 0/{controller.total_steps}  (press > or SPACE)'
            line2 = ''
            line3 = ''

        font.render_to(screen, (padding, info_y), line1,
                        size=font_size, fgcolor=_INFO_FG)
        font.render_to(screen, (padding, info_y + line_h), line2,
                        size=font_size, fgcolor=_INFO_FG)
        diverge_color = (200, 0, 0) if '*** DIVERGE ***' in line3 else _INFO_FG
        font.render_to(screen, (padding, info_y + line_h * 2), line3,
                        size=font_size, fgcolor=diverge_color)

        # Buttons
        back_enabled = not playing and step > 0
        fwd_enabled = not playing and step + 1 < controller.total_steps
        play_label = 'Pause' if playing else 'Play'

        draw_button(btn_back, '<', back_enabled, btn_back.collidepoint(mouse_pos))
        draw_button(btn_play, play_label, True, btn_play.collidepoint(mouse_pos))
        draw_button(btn_fwd, '>', fwd_enabled, btn_fwd.collidepoint(mouse_pos))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    env.close()
    print(f'Stopped at step {controller.current_step + 1}/{controller.total_steps}')


if __name__ == '__main__':
    main()
