# ref: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#creating-environment-instances
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5,
                 reward_0_step=1, reward_0_terminal=100,
                 reward_1_step=-1, reward_1_terminal=100,
                 reward_mode="default",
                 loop_detection=False, loop_window=10, loop_grace_period=5,
                 max_steps=100):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.reward_0_step = reward_0_step
        self.reward_0_terminal = reward_0_terminal
        self.reward_1_step = reward_1_step
        self.reward_1_terminal = reward_1_terminal
        self.loop_detection = loop_detection
        self.loop_window = loop_window
        self.loop_grace_period = loop_grace_period  # how many loop detections to tolerate before switching reward
        self._position_history = []
        self.max_steps = max_steps
        self._current_step = 0

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "phase": spaces.Discrete(2),
                "timestep": spaces.Box(0, 1, shape=(1,), dtype=np.float64),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.reward_mode = reward_mode
        self.reward_function = self.reward_func_0
        self.reward_phase = 0

    def reward_func_0(self):
        if self._is_at_terminal_state():
            return self.reward_0_terminal
        else:
            return self.reward_0_step

    def reward_func_1(self):
        if self._is_at_terminal_state():
            return self.reward_1_terminal
        else:
            return self.reward_1_step

    def reward_func_relative_position(self):
        """Calculate reward based on relative position to the target."""
        distance = np.linalg.norm(self._agent_location - self._target_location, ord=1)
        return -distance  # Negative reward proportional to Manhattan distance

    def _detect_loop(self):
        """Detect if the agent is cycling through a repeating sequence of states.
        Returns (is_loop, cycle_count) where cycle_count is the number of full
        repetitions of the detected cycle period within the history window.
        e.g. ababababab → period=2, cycle_count=5; abcbabcba → period=4, cycle_count=2.
        """
        h = self._position_history
        n = len(h)
        for period in range(2, n // 2 + 1):
            if h[-period:] == h[-2 * period:-period]:
                return True, n // period
        return False, 0

    def _is_at_terminal_state(self):
        return np.array_equal(self._agent_location, self._target_location)

    def action_masks(self):
        """Return valid actions based on agent position (no wall bumping)."""
        x, y = self._agent_location
        return np.array([
            x < self.size - 1,  # right
            y < self.size - 1,  # up
            x > 0,              # left
            y > 0,              # down
        ], dtype=np.int8)


    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "target": self._target_location,
            "phase": self.reward_phase,
            "timestep": np.array([self._current_step / self.max_steps], dtype=np.float64),
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None, reward_func=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self._position_history = []

        if reward_func:
            self.reward_function = reward_func
            self.reward_phase = 1
        elif self.reward_mode == "relative":
            self.reward_function = self.reward_func_relative_position
        else:
            self.reward_function = self.reward_func_0
            self.reward_phase = 0
            self._current_step = 0

        return observation, info

    def step(self, action):
        self._current_step += 1
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = False  # or self._is_at_terminal_state()
        truncated = False  # the wrapper sets time limit via max_episode_steps

        # Track position history for loop detection
        if self.loop_detection:
            self._position_history.append(tuple(self._agent_location))
            if len(self._position_history) > self.loop_window:
                self._position_history.pop(0)

        reward = self.reward_function()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        if self._is_at_terminal_state():
            if self.reward_function == self.reward_func_0:
                self.reset(reward_func=self.reward_func_1)
                observation = self._get_obs()
                info = self._get_info()
            else:
                terminated = True
        elif self.loop_detection:
            loop_found, cycle_count = self._detect_loop()
            if loop_found:
                info["loop_detected"] = True
                info["loop_count"] = cycle_count
                if cycle_count > self.loop_grace_period:
                    if self.reward_function == self.reward_func_0:
                        self.reset(reward_func=self.reward_func_1)
                    else:
                        terminated = True

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
