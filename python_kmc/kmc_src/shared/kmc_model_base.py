"""Base KMC Model abstract class.

This module provides the fundamental framework for all KMC models,
regardless of dimension. Subclasses implement dimension-specific
details (lattice structure, process handler, rate calculator).
"""

from abc import ABC

import numpy as np
from tqdm import tqdm


class BaseKMCModel(ABC):
    """Abstract base class for all KMC models, regardless of dimension."""

    def __init__(self, seed: int = 0):
        self.time_passed_seconds = 0
        self.loop_counter = 0
        self.total_transition_rate_per_second = np.array([0], dtype=np.float32)
        self.seed = seed

        # These will be instantiated by the child classes
        self.lattice = None
        self.process_handler = None
        self.rate_calculator = None

    def set_seed(self, seed: int):
        self.seed = seed
        self.process_handler.set_seed(seed)
        self.rate_calculator.set_seed(seed)

    def update_temperature(self, temperature_kelvin: float):
        """Update the temperature and recalculate all transition rates.

        Args:
            temperature_kelvin (float): New temperature in Kelvin.
        """
        self.rate_calculator.temperature_kelvin = np.float32(temperature_kelvin)
        self.rate_calculator.update_all()

    def calculate_delta_t(self) -> float:
        random_number = np.random.uniform()
        total_rate = self.total_transition_rate_per_second[0]
        if total_rate == 0:
            return float("inf")
        return -np.log(random_number) / total_rate

    def one_step(self, dt: float):
        self.time_passed_seconds += dt
        site, process = self.process_handler.select_process()
        self.process_handler.change_config(site, process)
        self.rate_calculator.update_transition_rate_matrix(
            indices=site, process=process
        )
        self.loop_counter += 1

    def run_simulation_t_end(
        self,
        t_end_seconds: float,
        show_progress: bool = False,
    ):
        """Run KMC simulation until the step closest to t_end is reached.

        Args:
            t_end_seconds (float): Target simulation time in seconds.
            show_progress (bool): Flag to show a progress bar during the simulation.
        """

        if show_progress:
            pbar = tqdm(total=t_end_seconds, desc="KMC Simulation", dynamic_ncols=True)

        while True:
            # 1. Peek at the next time step
            dt = self.calculate_delta_t()
            t_curr = self.time_passed_seconds
            t_next = t_curr + dt

            # 2. Check proximity: Is the NEXT step closer to t_end than the CURRENT one?
            # Math: |t_end - t_next| < |t_end - t_curr|
            if abs(t_end_seconds - t_next) <= abs(t_end_seconds - t_curr):
                self.one_step(dt)

                if show_progress:
                    pbar.n = float(min(self.time_passed_seconds, t_end_seconds))
                    pbar.refresh()
            elif abs(t_end_seconds - t_curr) == abs(t_end_seconds - t_next):
                print("dt: ", dt)
                print(
                    "Transition rate per second: ",
                    self.total_transition_rate_per_second[0],
                )
                raise ValueError(
                    "The current and next steps are equally close to t_end. Simulation cannot proceed."
                )
            else:
                # The current state is already the closest we can get to t_end
                pbar.close() if show_progress else None
                break

        if show_progress:
            pbar.close()

    def run_simulation_dt(
        self,
        dt: float,
        show_progress: bool = False,
    ):
        """Run a KMC simulation for a certain time interval.

        Args:
            dt (float): Simulation time duration in seconds.
            show_progress (bool): Flag to show a progress bar during the simulation.
        """
        t_end_seconds = self.time_passed_seconds + dt  # calculate end time
        self.run_simulation_t_end(
            t_end_seconds=t_end_seconds,
            show_progress=show_progress,
        )

    def run_simulation_steps(self, steps: int, show_progress: bool = False):
        """Run a KMC simulation for a certain number of steps.

        Args:
            steps (int): Number of KMC steps to perform.
            show_progress (bool): Flag to show a progress bar during the simulation.
        """
        if show_progress:  # if a progress bar should be shown
            pbar = tqdm(total=steps, desc="KMC Simulation", dynamic_ncols=True)
            for _ in range(steps):
                dt = self.calculate_delta_t()
                self.one_step(dt)  # single KMC-Step

                # Update progress bar by one step
                pbar.n += 1
                pbar.refresh()
            pbar.close()
        else:  # without progress bar
            for _ in range(steps):
                dt = self.calculate_delta_t()
                self.one_step(dt)  # single KMC-Step
