# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
#
# This file contains code adapted from:
#
# 1. PHC_MJX (https://github.com/ZhengyiLuo/PHC_MJX)

import os
import json
import csv
import torch
import numpy as np
import logging
from omegaconf import OmegaConf

os.environ["OMP_NUM_THREADS"] = "1"

from src.agents.agent_humanoid import AgentHumanoid
from src.learning.learning_utils import to_test, to_cpu
from src.env.myolegs_im import MyoLegsIm

logger = logging.getLogger(__name__)


class AgentIM(AgentHumanoid):
    """
    AgentIM is a specialized reinforcement learning agent for humanoid environments,
    extending AgentHumanoid with specific functionalities for the MyoLegsIm environment.
    """
    
    def __init__(self, cfg, dtype, device, training: bool = True, checkpoint_epoch: int = 0):
        """
        Initialize the AgentIM with configurations and set up necessary components.

        Args:
            cfg: Configuration object containing hyperparameters and settings.
            dtype: Data type for tensors (e.g., torch.float32).
            device: Device for computations (e.g., 'cuda' or 'cpu').
            training (bool, optional): Flag indicating if the agent is in training mode.
            checkpoint_epoch (int, optional): Epoch number from which to load the checkpoint.
        """
        super().__init__(cfg, dtype, device, training, checkpoint_epoch)

    def get_full_state_weights(self) -> dict:
        """
        Extends the state dictionary with termination history for checkpointing.

        Returns:
            dict: The state dictionary including termination history.
        """
        state = super().get_full_state_weights()
        return state
    
    def set_full_state_weights(self, state) -> None:
        """
        Loads the state dictionary.

        Args:
            state (dict): The state dictionary including termination history.
        """
        super().set_full_state_weights(state)
        
    
    def pre_epoch(self) -> None:
        """
        Performs operations before each training epoch, such as resampling motions.
        """
        if (self.epoch > 1) and self.epoch % self.cfg.env.resampling_interval == 1: # + 1 to evade the evaluations. 
            self.env.sample_motions()
        return super().pre_epoch()
    
    def setup_env(self):
        """
        Initializes the MyoLegsIm environment based on the configuration.
        """
        self.env = MyoLegsIm(self.cfg)
        logger.info("MyoLegsIm environment initialized.")

    def eval_policy(self, epoch: int = 0, dump: bool = False, runs = None) -> float:
        """
        Evaluates the current policy by running multiple episodes and computing success rates.

        Args:
            epoch (int, optional): Current epoch number for logging and checkpointing.
            dump (bool, optional): Flag indicating whether to dump evaluation results.

        Returns:
            float: The success rate of the policy.
        """
        logger.info("Starting policy evaluation.")
        res_dict_acc = {}
        self.env.start_eval(im_eval = True)

        # Set networks to evaluation mode
        to_test(*self.sample_modules)

        success_dict = {}
        mpjpe_dict = {}
        frame_coverage_dict = {}

        if runs is not None:
            run_ctr = 0

        with to_cpu(*self.sample_modules), torch.no_grad():
            for run_idx in self.env.forward_motions():
                success = False
                for attempt in range(1):
                    result, mpjpe, frame_coverage = self.eval_single_thread()
                    mpjpe_float = self._metric_to_float(mpjpe)
                    frame_coverage_float = self._metric_to_float(frame_coverage)
                    if result is True:
                        success = True
                        logger.info(
                            f"Run {run_idx}: Success on attempt {attempt + 1}. "
                            f"MPJPE: {mpjpe_float * 1000:.5f}, "
                            f"Frame Coverage: {frame_coverage_float * 100:.5f}"
                        )
                    else:
                        success = False
                        logger.info(
                            f"Run {run_idx}: Failure on attempt {attempt + 1}. "
                            f"MPJPE: {mpjpe_float * 1000:.5f}, "
                            f"Frame Coverage: {frame_coverage_float * 100:.5f}"
                        )
                    success_dict[run_idx] = success
                    mpjpe_dict[run_idx] = mpjpe
                    frame_coverage_dict[run_idx] = frame_coverage
                if runs is not None:
                    run_ctr += 1
                    if run_ctr >= runs:
                        break
                
        success_rate = np.mean(list(success_dict.values())) if success_dict else 0.0
        mean_mpjpe = np.mean([self._metric_to_float(v) for v in mpjpe_dict.values()]) if mpjpe_dict else np.nan
        mean_frame_coverage = (
            np.mean([self._metric_to_float(v) for v in frame_coverage_dict.values()])
            if frame_coverage_dict else np.nan
        )
        failed_keys = [k for k, v in success_dict.items() if not v]
        success_keys = [k for k, v in success_dict.items() if v]
        print(f"Success Rate: {success_rate * 100:.5f}")
        print("Mean MPJPE: ", mean_mpjpe * 1000)
        print("Mean frame coverage: ", mean_frame_coverage * 100)
        self._save_eval_metrics(
            success_dict,
            mpjpe_dict,
            frame_coverage_dict,
            success_rate,
            mean_mpjpe,
            mean_frame_coverage,
        )

        # save failed keys
        if dump:
            os.makedirs("data/dumps", exist_ok=True)
            failed_keys = np.array(failed_keys)
            np.save(f"data/dumps/failed_keys_{self.cfg.epoch}.npy", failed_keys)

        if self.env.recording_biomechanics:
            breakpoint()
            print("Saving recorded biomechanics data.")
            

        return mpjpe_dict, success_rate

    @staticmethod
    def _metric_to_float(value) -> float:
        arr = np.asarray(value, dtype=np.float64)
        return float(np.nanmean(arr))

    def _save_eval_metrics(
        self,
        success_dict,
        mpjpe_dict,
        frame_coverage_dict,
        success_rate,
        mean_mpjpe,
        mean_frame_coverage,
    ) -> None:
        """Persist real evaluation metrics for plotting scripts."""
        motions = []
        for key in mpjpe_dict:
            motions.append(
                {
                    "motion_id": str(key),
                    "success": bool(success_dict.get(key, False)),
                    "mpjpe_m": self._metric_to_float(mpjpe_dict[key]),
                    "mpjpe_mm": self._metric_to_float(mpjpe_dict[key]) * 1000.0,
                    "frame_coverage": self._metric_to_float(frame_coverage_dict[key]),
                }
            )

        cfg_snapshot = OmegaConf.to_container(
            self.cfg,
            resolve=True,
            throw_on_missing=False,
        )
        payload = {
            "exp_name": str(self.cfg.exp_name),
            "epoch": int(self.cfg.epoch),
            "metrics_source": "AgentIM.eval_policy",
            "success_rate": float(success_rate),
            "mean_mpjpe_m": float(mean_mpjpe),
            "mean_mpjpe_mm": float(mean_mpjpe) * 1000.0,
            "mean_frame_coverage": float(mean_frame_coverage),
            "num_motions": len(motions),
            "fixed_eval_protocol": {
                "motion_file": str(self.cfg.run.motion_file),
                "initial_pose_file": str(self.cfg.run.initial_pose_file),
                "random_sample": bool(self.cfg.run.random_sample),
                "random_start": bool(self.cfg.run.random_start),
                "num_motions": int(self.cfg.run.num_motions),
                "control_mode": str(self.cfg.run.control_mode),
                "termination_distance": float(self.cfg.env.termination_distance),
                "seed": int(self.cfg.seed),
            },
            "config": cfg_snapshot,
            "motions": motions,
        }

        output_dirs = [
            os.path.join("output", "precision"),
            str(self.cfg.output_dir),
        ]
        for output_dir in dict.fromkeys(output_dirs):
            os.makedirs(output_dir, exist_ok=True)
            metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
            csv_path = os.path.join(output_dir, "evaluation_metrics.csv")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["motion_id", "success", "mpjpe_m", "mpjpe_mm", "frame_coverage"],
                )
                writer.writeheader()
                writer.writerows(motions)
            logger.info("Saved evaluation metrics to %s and %s", metrics_path, csv_path)

    
    def eval_single_thread(self) -> bool:
        """
        Evaluates the policy in a single thread by running an episode.

        Returns:
            bool: True if the episode terminated successfully, False otherwise.
        """
        with to_cpu(*self.sample_modules), torch.no_grad():
            obs_dict, info = self.env.reset()
            state = self.preprocess_obs(obs_dict)
            for t in range(10000):
                actions = self.policy_net.select_action(
                    torch.from_numpy(state).to(self.dtype), True
                )[0].numpy()
                next_obs, reward, terminated, truncated, info = self.env.step(
                    self.preprocess_actions(actions)
                )
                next_state = self.preprocess_obs(next_obs)
                done = terminated or truncated

                if done:                      
                    return not terminated, self.env.mpjpe_value, self.env.frame_coverage
                state = next_state

        # If the loop exits without termination, consider it a failure
        return False, self.env.mpjpe, self.env.frame_coverage
            
            
    def run_policy(self, epoch: int = 0, dump: bool = False) -> dict:
        """
        Runs the trained policy in the environment.

        Args:
            epoch (int, optional): Current epoch number.
            dump (bool, optional): Flag indicating whether to dump run results.

        Returns:
            dict: Run metrics.
        """
        self.env.start_eval(im_eval = False)
        return super().run_policy(epoch, dump)
