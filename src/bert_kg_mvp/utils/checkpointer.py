import os
import torch
import shutil
from typing import Optional, Dict, Any


class ModelCheckpointer:
    """
    Production-grade model checkpointer for PyTorch.
    Handles continuous checkpointing for fault tolerance and tracks the best model based on a target metric.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        model_name: str,
        mode: str = "max",
        save_top_k: int = 1,
        save_last: bool = True,
    ):
        """
        Args:
            checkpoint_dir (str): Base directory to save checkpoints.
            model_name (str): Prefix for the model (e.g. 'baseline', 'disentangled').
            mode (str): 'max' (e.g. for F1) or 'min' (e.g. for Loss).
            save_top_k (int): Number of best checkpoints to keep.
            save_last (bool): Whether to always save the latest epoch for fault tolerance.
        """
        self.checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.model_name = model_name
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last

        # Track the best metrics (list of tuples: (metric_value, filepath))
        self.best_checkpoints = []
        self.best_metric_value = -float("inf") if mode == "max" else float("inf")

    def _is_better(self, new_val: float, old_val: float) -> bool:
        if self.mode == "max":
            return new_val > old_val
        return new_val < old_val

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        metric_value: float,
    ) -> bool:
        """
        Saves a checkpoint and updates the best model tracking.
        Returns True if this was a new 'best' model.
        """
        # Unwrap model if compiled or DataParallel
        model_to_save = getattr(model, "_orig_mod", model)
        model_to_save = getattr(model_to_save, "module", model_to_save)

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric_value": metric_value,
        }
        if scheduler:
            checkpoint_state["scheduler_state_dict"] = scheduler.state_dict()

        is_new_best = self._is_better(metric_value, self.best_metric_value)
        
        # 1. Save Latest for Fault Tolerance
        if self.save_last:
            latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
            torch.save(checkpoint_state, latest_path)

        # 2. Save Top-K Best Checkpoints
        if self.save_top_k > 0:
            # We save the new checkpoint if it's better than the worst in our top-k, or if we haven't hit k yet
            if len(self.best_checkpoints) < self.save_top_k or self._is_better(metric_value, self.best_checkpoints[-1][0]):
                best_filename = f"best_model_epoch_{epoch}.pt"
                best_filepath = os.path.join(self.checkpoint_dir, best_filename)
                
                torch.save(checkpoint_state, best_filepath)
                
                # Add to list and sort
                self.best_checkpoints.append((metric_value, best_filepath))
                self.best_checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
                
                # Update absolute best
                self.best_metric_value = self.best_checkpoints[0][0]
                
                # Create a symlink or copy to standard 'best_model.pt' for easy downstream consumption
                standard_best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                shutil.copyfile(self.best_checkpoints[0][1], standard_best_path)

                # Prune old checkpoints
                if len(self.best_checkpoints) > self.save_top_k:
                    removed_metric, removed_path = self.best_checkpoints.pop(-1)
                    if os.path.exists(removed_path):
                        os.remove(removed_path)

        return is_new_best
