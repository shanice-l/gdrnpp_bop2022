# just a reference implementation, no use

import logging
import time
import torch
from detectron2.engine import SimpleTrainer
import core.utils.my_comm as comm

logger = logging.getLogger(__name__)

try:
    import apex
    from apex import amp
except:
    logger.exception("Please install apex from https://www.github.com/nvidia/apex to use this ApexTrainer.")


class ApexTrainer(SimpleTrainer):
    """Like :class:`SimpleTrainer`, but uses NVIDIA's apex automatic mixed
    precision in the training loop."""

    def __init__(self, model, data_loader, optimizer, apex_opt_level="O1"):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
        """
        if comm.get_world_size() > 1:
            model, optimizer = amp.initialize(model, optimizer, opt_level=apex_opt_level)
        super().__init__(model, data_loader, optimizer)

    def run_step(self):
        """Implement the AMP training logic using apex."""
        assert self.model.training, "[ApexTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[ApexTrainer] CUDA is required for AMP training!"

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        with amp.scale_loss(losses, self.optimizer) as scaled_loss:
            scaled_loss.backwward()

        self._write_metrics(loss_dict, data_time)

        self.optimizer.step()

    def state_dict(self):
        ret = super().state_dict()
        # save amp state according to
        # https://nvidia.github.io/apex/amp.html#checkpointing
        ret["amp"] = amp.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "amp" in state_dict:
            amp.load_state_dict(state_dict["amp"])
