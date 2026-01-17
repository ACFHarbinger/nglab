"""
Logging utilities for TensorBoard and console output.
"""


def log_timeseries_values(
    loss, grad_norms, epoch, batch_id, step, output, tb_logger, opts
):
    """
    Log training metrics to console and TensorBoard.

    Args:
        loss (float): Current loss value.
        grad_norms (tuple): (unclipped_grad_norm, clipped_grad_norm).
        epoch (int): Current epoch number.
        batch_id (int): Current batch ID.
        step (int): Current global step.
        output (Tensor): Model output predictions.
        tb_logger: TensorBoard logger instance.
        opts (dict): Configuration options.
    """
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print(f"epoch: {epoch}, train_batch_id: {batch_id}, loss: {loss}")
    print(f"grad_norm: {grad_norms[0]}, clipped: {grad_norms_clipped[0]}")

    # Log values to tensorboard
    if not opts["no_tensorboard"]:
        tb_logger.log_value("loss", loss, step)
        output = output.movedim(0, -1)
        for day in range(1, len(output) + 1):
            tb_logger.log_value(f"predictions_day_{day}", output[day - 1][-1], step)

        tb_logger.log_value("grad_norm", grad_norms[0], step)
        tb_logger.log_value("grad_norm_clipped", grad_norms_clipped[0], step)
