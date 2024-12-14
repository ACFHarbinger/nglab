

def log_timeseries_values(loss, grad_norms, epoch, batch_id, step, output, tb_logger, opts):
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('epoch: {}, train_batch_id: {}, loss: {}'.format(epoch, batch_id, loss))
    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts['no_tensorboard']:
        tb_logger.log_value('loss', loss, step)
        output = output.movedim(0, -1)
        for day in range(1, len(output)+1):
            tb_logger.log_value(f'predictions_day_{day}', output[day-1][-1], step)
        
        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)