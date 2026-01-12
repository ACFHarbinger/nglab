"""
Reinforce Baselines for RL training.
"""
import copy
import torch
import scipy.stats as stats

from pipeline.train import rollout
from utils.model_utils import get_inner_model


# Attention, Learn to Solve Routing Problems
class Baseline(object):
    """
    Abstract Base Class for Reinforce Baselines.
    """
    def wrap_dataset(self, dataset):
        """Wrap dataset if needed."""
        return dataset

    def unwrap_batch(self, batch):
        """Unwrap batch if needed."""
        return batch, None

    def eval(self, x, c):
        """Evaluate baseline on input x and cost c."""
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        """Return learnable parameters."""
        return []

    def epoch_callback(self, model, epoch):
        """Callback at end of epoch."""
        pass

    def state_dict(self):
        """Return state dictionary."""
        return {}

    def load_state_dict(self, state_dict):
        """Load state dictionary."""
        pass


class WarmupBaseline(Baseline):
    """
    Baseline that warms up from another baseline.
    """
    def __init__(self, baseline, warmup_epochs=1, base_baseline=None):
        """
        Initialize.
        """
        super(WarmupBaseline, self).__init__()
        self.baseline = baseline
        self.warmup_epochs = warmup_epochs
        if base_baseline is None:
            base_baseline = NoBaseline()
        self.base_baseline = base_baseline

    def wrap_dataset(self, dataset):
        """Wrap dataset."""
        if self.warmup_epochs > 0:
            return self.base_baseline.wrap_dataset(dataset)
        return self.baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        """Unwrap batch."""
        if self.warmup_epochs > 0:
            return self.base_baseline.unwrap_batch(batch)
        return self.baseline.unwrap_batch(batch)

    def eval(self, x, c):
        """Evaluate."""
        if self.warmup_epochs > 0:
            return self.base_baseline.eval(x, c)
        return self.baseline.eval(x, c)

    def epoch_callback(self, model, epoch):
        """Epoch callback."""
        # Note: epoch is GIVEN as 1-indexed (see run.py)
        if epoch < self.warmup_epochs:
            print("  [*] Custom warmup baseline")
            self.base_baseline.epoch_callback(model, epoch)
        else:
            self.baseline.epoch_callback(model, epoch)

    def state_dict(self):
        """Return state dict."""
        return {
            'baseline': self.baseline.state_dict(),
            'warmup_epochs': self.warmup_epochs
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.baseline.load_state_dict(state_dict['baseline'])
        self.warmup_epochs = state_dict['warmup_epochs']


class NoBaseline(Baseline):
    """
    Baseline that always returns zero.
    """
    def __init__(self):
        """Initialize."""
        super(NoBaseline, self).__init__()

    def eval(self, x, c):
        """Evaluate."""
        return 0, 0  # No baseline, no variance


class ExponentialBaseline(Baseline):
    """
    Exponentially weighted moving average baseline.
    """
    def __init__(self, alpha):
        """
        Initialize.
        """
        super(ExponentialBaseline, self).__init__()
        self.alpha = alpha
        self.v = None

    def eval(self, x, c):
        """Evaluate."""
        if self.v is None:
            v = c.mean()
        else:
            v = self.alpha * self.v + (1.0 - self.alpha) * c.mean()

        self.v = v
        return v, 0  # No variance

    def state_dict(self):
        """Return state dict."""
        return {'v': self.v}

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.v = state_dict['v']


class CriticBaseline(Baseline):
    """
    Baseline using a critic network.
    """
    def __init__(self, critic):
        """Initialize."""
        super(CriticBaseline, self).__init__()
        self.critic = critic

    def eval(self, x, c):
        """Evaluate."""
        v = self.critic(x)
        # Detach v since it is only used as a baseline and we optimize the critic separately
        return v.detach(), 0  # To be improved

    def get_learnable_parameters(self):
        """Return learnable parameters."""
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        """Epoch callback."""
        pass

    def state_dict(self):
        """Return state dict."""
        return {'critic': self.critic.state_dict()}

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.critic.load_state_dict(state_dict['critic'])


class RolloutBaseline(Baseline):
    """
    Rollout baseline using a greedy policy.
    """
    def __init__(self, model, problem, opts, epoch=0):
        """
        Initialize.
        """
        super(Baseline, self).__init__()
        self.problem = problem
        self.opts = opts
        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        """Update the baseline model."""
        self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset

        if dataset is not None:
            if len(dataset) != self.opts['val_size']:
                print("Warning: not using saved baseline dataset since val_size does not match")
                dataset = None
            elif (dataset[0] if self.problem.NAME == 'tsp' else dataset[0]['loc']).size(0) != self.opts['graph_size']:
                print("Warning: not using saved baseline dataset since graph_size does not match")
                dataset = None

        if dataset is None:
            self.dataset = self.problem.make_dataset(
                size=self.opts['graph_size'], num_samples=self.opts['val_size'], distribution=self.opts['data_distribution'])
        else:
            self.dataset = dataset
        print("Evaluating baseline model on evaluation dataset")
        self.bl_vals = rollout(self.model, self.dataset, self.opts).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def wrap_dataset(self, dataset):
        """Wrap dataset."""
        print("Evaluating baseline on dataset...")
        # Need to convert baseline to 2D to prevent converting to double, see
        # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
        return BaselineDataset(dataset, rollout(self.model, dataset, self.opts).view(-1, 1))

    def unwrap_batch(self, batch):
        """Unwrap batch."""
        return batch['data'], batch['baseline'].view(-1)  # Flatten result to undo wrapping as 2D

    def eval(self, x, c):
        """Evaluate."""
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            v, _ = self.model(x)

        # There is no loss
        return v, 0

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """
        print("Evaluating candidate model on evaluation dataset")
        candidate_vals = rollout(model, self.dataset, self.opts).cpu().numpy()
        candidate_mean = candidate_vals.mean()
        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))

        if candidate_mean - self.mean < 0:
            # Calc p value
            t, p = stats.ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < self.opts['bl_alpha']:
                print('Update baseline')
                self._update_model(model, epoch)

    def state_dict(self):
        """Return state dict."""
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        # We make it such that it works whether model was saved as data parallel or not
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(get_inner_model(state_dict['model']).state_dict())
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])


class BaselineDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper to include baseline values.
    """
    def __init__(self, dataset=None, baseline=None):
        """
        Initialize.
        """
        super(BaselineDataset, self).__init__()
        self.dataset = dataset
        self.baseline = baseline
        assert (len(self.dataset) == len(self.baseline))

    def __getitem__(self, item):
        """Get item."""
        return {
            'data': self.dataset[item],
            'baseline': self.baseline[item]
        }

    def __len__(self):
        """Return length."""
        return len(self.dataset)