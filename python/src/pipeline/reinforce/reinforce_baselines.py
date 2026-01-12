"""
REINFORCE Baselines for NGLab.

Provides various baseline implementations for the REINFORCE algorithm,
including Rollout, Critic, and Exponential baselines, to reduce variance
during training.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import ttest_rel
import copy
from utils.ARP_HADRL.train import rollout, get_inner_model

# Attention, Learn to Solve Routing Problems and Heterogeneous Attentions for Solving PDP via DRL
class Baseline(object):
    """
    Abstract Base Class for REINFORCE baselines.
    """

    def wrap_dataset(self, dataset):
        """
        Wrap dataset if the baseline requires extra data.
        """
        return dataset

    def unwrap_batch(self, batch):
        """
        Unwrap a batch to separate data and baseline values.
        """
        return batch, None

    def eval(self, x, c):
        """
        Evaluate the baseline for a given state and cost.
        """
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        """
        Return inner model parameters if learnable.
        """
        return []

    def epoch_callback(self, model, epoch):
        """
        Perform any necessary updates at the end of an epoch.
        """
        pass

    def state_dict(self):
        """
        Return the state of the baseline.
        """
        return {}

    def load_state_dict(self, state_dict):
        """
        Load the baseline state.
        """
        pass


class WarmupBaseline(Baseline):
    """
    Baseline that warms up from an exponential baseline to another baseline.
    """

    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8, ):
        """
        Initialize Warmup baseline.
        """
        super(Baseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        """
        Wrap dataset according to the current warmup stage.
        """
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        """
        Unwrap batch according to the current warmup stage.
        """
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c):
        """
        Evaluate baseline combining warmup and final baselines.
        """

        if self.alpha == 1:
            return self.baseline.eval(x, c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c)
        v, l = self.baseline.eval(x, c)
        vw, lw = self.warmup_baseline.eval(x, c)
        # Return convex combination of baseline and of loss
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha) * lw

    def epoch_callback(self, model, epoch):
        """
        Update warmup alpha and call inner callback.
        """
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, epoch)
        if epoch < self.n_epochs:
            self.alpha = (epoch + 1) / float(self.n_epochs)
            print("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self):
        """
        Return inner baseline state dict.
        """
        # Checkpointing within warmup stage makes no sense, only save inner baseline
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        """
        Load inner baseline state.
        """
        # Checkpointing within warmup stage makes no sense, only load inner baseline
        self.baseline.load_state_dict(state_dict)


class NoBaseline(Baseline):
    """
    Empty baseline that returns zero.
    """

    def eval(self, x, c):
        """
        Return zero baseline and loss.
        """
        return 0, 0  # No baseline, no loss


class ExponentialBaseline(Baseline):
    """
    Exponentially moving average baseline.
    """

    def __init__(self, beta):
        """
        Initialize Exponential baseline.
        """
        super(Baseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, x, c):
        """
        Update and return EMA baseline value.
        """

        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss

    def state_dict(self):
        """
        Return EMA state.
        """
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict):
        """
        Load EMA state.
        """
        self.v = state_dict['v']


class CriticBaseline(Baseline):
    """
    Baseline based on a critic neural network.
    """

    def __init__(self, critic):
        """
        Initialize Critic baseline.
        """
        super(Baseline, self).__init__()

        self.critic = critic

    def eval(self, x, c):
        """
        Evaluate critic network and return value and MSE loss.
        """
        v = self.critic(x)
        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach(), F.mse_loss(v, c.detach())

    def get_learnable_parameters(self):
        """
        Return critic parameters.
        """
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        """
        Callback placeholder.
        """
        pass

    def state_dict(self):
        """
        Return critic state.
        """
        return {
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict):
        """
        Load critic state.
        """
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})


class RolloutBaseline(Baseline):
    """
    Baseline based on a fixed rollout model (Greedy baseline).
    """

    def __init__(self, model, problem, opts, epoch=0):
        """
        Initialize Rollout baseline.
        """
        super(Baseline, self).__init__()

        self.problem = problem
        self.opts = opts

        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        """
        Update the baseline model.
        """
        self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset

        if dataset is not None:
            if len(dataset) != self.opts.val_size:
                print("Warning: not using saved baseline dataset since val_size does not match")
                dataset = None
            elif (dataset[0] if self.problem.NAME == 'tsp' else dataset[0]['loc']).size(0) != self.opts.graph_size:
                print("Warning: not using saved baseline dataset since graph_size does not match")
                dataset = None

        if dataset is None:
            self.dataset = self.problem.make_dataset(
                size=self.opts.graph_size, num_samples=self.opts.val_size, distribution=self.opts.data_distribution)
        else:
            self.dataset = dataset
        print("Evaluating baseline model on evaluation dataset")
        self.bl_vals = rollout(self.model, self.dataset, self.opts).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def wrap_dataset(self, dataset):
        """
        Wrap dataset with baseline values.
        """
        print("Evaluating baseline on dataset...")
        # Need to convert baseline to 2D to prevent converting to double, see
        # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
        return BaselineDataset(dataset, rollout(self.model, dataset, self.opts).view(-1, 1))

    def unwrap_batch(self, batch):
        """
        Unwrap batch previously wrapped by wrap_dataset.
        """
        return batch['data'], batch['baseline'].view(-1)  # Flatten result to undo wrapping as 2D

    def eval(self, x, c):
        """
        Evaluate baseline model on a batch.
        """
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
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < self.opts.bl_alpha:
                print('Update baseline')
                self._update_model(model, epoch)

    def state_dict(self):
        """
        Return baseline model state.
        """
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        """
        Load baseline model state.
        """
        # We make it such that it works whether model was saved as data parallel or not
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(get_inner_model(state_dict['model']).state_dict())
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])


class BaselineDataset(Dataset):
    """
    Dataset wrapper that includes baseline values.
    """

    def __init__(self, dataset=None, baseline=None):
        """
        Initialize baseline dataset.
        """
        super(BaselineDataset, self).__init__()

        self.dataset = dataset
        self.baseline = baseline
        assert (len(self.dataset) == len(self.baseline))

    def __getitem__(self, item):
        """
        Get wrapped item index.
        """
        return {
            'data': self.dataset[item],
            'baseline': self.baseline[item]
        }

    def __len__(self):
        """
        Return length matching original dataset.
        """
        return len(self.dataset)
