import math
import torch
import torch.nn.functional as F

from torch import nn
from typing import NamedTuple
from torch.nn import DataParallel
from torch.utils.checkpoint import checkpoint

from .graph_encoder import GraphAttentionEncoder


# Attention, Learn to Solve Routing Problems and Heterogeneous Attentions for Solving PDP via DRL
def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    """
    :param input: (batch_size, graph_size, node_dim) input node features
    :return:
    """
    input = do_batch_rep(input, batch_rep)

    costs = []
    pis = []
    
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)
        # pi.view(-1, batch_rep, pi.size(-1))
        cost, mask = get_cost_func(input, pi)

        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)
    # (batch_size * batch_rep, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat(
        [F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis],
        1
    )  # .view(embeddings.size(0), batch_rep * iter_rep, max_length)
    costs = torch.cat(costs, 1)
    
    # (batch_size)
    mincosts, argmincosts = costs.min(-1)
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]
    
    return minpis, mincosts


def compute_in_batches(f, calc_batch_size, *args, n=None):
    """
    Computes memory heavy function f(*args) in batches
    :param n: the total number of elements, optional if it cannot be determined as args[0].size(0)
    :param f: The function that is computed, should take only tensors as arguments and return tensor or tuple of tensors
    :param calc_batch_size: The batch size to use when computing this function
    :param args: Tensor arguments with equally sized first batch dimension
    :return: f(*args), this should be one or multiple tensors with equally sized first batch dimension
    """
    if n is None:
        n = args[0].size(0)
    n_batches = (n + calc_batch_size - 1) // calc_batch_size  # ceil
    if n_batches == 1:
        return f(*args)

    # Run all batches
    # all_res = [f(*batch_args) for batch_args in zip(*[torch.chunk(arg, n_batches) for arg in args])]
    # We do not use torch.chunk such that it also works for other classes that support slicing
    all_res = [f(*(arg[i * calc_batch_size:(i + 1) * calc_batch_size] for arg in args)) for i in range(n_batches)]

    # Allow for functions that return None
    def safe_cat(chunks, dim=0):
        if chunks[0] is None:
            assert all(chunk is None for chunk in chunks)
            return None
        return torch.cat(chunks, dim)

    # Depending on whether the function returned a tuple we need to concatenate each element or only the result
    if isinstance(all_res[0], tuple):
        return tuple(safe_cat(res_chunks, 0) for res_chunks in zip(*all_res))
    return safe_cat(all_res, 0)


class CachedLookup(object):

    def __init__(self, data):
        self.orig = data
        self.key = None
        self.current = None

    def __getitem__(self, key):
        assert not isinstance(key, slice), "CachedLookup does not support slicing, " \
                                           "you can slice the result of an index operation instead"

        if torch.is_tensor(key):  # If tensor, idx all tensors by this tensor:

            if self.key is None:
                self.key = key
                self.current = self.orig[key]
            elif len(key) != len(self.key) or (key != self.key).any():
                self.key = key
                self.current = self.orig[key]

            return self.current

        return super(CachedLookup, self).__getitem__(key)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return self[key]


class AttentionModel(nn.Module):

    def __init__(self,
                 method,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None):
        super(AttentionModel, self).__init__()

        self.method = method
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'
        self.is_pdp = problem.NAME == 'pdp'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        # Problem specific context parameters (placeholder and step context dimension)
        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect
            step_context_dim = embedding_dim + 1

            if self.is_pctsp:
                node_dim = 4  # x, y, expected_prize, penalty
            else:
                node_dim = 3  # x, y, demand / prize

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim)
            
            if self.is_vrp and self.allow_partial:  # Need to include the demand if split delivery allowed
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)
        elif self.is_pdp:
            step_context_dim = embedding_dim 
            node_dim = 2
            self.init_embed_depot = nn.Linear(2, embedding_dim)
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations
            self.init_embed_pick = nn.Linear(node_dim * 2, embedding_dim)
            self.init_embed_delivery = nn.Linear(node_dim, embedding_dim)
        else:  # TSP
            assert problem.NAME == "tsp", "Unsupported problem: {}".format(problem.NAME)
            step_context_dim = 2 * embedding_dim  # Embedding of first and last node
            node_dim = 2  # x, y
            
            # Learned input symbols for first action
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            method=self.method,
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input))

        _log_p, pi = self._inner(input, embeddings)

        cost, mask = self.problem.get_costs(input, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, ll, pi

        return cost, ll

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) // ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, input):
        if self.is_pdp:
            n_loc = input['loc'].size(1)  # input['loc']: [batch_size, graph_size, 2]
            embed_depot = self.init_embed_depot(input['depot'])[:, None, :]

            # [batch_size, graph_size//2, 4] loc
            feature_pick = torch.cat([input['loc'][:, :n_loc // 2, :], input['loc'][:, n_loc // 2:, :]], -1)
            # [batch_size, graph_size//2, 2] loc
            feature_delivery = input['loc'][:, n_loc // 2:, :]  # [batch_size, graph_size//2, 2]
            embed_pick = self.init_embed_pick(feature_pick)
            embed_delivery = self.init_embed_delivery(feature_delivery)
            return torch.cat([embed_depot, embed_pick, embed_delivery], 1)  


        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            if self.is_vrp:
                features = ('demand',)  # [batch_size, graph_size]
            elif self.is_orienteering:
                features = ('prize',)
            else:
                assert self.is_pctsp
                features = ('deterministic_prize', 'penalty')
            return torch.cat(  # [batch_size, graph_size+1, embed_dim]
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_embed(torch.cat((  # [batch_size, graph_size, embed_dim]
                        input['loc'],  # [batch_size, graph_size, 2]
                        *(input[feat][:, :, None] for feat in features)  # [batch_size, graph_size]
                    ), -1))  # [batch_size, graph_size, node_dim]
                ),
                1
            )
        # TSP
        return self.init_embed(input)

    def _inner(self, input, embeddings):

        outputs = []
        sequences = []

        state = self.problem.make_state(input)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        batch_size = state.ids.size(0)

        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state.all_finished()):

            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]

            log_p, mask = self._get_log_p(fixed, state)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension

            state = state.update(selected)

            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
            (input, self.embedder(self._init_embed(input))[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, normalize=True):

        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        if self.is_vrp:
            # Embedding of previous node + remaining capacity
            if from_depot:
                # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
                # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
                return torch.cat(
                    (
                        embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1)),
                        # used capacity is 0 after visiting depot
                        self.problem.VEHICLE_CAPACITY - torch.zeros_like(state.used_capacity[:, :, None])
                    ),
                    -1
                )
            else:
                return torch.cat(
                    (
                        torch.gather(
                            embeddings,  # [batch_size, graph_size, embed_dim]
                            1,
                            current_node.contiguous()
                                .view(batch_size, num_steps, 1)
                                .expand(batch_size, num_steps, embeddings.size(-1))
                        ).view(batch_size, num_steps, embeddings.size(-1)),  # [batch_size, num_steps, embed_dim]
                        self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None]
                    ),
                    -1
                )
                
        if self.is_pdp:
            # Embedding of previous node + remaining capacity
            if from_depot:
                # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
                # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
                return embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1))
            else:
                return torch.gather(
                              embeddings,
                              1,
                              current_node.contiguous()
                                  .view(batch_size, num_steps, 1)
                                  .expand(batch_size, num_steps, embeddings.size(-1))
                          ).view(batch_size, num_steps, embeddings.size(-1))
                        
                        
        elif self.is_orienteering or self.is_pctsp:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous()
                            .view(batch_size, num_steps, 1)
                            .expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    (
                        state.get_remaining_length()[:, :, None]
                        if self.is_orienteering
                        else state.get_remaining_prize_to_collect()[:, :, None]
                    )
                ),
                -1
            )
        else:  # TSP
        
            if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
                if state.i.item() == 0:
                    # First and only step, ignore prev_a (this is a placeholder)
                    return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
                else:
                    return embeddings.gather(
                        1,
                        torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                    ).view(batch_size, 1, -1)
            # More than one step, assume always starting with first
            embeddings_per_step = embeddings.gather(
                1,
                current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
            )
            return torch.cat((
                # First step placeholder, cat in dim 1 (time steps)
                self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
                # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
                torch.cat((
                    embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                    embeddings_per_step
                ), 2)
            ), 1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):

        if self.is_vrp and self.allow_partial:

            # Need to provide information of how much each node has already been served
            # Clone demands as they are needed by the backprop whereas they are updated later
            glimpse_key_step, glimpse_val_step, logit_key_step = \
                self.project_node_step(state.demands_with_depot[:, :, :, None].clone()).chunk(3, dim=-1)

            # Projection of concatenation is equivalent to addition of projections but this is more efficient
            return (
                fixed.glimpse_key + self._make_heads(glimpse_key_step),
                fixed.glimpse_val + self._make_heads(glimpse_val_step),
                fixed.logit_key + logit_key_step,
            )

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )