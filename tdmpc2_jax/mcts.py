# import datetime
import math
import os
import random
import time
import datetime
import yaml
import pickle
import numpy as np
import ray
from matplotlib import pyplot as plt

import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter
import jax
import jax.numpy as jnp

class TreeNode:
    """
    TreeNode is an individual node of a search tree.
    It has one potential child for each potential action which, if it exists, is another TreeNode
    Its function is to hold the relevant statistics for deciding which action to take.
    """

    def __init__(
        self,
        latent,
        action_size,
        val_pred=None,
        pol_pred=None,
        parent=None,
        reward=0,
        minmax=None,
        config=None,
        num_visits=1,
        lstm_hiddens=None,
    ):

        self.action_size = action_size
        self.children = [None] * action_size
        self.latent = latent
        self.val_pred = val_pred
        self.pol_pred = pol_pred
        self.parent = parent
        self.average_val = val_pred
        self.num_visits = num_visits
        self.reward = reward

        self.minmax = minmax
        self.config = config
        self.lstm_hiddens = lstm_hiddens

    def insert(
        self,
        action_n,
        latent,
        val_pred,
        pol_pred,
        reward,
        minmax = None,
        config = None,
        lstm_hiddens=None,
    ):
        # The implementation here differs from the open MuZero (werner duvaud)
        # by only initializing tree nodes when they are chosen, rather than when their parent is chosen
        if self.children[action_n] is None:
            new_child = TreeNode(
                latent=latent,
                val_pred=val_pred,
                pol_pred=pol_pred,
                action_size=self.action_size,
                parent=self,
                reward=reward,
                minmax=minmax,
                config=self.config,
                lstm_hiddens=lstm_hiddens,
            )

            self.children[action_n] = new_child

        else:
            raise ValueError("This node has already been traversed")

    def update_val(self, curr_val):
        """Updates the average value of a node when a new value is receivied
        copies the formula of the muzero paper rather than the neater form of
        just tracking the sum and dividng as needed
        """
        nmtr = self.average_val * self.num_visits + curr_val
        dnmtr = self.num_visits + 1
        self.average_val = nmtr / dnmtr

    def action_score(self, action_n, total_visit_count):
        """
        Scoring function for the different potential actions, following the formula in Appendix B of muzero
        """
        c1 = 1.25
        c2 = 19652

        child = self.children[action_n]

        n = child.num_visits if child else 0

        q = self.minmax.normalize(child.average_val) if child else 0

        # p here is the prior - the expectation of what the the policy will look like
        prior = self.pol_pred[0][action_n]

        # This term increases the prior on those actions which have been taken only a small fraction
        # of the current number of visits to this node
        explore_term = math.sqrt(total_visit_count) / (1 + n)

        # This is intended to more heavily weight the prior as we take more and more actions.
        # Its utility is questionable, because with on the order of 100 simulations, this term will always be
        # close to 1.
        balance_term = c1 + math.log((total_visit_count + c2 + 1) / c2)
        score = q + (prior * explore_term * balance_term)
        return score

    def pick_action(self):
        """Gets the score each of the potential actions and picks the one with the highest"""
        total_visit_count = sum([a.num_visits if a else 0 for a in self.children])

        scores = [
            self.action_score(a, total_visit_count) for a in range(self.action_size)
        ]
        maxscore = max(scores)

        # Need to be careful not to always pick the first action as it common that two are scored identically
        action = np.random.choice(
            [a for a in range(self.action_size) if scores[a] == maxscore]
        )
        return action

    def pick_game_action(self, temperature):
        """
        Picks the action to actually be taken in game,
        taken by the root node after the full tree has been generated.
        Note that it only uses the visit counts, rather than the score or prior,
        these impact the decision only through their impact on where to visit
        """

        visit_counts = [a.num_visits if a else 0 for a in self.children]

        # zero temperature means always picking the highest visit count
        if temperature == 0:
            max_vis = max(visit_counts)
            action = np.random.choice(
                [a for a in range(self.action_size) if visit_counts[a] == max_vis]
            )

        # If temperature is non-zero, raise (visit_count + 1) to power (1 / T)
        # scale these to a probability distribution and use to select action
        else:
            scores = [(vc + 1) ** (1 / temperature) for vc in visit_counts]
            total_score = sum(scores)
            adjusted_scores = [score / total_score for score in scores]

            action = np.random.choice(self.action_size, p=adjusted_scores)

        # Prints a lot of useful information for how the algorithm is making decisions
        val_preds = [c.val_pred if c else 0 for c in self.children]
        # print(visit_counts, self.val_pred, val_preds)

        # one_hot_encode
        action = jnp.eye(self.action_size)[action]
        return action


class MinMax:
    """
    This class tracks the smallest and largest values that have been seen
    so that it can normalize the values
    this is for when deciding which branch of the tree to explore
    by putting the values on a 0-1 scale, they become comparable with the probabilities
    given by the prior

    It comes pretty much straight from the MuZero pseudocode
    """

    def __init__(self):
        # initialize at +-inf so that any value will supercede the max/min
        self.max_value = -float("inf")
        self.min_value = float("inf")

    def update(self, val):
        self.max_value = max(float(val), self.max_value)
        self.min_value = min(float(val), self.min_value)

    def normalize(self, val):
        # places val between 0 - 1 linearly depending on where it sits between min_value and max_value
        if self.max_value > self.min_value:
            return (val - self.min_value) / (self.max_value - self.min_value)
        else:
            return val


def add_dirichlet(prior, dirichlet_alpha, explore_frac):
    noise = torch.tensor(
        np.random.dirichlet([dirichlet_alpha] * len(prior)), device=prior.device
    )
    new_prior = (1 - explore_frac) * prior + explore_frac * noise
    return new_prior


last_time = datetime.datetime.now()

def support_to_scalar(support, epsilon=0.00001):
    print(f'Converting support to scalar: {support.shape}')
    squeeze = False
    if support.ndim == 1:
        squeeze = True
        support = support[None, :]  # Use None to add a batch dimension

    if not jnp.all(jnp.abs(jnp.sum(support, axis=1) - 1) < 0.01):
        print(support)

    half_width = (support.shape[1] - 1) // 2
    vals = jnp.arange(-half_width, half_width + 1, dtype=support.dtype)

    # Dot product of the two
    print(f'Vals: {vals.shape}, Support: {support.shape}')
    out_val = jnp.einsum("i,bi->b", vals, support)

    sign_out = jnp.where(out_val >= 0, 1, -1)

    num = jnp.sqrt(1 + 4 * epsilon * (jnp.abs(out_val) + 1 + epsilon)) - 1
    res = (num / (2 * epsilon)) ** 2

    output = sign_out * (res - 1)

    if squeeze:
        output = output.squeeze(axis=0)

    return output

if __name__ == "__main__":
    pass