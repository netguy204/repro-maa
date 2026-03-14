# Chunk: docs/chunks/mdl_curiosity_scorer - MDL-based curiosity signal for curriculum selection
"""MDL-based curiosity scorer for curriculum selection.

Computes a curiosity score by comparing description lengths under a unimodal
vs. bimodal Gaussian model. High scores indicate the learning frontier (mixed
success/failure); low scores indicate mastered or unreachable cells.
"""

from __future__ import annotations

import numpy as np

EPSILON = 1e-10


class MDLScorer:
    """Minimum Description Length curiosity scorer.

    Given a window of recent reward outcomes, computes the MDL improvement
    when switching from a unimodal to a bimodal Gaussian model. The score
    is positive when the data has two modes (the learning frontier) and
    zero when it doesn't (mastered or unreachable).
    """

    def score(self, rewards: list[float]) -> float:
        """Compute the MDL curiosity score for a reward window.

        Args:
            rewards: List of recent reward values.

        Returns:
            Non-negative float curiosity score. Zero for empty or
            single-element windows.
        """
        if len(rewards) < 2:
            return 0.0

        arr = np.asarray(rewards, dtype=np.float64)

        # If all values are identical, unimodal is optimal
        if np.ptp(arr) == 0.0:
            return 0.0

        l_uni = self._unimodal_mdl(arr)
        l_bi = self._bimodal_mdl(arr)
        return float(max(0.0, l_uni - l_bi))

    def _unimodal_mdl(self, rewards: np.ndarray) -> float:
        """Description length under a single-Gaussian model.

        BIC model cost: (k/2) * ln(n) with k=2 (mean, variance).
        Data cost: (n/2)(1 + ln(2π σ²)).
        """
        n = len(rewards)
        var = float(np.var(rewards))  # ML variance (not Bessel-corrected)
        if var < EPSILON:
            var = EPSILON
        data_cost = (n / 2) * (1 + np.log(2 * np.pi * var))
        model_cost = np.log(n)  # (2/2) * ln(n)
        return float(data_cost + model_cost)

    def _bimodal_mdl(self, rewards: np.ndarray) -> float:
        """Description length under a bimodal Gaussian model.

        Finds the optimal hard-clustering split point by sorting rewards
        and scanning all valid partitions. Uses cumulative sums for O(n)
        variance computation.

        BIC model cost: (k/2) * ln(n) with k=5 (two means, two variances,
        mixing weight).
        """
        n = len(rewards)
        sorted_r = np.sort(rewards)

        # Precompute cumulative sums for efficient variance calculation
        cum_sum = np.cumsum(sorted_r)
        cum_sq = np.cumsum(sorted_r**2)

        model_cost = (5 / 2) * np.log(n)
        best = np.inf

        # Try every split: left = [:i], right = [i:] for i in 1..n-1
        for i in range(1, n):
            n_l = i
            n_r = n - i

            # Left cluster statistics
            s_l = cum_sum[i - 1]
            sq_l = cum_sq[i - 1]
            var_l = sq_l / n_l - (s_l / n_l) ** 2
            if var_l < EPSILON:
                var_l = EPSILON

            # Right cluster statistics
            s_r = cum_sum[n - 1] - s_l
            sq_r = cum_sq[n - 1] - sq_l
            var_r = sq_r / n_r - (s_r / n_r) ** 2
            if var_r < EPSILON:
                var_r = EPSILON

            data_cost_l = (n_l / 2) * (1 + np.log(2 * np.pi * var_l))
            data_cost_r = (n_r / 2) * (1 + np.log(2 * np.pi * var_r))

            total = data_cost_l + data_cost_r + model_cost
            if total < best:
                best = total

        return float(best)
