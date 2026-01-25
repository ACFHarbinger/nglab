
"""Eclat algorithm implementation for association rule learning."""

import itertools
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


class EclatAlgorithm:
    """Eclat (Equivalence Class Transformation) Algorithm for Association Rule Learning."""

    def __init__(
        self, min_support: float = 0.5, min_confidence: float = 0.7, **kwargs: Any
    ) -> None:
        """Initialize Eclat."""
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rules: list[dict[str, Any]] = []
        self.frequent_itemsets: dict[frozenset[Any], int] = (
            {}
        )  # itemset -> support_count

    def fit(self, X: NDArray[Any] | Any) -> "EclatAlgorithm":  # noqa: N803
        """Fit the model."""
        # Expect X to be (n_samples, n_items) binary matrix (0/1)
        if hasattr(X, "numpy"):
            X_data = cast(Any, X).numpy()
        else:
            X_data = np.asarray(X)

        n_transactions, n_items = X_data.shape
        min_support_count = self.min_support * n_transactions

        # 1. Transform to Vertical Format: Item -> Set of Transaction IDs (TIDs)
        tid_sets: dict[frozenset[Any], set[int]] = {}
        for item in range(n_items):
            tids = set(np.where(X_data[:, item] > 0)[0])
            if len(tids) >= min_support_count:
                itemset = frozenset([item])
                tid_sets[itemset] = tids
                self.frequent_itemsets[itemset] = len(tids)

        # 2. Depth-First Search for frequent itemsets
        sorted_itemsets = sorted(tid_sets.keys(), key=lambda k: len(tid_sets[k]))
        self._eclat(sorted_itemsets, tid_sets, min_support_count)

        # 3. Generate Rules
        self._generate_rules(n_transactions)
        return self

    def _eclat(
        self,
        itemsets: list[frozenset[Any]],
        tid_sets: dict[frozenset[Any], set[int]],
        min_support_count: float,
    ) -> None:
        """Recursive DFS for Eclat."""
        for i in range(len(itemsets)):
            itemset_i = itemsets[i]
            tids_i = tid_sets[itemset_i]

            suffix_itemsets: list[frozenset[Any]] = []
            suffix_tid_sets: dict[frozenset[Any], set[int]] = {}

            for j in range(i + 1, len(itemsets)):
                itemset_j = itemsets[j]
                tids_j = tid_sets[itemset_j]

                # Intersection
                tids_ij = tids_i.intersection(tids_j)

                if len(tids_ij) >= min_support_count:
                    new_itemset = itemset_i.union(itemset_j)
                    self.frequent_itemsets[new_itemset] = len(tids_ij)
                    suffix_tid_sets[new_itemset] = tids_ij
                    suffix_itemsets.append(new_itemset)

            if suffix_itemsets:
                self._eclat(suffix_itemsets, suffix_tid_sets, min_support_count)

    def _generate_rules(self, n_transactions: int) -> None:
        """Generate association rules from frequent itemsets."""
        self.rules = []
        for itemset, support_count in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue

            support = support_count / n_transactions
            items_list = list(itemset)
            for r in range(1, len(itemset)):
                for antecedent in itertools.combinations(items_list, r):
                    antecedent_set = frozenset(antecedent)
                    consequent_set = itemset - antecedent_set

                    if not consequent_set:
                        continue

                    ant_support_count = self.frequent_itemsets.get(antecedent_set)
                    if not ant_support_count:
                        continue

                    confidence = support_count / ant_support_count
                    if confidence >= self.min_confidence:
                        consequent_support_count = self.frequent_itemsets.get(
                            consequent_set, 0
                        )
                        lift = (
                            (confidence / (consequent_support_count / n_transactions))
                            if (consequent_support_count > 0)
                            else 0.0
                        )

                        self.rules.append(
                            {
                                "antecedent": list(antecedent_set),
                                "consequent": list(consequent_set),
                                "support": support,
                                "confidence": confidence,
                                "lift": lift,
                            }
                        )
