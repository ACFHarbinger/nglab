"""
Apriori Algorithm for Association Rule Learning.
"""

from typing import Any, Dict, List, Set, Union, Tuple
import pandas as pd


class AprioriAlgorithm:
    """
    Apriori algorithm implementation for finding frequent itemsets and association rules.
    """

    def __init__(self, min_support: float = 0.5, min_confidence: float = 0.7) -> None:
        """
        Initialize the Apriori algorithm.
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets: Dict[int, Dict[frozenset[Any], float]] = {}
        self.rules: List[Dict[str, Any]] = []

    def fit(self, transactions: Union[pd.DataFrame, List[List[Any]]]) -> "AprioriAlgorithm":
        """
        Execute the Apriori algorithm on the dataset.
        """
        processed_transactions: List[List[Any]]
        if isinstance(transactions, pd.DataFrame):
            items = transactions.columns.tolist()
            transaction_list = []
            for _, row in transactions.iterrows():
                transaction_list.append([items[i] for i, v in enumerate(row) if v > 0])
            processed_transactions = transaction_list
        else:
            processed_transactions = transactions

        self.frequent_itemsets = self._find_frequent_itemsets(processed_transactions)
        self.rules = self._generate_rules()
        return self

    def _find_frequent_itemsets(self, transactions: List[List[Any]]) -> Dict[int, Dict[frozenset[Any], float]]:
        """Find frequent itemsets using Apriori logic."""
        n_transactions = len(transactions)
        item_counts: Dict[frozenset[Any], int] = {}
        for t in transactions:
            for item in t:
                item_set = frozenset([item])
                item_counts[item_set] = item_counts.get(item_set, 0) + 1

        frequent = {
            item: count / n_transactions
            for item, count in item_counts.items()
            if count / n_transactions >= self.min_support
        }
        all_frequent: Dict[int, Dict[frozenset[Any], float]] = {1: frequent}

        k = 2
        while True:
            candidates = self._generate_candidates(set(all_frequent[k - 1].keys()), k)
            if not candidates:
                break

            counts = {c: 0 for c in candidates}
            for t in transactions:
                t_set = set(t)
                for c in candidates:
                    if c.issubset(t_set):
                        counts[c] += 1

            frequent_k = {
                c: count / n_transactions
                for c, count in counts.items()
                if count / n_transactions >= self.min_support
            }
            if not frequent_k:
                break
            all_frequent[k] = frequent_k
            k += 1

        return all_frequent

    def _generate_candidates(self, prev_frequent: Set[frozenset[Any]], k: int) -> Set[frozenset[Any]]:
        """Generate candidates of size k from frequent itemsets of size k-1."""
        items = list(prev_frequent)
        candidates: Set[frozenset[Any]] = set()
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                l1 = sorted(list(items[i]))
                l2 = sorted(list(items[j]))
                if l1[: k - 2] == l2[: k - 2]:
                    candidates.add(items[i] | items[j])
        return candidates

    def _generate_rules(self) -> List[Dict[str, Any]]:
        """Generate association rules from frequent itemsets."""
        rules: List[Dict[str, Any]] = []
        for k, itemsets in self.frequent_itemsets.items():
            if k < 2:
                continue
            for itemset, support in itemsets.items():
                subsets = self._get_all_subsets(itemset)
                for s in subsets:
                    antecedent = frozenset(s)
                    consequent = itemset - antecedent
                    if antecedent and consequent:
                        support_a = self._get_support(antecedent)
                        if support_a > 0:
                            confidence = support / support_a
                            if confidence >= self.min_confidence:
                                rules.append(
                                    {
                                        "antecedent": list(antecedent),
                                        "consequent": list(consequent),
                                        "support": support,
                                        "confidence": confidence,
                                    }
                                )
        return rules

    def _get_all_subsets(self, itemset: frozenset[Any]) -> List[Tuple[Any, ...]]:
        """Get all non-empty subsets of an itemset."""
        from itertools import combinations

        s = list(itemset)
        subsets: List[Tuple[Any, ...]] = []
        for i in range(1, len(s)):
            subsets.extend(combinations(s, i))
        return subsets

    def _get_support(self, itemset: frozenset[Any]) -> float:
        """Get the support of a given itemset."""
        k = len(itemset)
        return self.frequent_itemsets.get(k, {}).get(itemset, 0.0)
