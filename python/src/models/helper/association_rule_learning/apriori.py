"""
Apriori Algorithm for Association Rule Learning.
"""

import pandas as pd


class AprioriAlgorithm:
    """
    Apriori algorithm implementation for finding frequent itemsets and association rules.
    """

    def __init__(self, min_support=0.5, min_confidence=0.7):
        """
        Initialize the Apriori algorithm.

        Args:
            min_support (float, optional): Minimum support threshold. Defaults to 0.5.
            min_confidence (float, optional): Minimum confidence threshold. Defaults to 0.7.
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}
        self.rules = []

    def fit(self, transactions):
        """
        Execute the Apriori algorithm on the dataset.

        Args:
            transactions: List of lists (itemsets) or a pandas DataFrame (one-hot/count).

        Returns:
            self: The fitted model.
        """
        if isinstance(transactions, pd.DataFrame):
            # Assume it's a one-hot encoded DataFrame
            items = transactions.columns.tolist()
            transaction_list = []
            for _, row in transactions.iterrows():
                transaction_list.append([items[i] for i, v in enumerate(row) if v > 0])
            transactions = transaction_list

        self.frequent_itemsets = self._find_frequent_itemsets(transactions)
        self.rules = self._generate_rules()
        return self

    def _find_frequent_itemsets(self, transactions):
        # Basic Apriori implementation
        n_transactions = len(transactions)
        item_counts = {}
        for t in transactions:
            for item in t:
                item_set = frozenset([item])
                item_counts[item_set] = item_counts.get(item_set, 0) + 1

        frequent = {
            item: count / n_transactions
            for item, count in item_counts.items()
            if count / n_transactions >= self.min_support
        }
        all_frequent = {1: frequent}

        k = 2
        while True:
            candidates = self._generate_candidates(all_frequent[k - 1].keys(), k)
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

    def _generate_candidates(self, prev_frequent, k):
        items = list(prev_frequent)
        candidates = set()
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                l1 = sorted(list(items[i]))
                l2 = sorted(list(items[j]))
                if l1[: k - 2] == l2[: k - 2]:
                    candidates.add(items[i] | items[j])
        return candidates

    def _generate_rules(self):
        rules = []
        for k, itemsets in self.frequent_itemsets.items():
            if k < 2:
                continue
            for itemset, support in itemsets.items():
                # Generate all non-empty subsets
                subsets = self._get_all_subsets(itemset)
                for s in subsets:
                    antecedent = frozenset(s)
                    consequent = itemset - antecedent
                    if antecedent and consequent:
                        support_a = self._get_support(antecedent)
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

    def _get_all_subsets(self, itemset):
        from itertools import combinations

        s = list(itemset)
        subsets = []
        for i in range(1, len(s)):
            subsets.extend(combinations(s, i))
        return subsets

    def _get_support(self, itemset):
        k = len(itemset)
        return self.frequent_itemsets.get(k, {}).get(itemset, 0)
