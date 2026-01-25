
"""FP-Growth algorithm implementation for association rule learning."""

from typing import Any, Optional

import pandas as pd


class FPTreeNode:
    """Node in an FP-Tree."""

    def __init__(self, name: Any, count: int, parent: Optional["FPTreeNode"]) -> None:
        """Initialize FP-Tree Node."""
        self.name = name
        self.count = count
        self.parent = parent
        self.children: dict[Any, FPTreeNode] = {}
        self.neighbor: FPTreeNode | None = None


class FPGrowthAlgorithm:
    """FP-Growth Association Rule Learning Algorithm."""

    def __init__(self, min_support: float = 0.5, min_confidence: float = 0.7) -> None:
        """Initialize FP-Growth."""
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets: dict[int, dict[frozenset[Any], float]] = {}
        self.rules: list[dict[str, Any]] = []

    def fit(self, transactions: pd.DataFrame | list[list[Any]]) -> "FPGrowthAlgorithm":
        """Fit the model."""
        if isinstance(transactions, pd.DataFrame):
            items = transactions.columns.tolist()
            transaction_list = []
            for _, row in transactions.iterrows():
                transaction_list.append([items[i] for i, v in enumerate(row) if v > 0])
            transactions_processed: list[list[Any]] = transaction_list
        else:
            transactions_processed = transactions

        n_transactions = len(transactions_processed)
        min_count = self.min_support * n_transactions

        # Step 1: Count item frequencies
        item_counts: dict[Any, int] = {}
        for t in transactions_processed:
            for item in t:
                item_counts[item] = item_counts.get(item, 0) + 1

        # Step 2: Remove infrequent items and sort
        frequent_items = {k: v for k, v in item_counts.items() if v >= min_count}
        if not frequent_items:
            return self

        sorted_items = sorted(frequent_items.items(), key=lambda x: x[1], reverse=True)
        rank = {item[0]: i for i, item in enumerate(sorted_items)}

        # Step 3: Build FP-Tree
        tree_root = FPTreeNode("Null", 1, None)
        header_table: dict[Any, list[Any]] = {
            item[0]: [item[1], None] for item in sorted_items
        }

        for t in transactions_processed:
            frequent_t = [item for item in t if item in frequent_items]
            frequent_t.sort(key=lambda x: rank[x])
            if frequent_t:
                self._insert_tree(frequent_t, tree_root, header_table)

        # Step 4: Mine FP-Tree
        raw_frequent_itemsets: dict[frozenset[Any], int] = {}
        self._mine_tree(header_table, min_count, set(), raw_frequent_itemsets)

        # Normalize frequent_itemsets to support
        normalized_frequent: dict[int, dict[frozenset[Any], float]] = {}
        for itemset, count in raw_frequent_itemsets.items():
            k = len(itemset)
            if k not in normalized_frequent:
                normalized_frequent[k] = {}
            normalized_frequent[k][frozenset(itemset)] = count / n_transactions

        self.frequent_itemsets = normalized_frequent
        self.rules = self._generate_rules()
        return self

    def _insert_tree(
        self, items: list[Any], node: FPTreeNode, header_table: dict[Any, list[Any]]
    ) -> None:
        """Insert items into the FP-Tree."""
        if items[0] in node.children:
            node.children[items[0]].count += 1
        else:
            new_node = FPTreeNode(items[0], 1, node)
            node.children[items[0]] = new_node
            # Link to neighbors
            if header_table[items[0]][1] is None:
                header_table[items[0]][1] = new_node
            else:
                current = header_table[items[0]][1]
                while current.neighbor is not None:
                    current = current.neighbor
                current.neighbor = new_node

        if len(items) > 1:
            self._insert_tree(items[1:], node.children[items[0]], header_table)

    def _mine_tree(
        self,
        header_table: dict[Any, list[Any]],
        min_count: float,
        prefix: set[Any],
        frequent_itemsets: dict[frozenset[Any], int],
    ) -> None:
        """Mine the FP-Tree for frequent itemsets."""
        # Sort items in header table
        sorted_items = [
            item[0] for item in sorted(header_table.items(), key=lambda x: x[1][0])
        ]

        for item in sorted_items:
            new_prefix = prefix.copy()
            new_prefix.add(item)
            frequent_itemsets[frozenset(new_prefix)] = header_table[item][0]

            # Find conditional pattern base
            conditional_patterns: list[tuple[list[Any], int]] = []
            node = header_table[item][1]
            while node is not None:
                path: list[Any] = []
                parent = node.parent
                while parent is not None and parent.name != "Null":
                    path.append(parent.name)
                    parent = parent.parent
                if path:
                    conditional_patterns.append((path, node.count))
                node = node.neighbor

            # Build conditional FP-Tree
            cond_counts: dict[Any, int] = {}
            for path, count in conditional_patterns:
                for p_item in path:
                    cond_counts[p_item] = cond_counts.get(p_item, 0) + count

            # Filter infrequent
            cond_header_table = {
                k: [v, None] for k, v in cond_counts.items() if v >= min_count
            }

            if cond_header_table:
                # Build tree
                cond_tree_root = FPTreeNode("Null", 1, None)
                for path, count in conditional_patterns:
                    frequent_path = [
                        p_item for p_item in path if p_item in cond_header_table
                    ]
                    if frequent_path:
                        self._insert_tree_with_count(
                            frequent_path, count, cond_tree_root, cond_header_table
                        )

                self._mine_tree(
                    cond_header_table, min_count, new_prefix, frequent_itemsets
                )

    def _insert_tree_with_count(
        self,
        items: list[Any],
        count: int,
        node: FPTreeNode,
        header_table: dict[Any, list[Any]],
    ) -> None:
        """Insert items into the conditional FP-Tree with specified count."""
        if items[0] in node.children:
            node.children[items[0]].count += count
        else:
            new_node = FPTreeNode(items[0], count, node)
            node.children[items[0]] = new_node
            if header_table[items[0]][1] is None:
                header_table[items[0]][1] = new_node
            else:
                current = header_table[items[0]][1]
                while current.neighbor is not None:
                    current = current.neighbor
                current.neighbor = new_node
        if len(items) > 1:
            self._insert_tree_with_count(
                items[1:], count, node.children[items[0]], header_table
            )

    def _generate_rules(self) -> list[dict[str, Any]]:
        """Generate association rules."""
        from .apriori import AprioriAlgorithm

        dummy = AprioriAlgorithm(self.min_support, self.min_confidence)
        dummy.frequent_itemsets = self.frequent_itemsets
        return dummy._generate_rules()
