import pandas as pd


class FPTreeNode:
    def __init__(self, name, count, parent):
        self.name = name
        self.count = count
        self.parent = parent
        self.children = {}
        self.neighbor = None


class FPGrowthAlgorithm:
    def __init__(self, min_support=0.5, min_confidence=0.7):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}
        self.rules = []

    def fit(self, transactions):
        if isinstance(transactions, pd.DataFrame):
            items = transactions.columns.tolist()
            transaction_list = []
            for _, row in transactions.iterrows():
                transaction_list.append([items[i] for i, v in enumerate(row) if v > 0])
            transactions = transaction_list

        n_transactions = len(transactions)
        min_count = self.min_support * n_transactions

        # Step 1: Count item frequencies
        item_counts = {}
        for t in transactions:
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
        header_table = {item[0]: [item[1], None] for item in sorted_items}

        for t in transactions:
            frequent_t = [item for item in t if item in frequent_items]
            frequent_t.sort(key=lambda x: rank[x])
            if frequent_t:
                self._insert_tree(frequent_t, tree_root, header_table)

        # Step 4: Mine FP-Tree
        self.frequent_itemsets = {}
        self._mine_tree(header_table, min_count, set(), self.frequent_itemsets)

        # Normalize frequent_itemsets to support
        normalized_frequent = {}
        for itemset, count in self.frequent_itemsets.items():
            k = len(itemset)
            if k not in normalized_frequent:
                normalized_frequent[k] = {}
            normalized_frequent[k][frozenset(itemset)] = count / n_transactions

        self.frequent_itemsets = normalized_frequent
        # Rule generation same as Apriori (could be refactored into a base class if needed)
        self.rules = self._generate_rules()
        return self

    def _insert_tree(self, items, node, header_table):
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

    def _mine_tree(self, header_table, min_count, prefix, frequent_itemsets):
        # Sort items in header table
        sorted_items = [
            item[0] for item in sorted(header_table.items(), key=lambda x: x[1][0])
        ]

        for item in sorted_items:
            new_prefix = prefix.copy()
            new_prefix.add(item)
            frequent_itemsets[frozenset(new_prefix)] = header_table[item][0]

            # Find conditional pattern base
            conditional_patterns = []
            node = header_table[item][1]
            while node is not None:
                path = []
                parent = node.parent
                while parent.name != "Null":
                    path.append(parent.name)
                    parent = parent.parent
                if path:
                    conditional_patterns.append((path, node.count))
                node = node.neighbor

            # Build conditional FP-Tree
            cond_header_table = {}
            for path, count in conditional_patterns:
                for p_item in path:
                    cond_header_table[p_item] = cond_header_table.get(p_item, 0) + count

            # Filter infrequent
            cond_header_table = {
                k: [v, None] for k, v in cond_header_table.items() if v >= min_count
            }

            if cond_header_table:
                # Build tree
                cond_tree_root = FPTreeNode("Null", 1, None)
                for path, count in conditional_patterns:
                    frequent_path = [
                        p_item for p_item in path if p_item in cond_header_table
                    ]
                    # Sort by rank... actually need rank of conditional items
                    # For simplicity, we skip full sorting here as path is already mostly sorted
                    if frequent_path:
                        # Re-insert with counts
                        self._insert_tree_with_count(
                            frequent_path, count, cond_tree_root, cond_header_table
                        )

                self._mine_tree(
                    cond_header_table, min_count, new_prefix, frequent_itemsets
                )

    def _insert_tree_with_count(self, items, count, node, header_table):
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

    def _generate_rules(self):
        # reuse apriori rule gen
        from .apriori import AprioriAlgorithm

        dummy = AprioriAlgorithm(self.min_support, self.min_confidence)
        dummy.frequent_itemsets = self.frequent_itemsets
        return dummy._generate_rules()
