import numpy as np

class EclatAlgorithm:
    def __init__(self, min_support=0.5, min_confidence=0.7, **kwargs):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rules = []
        self.frequent_itemsets = {}  # tuple(items) -> support_count

    def fit(self, X):
        # Expect X to be (n_samples, n_items) binary matrix (0/1)
        if hasattr(X, "numpy"):
             X = X.numpy()
             
        n_transactions, n_items = X.shape
        min_support_count = self.min_support * n_transactions

        # 1. Transform to Vertical Format: Item -> Set of Transaction IDs (TIDs)
        tid_sets = {}
        for item in range(n_items):
             tids = set(np.where(X[:, item] > 0)[0])
             if len(tids) >= min_support_count:
                 tid_sets[frozenset([item])] = tids
                 self.frequent_itemsets[frozenset([item])] = len(tids)

        # 2. Depth-First Search for frequent itemsets
        # Sort items by support count ascending (heuristic for Eclat)
        sorted_items = sorted(tid_sets.keys(), key=lambda k: len(tid_sets[k]))
        
        self._eclat(sorted_items, tid_sets, min_support_count)
        
        # 3. Generate Rules
        self._generate_rules(n_transactions)
        
        return self

    def _eclat(self, items, tid_sets, min_support_count):
        for i in range(len(items)):
            item_i = items[i]
            tids_i = tid_sets[item_i]
            
            # Intersection with subsequent items
            suffix_items = []
            suffix_tids = {}
            
            for j in range(i + 1, len(items)):
                item_j = items[j]
                tids_j = tid_sets[item_j]
                
                # Intersection
                tids_ij = tids_i.intersection(tids_j)
                
                if len(tids_ij) >= min_support_count:
                    new_itemset = item_i.union(item_j)
                    self.frequent_itemsets[new_itemset] = len(tids_ij)
                    suffix_tids[new_itemset] = tids_ij
                    suffix_items.append(new_itemset)
            
            if suffix_items:
                self._eclat(suffix_items, suffix_tids, min_support_count)

    def _generate_rules(self, n_transactions):
        # Simple rule generation: For each frequent itemset, generate all non-empty subsets
        # Rule: A -> B (where A U B = itemset, A intersection B = empty)
        # Confidence = Support(A U B) / Support(A)
        
        self.rules = []
        import itertools
        
        for itemset, support_count in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue
                
            support = support_count / n_transactions
            
            # Generate all subsets A
            # Iterate through all lengths from 1 to len(itemset)-1
            l = list(itemset)
            for r in range(1, len(itemset)):
                for antecedent in itertools.combinations(l, r):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    if not consequent:
                        continue
                        
                    # Calculate confidence
                    ant_support_count = self.frequent_itemsets.get(antecedent)
                    if not ant_support_count:
                        # Should exist by downward closure property, but just in case
                         continue
                         
                    confidence = support_count / ant_support_count
                    
                    if confidence >= self.min_confidence:
                        self.rules.append({
                            "antecedent": list(antecedent),
                            "consequent": list(consequent),
                            "support": support,
                            "confidence": confidence,
                            "lift": confidence / (self.frequent_itemsets.get(consequent, 0) / n_transactions) if (self.frequent_itemsets.get(consequent, 0) > 0) else 0.0
                        })
