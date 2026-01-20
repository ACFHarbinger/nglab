import numpy as np


class KMediansAlgorithm:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape

        # Initialize centroids randomly from data points
        random_indices = rng.permutation(n_samples)[: self.n_clusters]
        self.cluster_centers_ = X[random_indices]

        for _i in range(self.max_iter):
            # Assign labels based on L1 distance (Manhattan)
            # dist shape: (n_samples, n_clusters)
            # Broadcasting: X[:, none, :] - C[none, :, :]
            dist = np.sum(
                np.abs(X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]),
                axis=2,
            )
            self.labels_ = np.argmin(dist, axis=1)

            new_centers = np.empty((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                mask = self.labels_ == k
                if np.any(mask):
                    new_centers[k] = np.median(X[mask], axis=0)
                else:
                    # Handle empty cluster: re-initialize randomly or keep old
                    # Here we keep old to avoid drift or re-init random point
                    new_centers[k] = self.cluster_centers_[k]

            # Check convergence
            center_shift = np.sum(np.abs(new_centers - self.cluster_centers_))
            if center_shift < self.tol:
                self.cluster_centers_ = new_centers
                break

            self.cluster_centers_ = new_centers

        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet.")
        dist = np.sum(
            np.abs(X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]),
            axis=2,
        )
        return np.argmin(dist, axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_
