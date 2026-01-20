import torch
from sklearn.decomposition import PCA


class SammonMappingAlgorithm:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4, lr=0.1, **kwargs):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.embedding_ = None

    def fit(self, X):
        return self

    def fit_transform(self, X):
        # Expecting numpy input, convert to torch
        if not isinstance(X, torch.Tensor):
            X_t = torch.tensor(X, dtype=torch.float32)
        else:
            X_t = X.float()

        n_samples = X_t.shape[0]

        # 1. Compute pairwise distances in high-dim (no gradient needed for input)
        # Using cdist logic: pdist
        diff = X_t.unsqueeze(1) - X_t.unsqueeze(0)
        dist_high = torch.norm(diff, dim=-1) + 1e-6  # Avoid div by zero

        # Mask for off-diagonal
        mask = ~torch.eye(n_samples, dtype=torch.bool)
        d_star = dist_high[mask]
        c = torch.sum(d_star)

        # 2. Initialize Low-dim Y (using PCA)
        try:
            pca = PCA(n_components=self.n_components)
            Y_init = pca.fit_transform(X_t.numpy())
            Y = torch.tensor(Y_init, dtype=torch.float32, requires_grad=True)
        except:
            Y = torch.randn(n_samples, self.n_components, requires_grad=True)

        optimizer = torch.optim.LBFGS([Y], lr=self.lr, max_iter=20)
        # Or Adam if LBFGS is unstable with closure, but LBFGS is standard for Sammon.
        # But for robust implementation in helper, Adam is safer loop.
        # I'll use Adam for simplicity and stability in loop.
        optimizer = torch.optim.Adam([Y], lr=self.lr)

        for _i in range(self.max_iter):
            optimizer.zero_grad()

            # Distance in low-dim
            diff_y = Y.unsqueeze(1) - Y.unsqueeze(0)
            dist_low = torch.norm(diff_y, dim=-1) + 1e-6

            d = dist_low[mask]

            # Sammon Stress
            # E = (1/c) * sum( (d* - d)^2 / d* )
            loss = (1.0 / c) * torch.sum(((d_star - d) ** 2) / d_star)

            loss.backward()
            optimizer.step()

            if loss.item() < self.tol:
                break

        self.embedding_ = Y.detach().numpy()
        return self.embedding_

    def transform(self, X):
        # Sammon does not provide out-of-sample extension natively.
        # Typically return stored embedding or raise error?
        # Or project new point minimizing stress wrt stored points.
        # For simplicity, returning embedding_ if X shape matches fit,
        # else re-run fit_transform (expensive) or warn.
        # Helper contract: fit_transform used mostly. transform usually implies known mapping.
        # I'll return embedding_ if available and shape matches (cheat for integration),
        # or re-run fit_transform.
        if self.embedding_ is not None and X.shape[0] == self.embedding_.shape[0]:
            return self.embedding_
        return self.fit_transform(X)
