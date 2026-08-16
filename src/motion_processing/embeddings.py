"""
Motion embedding — convert motion sequences to fixed-size vectors
for storage, retrieval, and similarity comparison.
"""

import numpy as np
from sklearn.decomposition import PCA


class MotionEmbedder:
    """Convert variable-length motion sequences to fixed-size embeddings."""

    def __init__(self, config: dict):
        cfg = config.get("motion_storage", {})
        self.embedding_dim = cfg.get("embedding_dim", 256)
        self.method = cfg.get("method", "temporal_pooling")
        self._pca = None

    def embed(self, keypoints: np.ndarray) -> np.ndarray:
        """Create embedding for a motion sequence.
        
        Args:
            keypoints: (T, K, 2) normalized keypoint sequence.
            
        Returns:
            (embedding_dim,) vector.
        """
        if self.method == "temporal_pooling":
            return self._temporal_pooling(keypoints)
        elif self.method == "pca":
            return self._pca_embed(keypoints)
        else:
            return self._temporal_pooling(keypoints)

    def embed_batch(self, sequences: list[np.ndarray]) -> np.ndarray:
        """Embed a batch of motion sequences.
        
        Args:
            sequences: list of (T_i, K, 2) arrays.
            
        Returns:
            (N, embedding_dim) array.
        """
        return np.stack([self.embed(seq) for seq in sequences])

    def _temporal_pooling(self, keypoints: np.ndarray) -> np.ndarray:
        """Multi-scale temporal pooling embedding.
        
        Captures both static pose statistics and temporal dynamics:
        1. Mean pose (spatial)
        2. Std pose (variation)
        3. Velocity statistics
        4. Acceleration statistics
        5. Frequency-domain features
        """
        T, K, D = keypoints.shape
        features = []

        # Flatten keypoints per frame: (T, K*D)
        flat = keypoints.reshape(T, -1)

        # 1. Mean pose
        features.append(flat.mean(axis=0))

        # 2. Std of pose
        features.append(flat.std(axis=0))

        # 3. Velocity (first derivative)
        if T > 1:
            velocity = np.diff(flat, axis=0)
            features.append(velocity.mean(axis=0))
            features.append(velocity.std(axis=0))
        else:
            features.append(np.zeros(K * D))
            features.append(np.zeros(K * D))

        # 4. Acceleration (second derivative)
        if T > 2:
            accel = np.diff(flat, n=2, axis=0)
            features.append(accel.mean(axis=0))
            features.append(accel.std(axis=0))
        else:
            features.append(np.zeros(K * D))
            features.append(np.zeros(K * D))

        # 5. Temporal quantiles (capture motion range)
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            features.append(np.quantile(flat, q, axis=0))

        # Concatenate all features
        all_features = np.concatenate(features)

        # Project to target dim via random projection (deterministic)
        rng = np.random.RandomState(42)
        proj_matrix = rng.randn(len(all_features), self.embedding_dim).astype(np.float32)
        proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)

        embedding = all_features.astype(np.float32) @ proj_matrix

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding

    def _pca_embed(self, keypoints: np.ndarray) -> np.ndarray:
        """PCA-based embedding — requires fitting first."""
        T, K, D = keypoints.shape
        flat = keypoints.reshape(T, -1)

        if self._pca is None:
            n_components = min(self.embedding_dim, T, K * D)
            self._pca = PCA(n_components=n_components)
            self._pca.fit(flat)

        transformed = self._pca.transform(flat)
        # Temporal pooling of PCA components
        embedding = np.concatenate([
            transformed.mean(axis=0),
            transformed.std(axis=0),
        ])

        # Pad or truncate to target dim
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dim]

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding.astype(np.float32)

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2))
