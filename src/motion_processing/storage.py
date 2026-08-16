"""
Motion storage — persist and retrieve motion data using HDF5 + FAISS.

Stores:
- Raw normalized keypoints (HDF5)
- Motion embeddings for fast retrieval (FAISS index)
- Metadata (source video, labels, tags)
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class MotionStorage:
    """Persistent motion database with vector search."""

    def __init__(self, config: dict):
        cfg = config.get("motion_storage", {})
        self.db_path = Path(cfg.get("db_path", "data/motion_db"))
        self.embedding_dim = cfg.get("embedding_dim", 256)
        self.index_type = cfg.get("index_type", "Flat")
        self.nlist = cfg.get("nlist", 100)

        self.db_path.mkdir(parents=True, exist_ok=True)
        self.h5_path = self.db_path / "motions.h5"
        self.index_path = self.db_path / "faiss.index"
        self.meta_path = self.db_path / "metadata.json"

        self._index = None
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> list[dict]:
        if self.meta_path.exists():
            with open(self.meta_path, "r") as f:
                return json.load(f)
        return []

    def _save_metadata(self):
        with open(self.meta_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def _get_faiss_index(self):
        """Lazy-load or create FAISS index."""
        if self._index is not None:
            return self._index

        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS is required: pip install faiss-cpu")

        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            logger.info("Loaded FAISS index with %d vectors", self._index.ntotal)
        else:
            if self.index_type == "Flat":
                self._index = faiss.IndexFlatIP(self.embedding_dim)
            elif self.index_type == "IVFFlat":
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self._index = faiss.IndexIVFFlat(
                    quantizer, self.embedding_dim, self.nlist
                )
            else:
                self._index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info("Created new FAISS index (type=%s, dim=%d)", self.index_type, self.embedding_dim)

        return self._index

    def save_motion(
        self,
        motion_id: str,
        keypoints: np.ndarray,
        scores: np.ndarray,
        embedding: np.ndarray,
        metadata: dict | None = None,
    ):
        """Save a motion sequence to the database.
        
        Args:
            motion_id: Unique identifier for this motion.
            keypoints: (T, K, 2) normalized keypoints.
            scores: (T, K) confidence scores.
            embedding: (D,) motion embedding vector.
            metadata: Optional dict with tags, source info, etc.
        """
        # Save keypoints to HDF5
        with h5py.File(self.h5_path, "a") as f:
            grp = f.require_group(motion_id)
            if "keypoints" in grp:
                del grp["keypoints"]
            if "scores" in grp:
                del grp["scores"]
            if "embedding" in grp:
                del grp["embedding"]
            grp.create_dataset("keypoints", data=keypoints, compression="gzip")
            grp.create_dataset("scores", data=scores, compression="gzip")
            grp.create_dataset("embedding", data=embedding)

        # Add embedding to FAISS
        index = self._get_faiss_index()
        emb = embedding.reshape(1, -1).astype(np.float32)
        index.add(emb)
        self._save_faiss_index()

        # Store metadata
        entry = {
            "motion_id": motion_id,
            "n_frames": int(keypoints.shape[0]),
            "n_keypoints": int(keypoints.shape[1]),
            "index_position": index.ntotal - 1,
        }
        if metadata:
            entry.update(metadata)
        self._metadata.append(entry)
        self._save_metadata()

        logger.info("Saved motion '%s' (%d frames)", motion_id, keypoints.shape[0])

    def load_motion(self, motion_id: str) -> dict:
        """Load a motion sequence by ID.
        
        Returns:
            dict with 'keypoints', 'scores', 'embedding', and metadata.
        """
        with h5py.File(self.h5_path, "r") as f:
            if motion_id not in f:
                raise KeyError(f"Motion '{motion_id}' not found in database")
            grp = f[motion_id]
            data = {
                "keypoints": grp["keypoints"][:],
                "scores": grp["scores"][:],
                "embedding": grp["embedding"][:],
            }

        # Find metadata
        for entry in self._metadata:
            if entry["motion_id"] == motion_id:
                data["metadata"] = entry
                break

        return data

    def search_similar(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[dict]:
        """Find motions similar to query embedding.
        
        Args:
            query_embedding: (D,) query vector.
            top_k: Number of results.
            
        Returns:
            List of dicts with 'motion_id', 'score', and metadata.
        """
        index = self._get_faiss_index()
        if index.ntotal == 0:
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = index.search(query, min(top_k, index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if idx < len(self._metadata):
                result = dict(self._metadata[idx])
                result["similarity_score"] = float(score)
                results.append(result)

        return results

    def list_motions(self) -> list[dict]:
        """List all stored motions."""
        return list(self._metadata)

    def delete_motion(self, motion_id: str):
        """Remove a motion from the database (metadata and HDF5 only, FAISS rebuild needed)."""
        with h5py.File(self.h5_path, "a") as f:
            if motion_id in f:
                del f[motion_id]
        self._metadata = [m for m in self._metadata if m["motion_id"] != motion_id]
        self._save_metadata()
        logger.info("Deleted motion '%s'", motion_id)

    def rebuild_index(self):
        """Rebuild the FAISS index from stored embeddings."""
        import faiss

        embeddings = []
        with h5py.File(self.h5_path, "r") as f:
            for entry in self._metadata:
                mid = entry["motion_id"]
                if mid in f:
                    embeddings.append(f[mid]["embedding"][:])

        if not embeddings:
            self._index = faiss.IndexFlatIP(self.embedding_dim)
            self._save_faiss_index()
            return

        emb_matrix = np.stack(embeddings).astype(np.float32)

        if self.index_type == "Flat":
            self._index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            nlist = min(self.nlist, len(embeddings))
            self._index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self._index.train(emb_matrix)

        self._index.add(emb_matrix)
        self._save_faiss_index()

        # Update index positions
        for i, entry in enumerate(self._metadata):
            entry["index_position"] = i
        self._save_metadata()

        logger.info("Rebuilt FAISS index with %d vectors", self._index.ntotal)

    def _save_faiss_index(self):
        import faiss
        if self._index is not None:
            faiss.write_index(self._index, str(self.index_path))

    # ── JSON Export / Import for Motion Data ──

    def export_to_json(
        self,
        motion_id: str,
        output_path: str | None = None,
        frame_range: tuple[int, int] | None = None,
        keypoint_groups: list[str] | None = None,
    ) -> str:
        """Export a motion to JSON format for sharing/inspection.
        
        Args:
            motion_id: Motion to export.
            output_path: Output file path. Auto-generated if None.
            frame_range: Optional (start, end) frame slice.
            keypoint_groups: Optional filter e.g. ['body', 'hands', 'face'].
            
        Returns:
            Path to the exported JSON file.
        """
        data = self.load_motion(motion_id)
        kps = data["keypoints"]    # (T, K, 2)
        scores = data["scores"]    # (T, K)

        # Apply frame range
        if frame_range:
            start, end = frame_range
            kps = kps[start:end]
            scores = scores[start:end]

        # Apply keypoint group filter
        if keypoint_groups:
            from ..pose_extraction.utils import JOINT_GROUPS
            indices = []
            group_map = {
                "body": list(range(0, 17)),
                "feet": list(range(17, 23)),
                "face": list(range(23, 91)),
                "left_hand": list(range(91, 112)),
                "right_hand": list(range(112, 133)),
                "hands": list(range(91, 133)),
            }
            for g in keypoint_groups:
                if g in group_map:
                    indices.extend(group_map[g])
                elif g in JOINT_GROUPS:
                    indices.extend(JOINT_GROUPS[g])
            indices = sorted(set(indices))
            kps = kps[:, indices]
            scores = scores[:, indices]

        # Build JSON structure
        export = {
            "motion_id": motion_id,
            "format": "dance_motion_v1",
            "n_frames": int(kps.shape[0]),
            "n_keypoints": int(kps.shape[1]),
            "frame_range": list(frame_range) if frame_range else [0, int(kps.shape[0])],
            "keypoint_groups": keypoint_groups or ["all"],
            "metadata": data.get("metadata", {}),
            "frames": [],
        }

        for t in range(kps.shape[0]):
            frame_data = {
                "frame": t + (frame_range[0] if frame_range else 0),
                "keypoints": kps[t].tolist(),
                "scores": scores[t].tolist(),
            }
            export["frames"].append(frame_data)

        if output_path is None:
            output_path = str(self.db_path / f"{motion_id}.json")

        with open(output_path, "w") as f:
            json.dump(export, f, indent=2)

        logger.info("Exported motion '%s' to %s (%d frames)", motion_id, output_path, kps.shape[0])
        return output_path

    def import_from_json(self, json_path: str, motion_id: str | None = None) -> str:
        """Import a motion from JSON format.
        
        Args:
            json_path: Path to JSON file.
            motion_id: Override motion ID. Uses file's ID if None.
            
        Returns:
            The motion_id of the imported motion.
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        mid = motion_id or data.get("motion_id", Path(json_path).stem)
        frames = data["frames"]

        kps = np.array([f["keypoints"] for f in frames], dtype=np.float32)
        scores = np.array([f["scores"] for f in frames], dtype=np.float32)

        # Generate embedding
        from .embeddings import MotionEmbedder
        embedder = MotionEmbedder({"motion_storage": {"embedding_dim": self.embedding_dim}})
        embedding = embedder.embed(kps)

        metadata = data.get("metadata", {})
        metadata["imported_from"] = json_path

        self.save_motion(mid, kps, scores, embedding, metadata)
        return mid

    def get_segment(
        self,
        motion_id: str,
        start_frame: int,
        end_frame: int,
    ) -> dict:
        """Get a segment of a motion (for stitching/mixing).
        
        Args:
            motion_id: Motion to slice.
            start_frame: Start frame (inclusive).
            end_frame: End frame (exclusive).
            
        Returns:
            dict with 'keypoints', 'scores' for the segment.
        """
        data = self.load_motion(motion_id)
        return {
            "keypoints": data["keypoints"][start_frame:end_frame],
            "scores": data["scores"][start_frame:end_frame],
            "source_motion": motion_id,
            "frame_range": (start_frame, end_frame),
            "n_frames": end_frame - start_frame,
        }

    def get_random_segments(
        self,
        n_segments: int = 4,
        segment_length: int = 60,
        motion_ids: list[str] | None = None,
    ) -> list[dict]:
        """Get random motion segments for stitching together.
        
        Args:
            n_segments: Number of segments to fetch.
            segment_length: Frames per segment.
            motion_ids: Specific motions to sample from. All if None.
            
        Returns:
            List of segment dicts.
        """
        available = motion_ids or [m["motion_id"] for m in self._metadata]
        if not available:
            return []

        rng = np.random.default_rng()
        segments = []

        for _ in range(n_segments):
            mid = rng.choice(available)
            data = self.load_motion(mid)
            total = data["keypoints"].shape[0]
            if total <= segment_length:
                start = 0
                end = total
            else:
                start = int(rng.integers(0, total - segment_length))
                end = start + segment_length

            segments.append(self.get_segment(mid, start, end))

        return segments
