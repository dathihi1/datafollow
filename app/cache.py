"""Caching utilities for dashboard data and computations."""

import hashlib
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

T = TypeVar("T")

CACHE_DIR = Path(__file__).parent.parent / "DATA" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TTL_HOURS = 24


def get_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate deterministic cache key from arguments.

    Args:
        prefix: Key prefix (usually function name)
        *args: Positional arguments to hash
        **kwargs: Keyword arguments to hash

    Returns:
        Deterministic hash string
    """
    def make_hashable(obj: Any) -> Any:
        """Convert objects to hashable representations."""
        if isinstance(obj, np.ndarray):
            return ("ndarray", obj.tobytes().hex()[:32], obj.shape)
        if isinstance(obj, (list, tuple)):
            return tuple(make_hashable(x) for x in obj)
        if isinstance(obj, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        return obj

    hashable_args = make_hashable(args)
    hashable_kwargs = make_hashable(kwargs)

    key_data = json.dumps(
        {"args": hashable_args, "kwargs": hashable_kwargs},
        sort_keys=True,
        default=str,
    )
    return f"{prefix}_{hashlib.md5(key_data.encode()).hexdigest()}"


def get_data_hash(data: np.ndarray) -> str:
    """Get SHA256 hash of numpy array data.

    Args:
        data: Numpy array

    Returns:
        Hex digest of SHA256 hash
    """
    return hashlib.sha256(data.tobytes()).hexdigest()


def disk_cache(ttl_hours: int = DEFAULT_TTL_HOURS):
    """Decorator for disk-based caching of expensive computations.

    Args:
        ttl_hours: Time-to-live in hours

    Returns:
        Decorated function with disk caching
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            cache_key = get_cache_key(func.__name__, *args, **kwargs)
            cache_file = CACHE_DIR / f"{cache_key}.pkl"
            meta_file = CACHE_DIR / f"{cache_key}.meta.json"

            # Check if cached and not expired
            if cache_file.exists() and meta_file.exists():
                try:
                    with open(meta_file, encoding="utf-8") as f:
                        meta = json.load(f)
                    cached_at = datetime.fromisoformat(meta["cached_at"])
                    if datetime.now() - cached_at < timedelta(hours=ttl_hours):
                        with open(cache_file, "rb") as f:
                            return pickle.load(f)
                except (json.JSONDecodeError, KeyError, pickle.PickleError):
                    # Invalid cache, recompute
                    pass

            # Compute and cache
            result = func(*args, **kwargs)

            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "cached_at": datetime.now().isoformat(),
                        "function": func.__name__,
                    }, f)
            except (OSError, pickle.PickleError):
                # Failed to cache, but still return result
                pass

            return result
        return wrapper
    return decorator


def clear_cache(prefix: str | None = None) -> int:
    """Clear cache files, optionally filtered by prefix.

    Args:
        prefix: Optional prefix to filter cache files

    Returns:
        Number of cache entries cleared
    """
    count = 0
    for cache_file in CACHE_DIR.glob("*.pkl"):
        if prefix is None or cache_file.stem.startswith(prefix):
            try:
                cache_file.unlink()
                meta_file = cache_file.with_suffix(".meta.json")
                if meta_file.exists():
                    meta_file.unlink()
                count += 1
            except OSError:
                pass
    return count


def get_cache_stats() -> dict:
    """Get statistics about the cache.

    Returns:
        Dictionary with cache statistics
    """
    total_size = 0
    count = 0
    oldest = None
    newest = None

    for cache_file in CACHE_DIR.glob("*.pkl"):
        count += 1
        total_size += cache_file.stat().st_size

        meta_file = cache_file.with_suffix(".meta.json")
        if meta_file.exists():
            try:
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
                cached_at = datetime.fromisoformat(meta["cached_at"])
                if oldest is None or cached_at < oldest:
                    oldest = cached_at
                if newest is None or cached_at > newest:
                    newest = cached_at
            except (json.JSONDecodeError, KeyError):
                pass

    return {
        "count": count,
        "total_size_mb": total_size / (1024 * 1024),
        "oldest": oldest.isoformat() if oldest else None,
        "newest": newest.isoformat() if newest else None,
    }
