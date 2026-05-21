import os
import random

import grain.python as grain
import jax
import numpy as np
from numpy.lib import recfunctions as rfn
import zarr

from kinetix.environment.env_state import StaticEnvParams
from kinetix.data import bc_utils
from kinetix.data.bc_types import ActionEnvStateMask


def _prepare_shard_list(
    shard_names: list[str], seed: int, maximum_number_of_shards: int, n_val_shards: int
) -> tuple[list[str], list[str]]:
    """Shuffle, limit, and split shard names into (train, val) lists."""
    random.seed(seed)
    random.shuffle(shard_names)
    if maximum_number_of_shards > 0:
        shard_names = shard_names[:maximum_number_of_shards]
    if n_val_shards == 0:
        return shard_names, []
    assert n_val_shards < len(
        shard_names
    ), f"n_val_shards={n_val_shards} must be less than total shards={len(shard_names)}"
    return shard_names[n_val_shards:], shard_names[:n_val_shards]


def _make_grain_loader(
    data_source: grain.RandomAccessDataSource,
    shard_index: int,
    total_shard_count: int,
    seed: int,
    operations: list | None = None,
):
    """Create a shuffled, sharded grain iterator over data_source."""
    operations = operations or []
    shard_options = grain.ShardOptions(shard_index=shard_index, shard_count=total_shard_count, drop_remainder=True)
    sampler = grain.IndexSampler(num_records=len(data_source), shard_options=shard_options, shuffle=True, seed=seed)
    return iter(grain.DataLoader(data_source=data_source, operations=operations, sampler=sampler, worker_count=0))


class PostBatchWrapper(grain.MapTransform):
    def __init__(self, static_env_params: StaticEnvParams):
        super().__init__()
        self.static_env_params = static_env_params

    def map(self, data: ActionEnvStateMask) -> ActionEnvStateMask:
        expanded_state = bc_utils.expand_env_state_numpy_batched(data.env_state, self.static_env_params)
        return data._replace(env_state=expanded_state)


class _ZarrDataSourceBase(grain.RandomAccessDataSource):
    """Base random-access data source over a zarr group of array shards.

    Args:
        dataset_dir: Path to a zarr group; each array key is treated as one shard.
        batch_size: Records per item returned by __getitem__.
        seed: RNG seed for shard shuffling.
        maximum_number_of_shards: Cap on shards; -1 uses all.
        n_val_shards: Shards reserved for validation (not used in __getitem__).
        val_batch_size: Max records returned by load_validation_batch; None loads all.
    """

    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        seed: int,
        maximum_number_of_shards: int = -1,
        n_val_shards: int = 0,
        val_batch_size: int | None = None,
    ):
        self._val_batch_size = val_batch_size
        assert os.path.exists(dataset_dir), f"{dataset_dir} does not exist"
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.root = zarr.open_group(dataset_dir, mode="r")
        self.all_shard_names = sorted(self.root.array_keys())
        assert self.all_shard_names, f"No shards found in {dataset_dir}"
        # Filter out empty shards so we don't crash the sampler or load 0 records
        shard_names = [name for name in self.all_shard_names if self.root[name].shape[0] > 0]
        self.shard_names, self._val_shard_names = _prepare_shard_list(
            shard_names, seed, maximum_number_of_shards, n_val_shards
        )
        self.shards = [self.root[name] for name in self.shard_names]
        self.shard_lengths = [s.shape[0] for s in self.shards]
        self.cumulative_lengths = np.cumsum([0] + self.shard_lengths)
        self.total_batches = self.cumulative_lengths[-1] // batch_size

    def _convert(self, _) -> ActionEnvStateMask:
        raise NotImplementedError

    def load_validation_batch(self) -> ActionEnvStateMask:
        assert self._val_shard_names, "No val shards reserved (pass n_val_shards > 0)"
        remaining = self._val_batch_size
        parts = []
        for name in self._val_shard_names:
            parts.append(self.root[name][:remaining])
            if remaining is not None:
                remaining -= parts[-1].shape[0]
                if remaining <= 0:
                    break
        return self._convert(np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0])

    def _get_range(self, start_idx, length):
        shard_idx = np.searchsorted(self.cumulative_lengths, start_idx, side="right") - 1
        local_start = start_idx - self.cumulative_lengths[shard_idx]
        shard_len = self.shard_lengths[shard_idx]
        if local_start + length <= shard_len:
            return self.shards[shard_idx][local_start : local_start + length]
        part1_len = shard_len - local_start
        part1 = self.shards[shard_idx][local_start:]
        part2 = self._get_range(start_idx + part1_len, length - part1_len)
        return np.concatenate([part1, part2], axis=0)

    def __len__(self):
        return self.total_batches

    def __getitem__(self, batch_idx: int) -> ActionEnvStateMask:
        return self._convert(self._get_range(batch_idx * self.batch_size, self.batch_size))


class ZarrBatchDataSource(_ZarrDataSourceBase):
    def _convert(self, raw) -> ActionEnvStateMask:
        if "level_rngs" in raw.dtype.names:
            raw = rfn.drop_fields(raw, "level_rngs")
        return bc_utils.map_raw_dict_to_action_env_state(raw)


class MultiFileBatchDataSource(grain.RandomAccessDataSource):
    """Combines multiple ZarrBatchDataSource instances into one, mixing per-batch by proportion.

    Args:
        zarr_filepaths: Paths to zarr groups, one per dataset.
        proportions: Fraction of each batch drawn from the corresponding dataset; must sum to 1
            and `int(batch_size * p)` values must sum exactly to batch_size.
        batch_size: Total transitions per batch.
        seed: Base RNG seed; each sub-source gets a deterministically derived seed.
        **kwargs: Forwarded to ZarrBatchDataSource (e.g. maximum_number_of_shards).
    """

    def __init__(
        self,
        zarr_filepaths: list[str],
        proportions: list[float],
        batch_size: int,
        seed: int,
        **kwargs,
    ):
        assert len(zarr_filepaths) == len(proportions)
        self.num_elems_from_each = [int(batch_size * p) for p in proportions]
        assert sum(self.num_elems_from_each) == batch_size
        self.data_sources = [
            ZarrBatchDataSource(fpath, num_elems, seed * 1000 + i, **kwargs)
            for i, (fpath, num_elems) in enumerate(zip(zarr_filepaths, self.num_elems_from_each))
        ]
        self.all_shard_names = self.shard_names = []

    def __len__(self):
        return sum([len(s) for s in self.data_sources]) // len(self.data_sources)

    def __getitem__(self, batch_index: int) -> ActionEnvStateMask:
        # Multiply by a large prime so each source gets a different effective index,
        # avoiding correlated draws when all sources have similar lengths.
        parts = [source[(batch_index * 15485863) % len(source)] for source in self.data_sources]
        return jax.tree.map(lambda *x: np.concatenate(x, axis=0), *parts)


class ZarrTrajDataSource(_ZarrDataSourceBase):
    def _convert(self, raw) -> ActionEnvStateMask:
        flat = {field: raw[field].squeeze(2) for field in raw.dtype.names}
        flat["mask"] = np.ones(flat["action"].shape[:2], dtype=bool)
        assert "done" in flat, (
            f"Dataset at {self.dataset_dir} is missing the 'done' field — " "was it converted with the correct script?"
        )
        done = flat.pop("done")
        return bc_utils.map_raw_dict_to_action_env_state(flat)._replace(done=done)


class ShuffledDatasetManager:
    """Shuffled batch dataset manager backed by one or more zarr dataset directories.

    Yields randomly shuffled batches of flattened (non-trajectory) transitions.
    When the underlying iterator is exhausted it re-initialises with a new seed so
    training can run for arbitrarily many steps.

    Warning: We have not released data that works with this DatasetManager yet.
    """

    def __init__(
        self,
        dataset_dir: str | list[str],
        batch_size: int,
        static_env_params: StaticEnvParams | None = None,
        seed: int = 42,
        maximum_number_of_shards: int = -1,
        dataset_proportions: list[float] = [1.0],
        shard_index: int = 0,
        total_shard_count: int = 1,
        should_expand_static_env_params: bool = False,
        n_val_shards: int = 0,
        val_batch_size: int | None = None,
    ):
        """
        Args:
            dataset_dir: Path (or list of paths) to a zarr group whose array keys are shards.
            batch_size: Number of transitions per batch.
            static_env_params: Required when should_expand_static_env_params=True.
            seed: RNG seed for shuffling and shard ordering.
            maximum_number_of_shards: Cap on shards to load; -1 uses all available.
            dataset_proportions: Per-path fraction of each batch (multi-dir only); must sum to 1.
            shard_index: This process's index for data-parallel sharding across workers.
            total_shard_count: Total number of parallel workers sharing the data.
            should_expand_static_env_params: If True, expand env state fields via static_env_params after batching.
            n_val_shards: Number of shards to hold out as a validation set (single dataset_dir only).
            val_batch_size: Max transitions to load from val shards; None loads all.
        """
        assert (
            not should_expand_static_env_params or static_env_params is not None
        ), "static_env_params is required when should_expand_static_env_params=True"
        self.static_env_params = static_env_params
        self.should_expand_static_env_params = should_expand_static_env_params
        if type(dataset_dir) == list and len(dataset_dir) == 1:
            dataset_dir = dataset_dir[0]
        if type(dataset_proportions) != list:
            dataset_proportions = [1 / len(dataset_dir) for _ in range(len(dataset_dir))]
        self.dataset_proportions = dataset_proportions
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.maximum_number_of_shards = maximum_number_of_shards
        self.shard_index = shard_index
        self.total_shard_count = total_shard_count
        self._n_val_shards_to_reserve = n_val_shards  # only used on first init
        self._val_batch_size = val_batch_size
        self._val_reserved = False
        self._init_loader(dataset_dir, batch_size, seed)
        self.seed = seed

        # Load validation batch from the reserved shards (single-file only).
        self.validation_batch: ActionEnvStateMask | None = None
        if n_val_shards > 0 and isinstance(self.data_source, ZarrBatchDataSource):
            self.validation_batch = self.data_source.load_validation_batch()
            if should_expand_static_env_params:
                expanded = bc_utils.expand_env_state_numpy_batched(self.validation_batch.env_state, static_env_params)
                self.validation_batch = self.validation_batch._replace(env_state=expanded)
        self._val_reserved = True

    def _init_loader(self, dataset_dir: str | list[str], batch_size: int, zarr_seed):
        n_val = self._n_val_shards_to_reserve if not self._val_reserved else 0
        if type(dataset_dir) == list:
            data_source = MultiFileBatchDataSource(
                dataset_dir,
                self.dataset_proportions,
                batch_size,
                seed=zarr_seed,
                maximum_number_of_shards=self.maximum_number_of_shards,
            )
        else:
            data_source = ZarrBatchDataSource(
                dataset_dir,
                batch_size,
                seed=zarr_seed,
                maximum_number_of_shards=self.maximum_number_of_shards,
                n_val_shards=n_val,
                val_batch_size=self._val_batch_size if not self._val_reserved else None,
            )

        self.data_source = data_source
        operations = [PostBatchWrapper(self.static_env_params)] if self.should_expand_static_env_params else []
        self.iterator = _make_grain_loader(data_source, self.shard_index, self.total_shard_count, zarr_seed, operations)
        self.length = len(data_source)

        self.shard_names = data_source.shard_names
        self.all_shard_names = data_source.all_shard_names

    def load_next_batch(self) -> ActionEnvStateMask:
        try:
            return next(self.iterator)
        except StopIteration:
            self.seed += 1
            self._init_loader(self.dataset_dir, self.batch_size, zarr_seed=self.seed)
            return self.load_next_batch()


class TrajectoryDatasetManager:
    """
    Loads trajectory zarr shards and returns whole trajectories to preserve temporal structure.
    Each batch has shape (batch_size, T, *dims); batch_size is in trajectories, not timesteps.
    Interface matches ShuffledDatasetManager.
    """

    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        static_env_params: StaticEnvParams | None = None,
        seed: int = 42,
        maximum_number_of_shards: int = -1,
        n_val_shards: int = 0,
        val_batch_size: int | None = None,
        shard_index: int = 0,
        total_shard_count: int = 1,
        should_expand_static_env_params: bool = False,
    ):
        """
        Args:
            dataset_dir: Path to a zarr group whose array keys are trajectory shards.
            batch_size: Number of trajectories per batch (not timesteps).
            static_env_params: Required when should_expand_static_env_params=True.
            seed: RNG seed for shuffling and shard ordering.
            maximum_number_of_shards: Cap on shards to load; -1 uses all available.
            n_val_shards: Number of shards to hold out as a validation set.
            val_batch_size: Max trajectories to load from val shards; None loads all.
            shard_index: This process's index for data-parallel sharding across workers.
            total_shard_count: Total number of parallel workers sharing the data.
            should_expand_static_env_params: If True, expand env state fields via static_env_params after batching.
        """
        assert (
            not should_expand_static_env_params or static_env_params is not None
        ), "static_env_params is required when should_expand_static_env_params=True"
        self.static_env_params = static_env_params
        self.should_expand_static_env_params = should_expand_static_env_params
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.maximum_number_of_shards = maximum_number_of_shards
        self.shard_index = shard_index
        self.total_shard_count = total_shard_count
        self._n_val_shards = n_val_shards
        self._val_batch_size = val_batch_size
        self.seed = seed

        data_source = ZarrTrajDataSource(
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            seed=seed,
            maximum_number_of_shards=maximum_number_of_shards,
            n_val_shards=n_val_shards,
            val_batch_size=val_batch_size,
        )
        self.shard_names = data_source.shard_names
        self.all_shard_names = data_source.all_shard_names
        self.length = len(data_source)

        self.validation_batch: ActionEnvStateMask | None = None
        if n_val_shards > 0:
            self.validation_batch = data_source.load_validation_batch()
            if should_expand_static_env_params:
                expanded = bc_utils.expand_env_state_numpy_batched(self.validation_batch.env_state, static_env_params)
                self.validation_batch = self.validation_batch._replace(env_state=expanded)

        self._init_loader(data_source)

    def _init_loader(self, data_source: ZarrTrajDataSource) -> None:
        operations = [PostBatchWrapper(self.static_env_params)] if self.should_expand_static_env_params else []
        self.iterator = _make_grain_loader(data_source, self.shard_index, self.total_shard_count, self.seed, operations)

    def load_next_batch(self) -> ActionEnvStateMask:
        try:
            return next(self.iterator)
        except StopIteration:
            self.seed += 1
            data_source = ZarrTrajDataSource(
                dataset_dir=self.dataset_dir,
                batch_size=self.batch_size,
                seed=self.seed,
                maximum_number_of_shards=self.maximum_number_of_shards,
            )
            self._init_loader(data_source)
            return self.load_next_batch()
