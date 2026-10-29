"""Lossless filtering of exact evaluation boards from materialized train rows."""
import hashlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest

import pyarrow as pa
import pyarrow.parquet as pq

from tinychess.data.sources import sha256_file
from tinychess.data.storage import POSITION_SCHEMA


def prefetched(items, fetch, workers=2):
    """Ordered, bounded I/O prefetch; no unbounded executor/map queue.

    At most `workers` future payloads are queued alongside the consumer's
    current payload. Hash verification remains the fetcher's responsibility.
    """
    source = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for item in source:
            pending.append((item, pool.submit(fetch, item)))
            if len(pending) == workers:
                break
        while pending:
            item, future = pending.popleft()
            value = future.result()  # Preserve deterministic source ordering.
            following = next(source, None)
            if following is not None:
                pending.append((following, pool.submit(fetch, following)))
            yield item, value


def filtered_batches(payload, excluded):
    file = pq.ParquetFile(pa.BufferReader(payload))
    if (not file.schema_arrow.equals(POSITION_SCHEMA, check_metadata=False)
            or file.schema_arrow.metadata != POSITION_SCHEMA.metadata):
        raise ValueError('Cannot filter incompatible feature schema')
    for batch in file.iter_batches(batch_size=128):
        rows = batch.to_pylist()
        keep = []
        for row in rows:
            if row['split'] != 'train':
                raise ValueError('Physical Dataset D exclusions only apply to training')
            keep.append(row['position_hash'] not in excluded)
        if any(keep):
            yield pa.Table.from_batches([batch], schema=POSITION_SCHEMA).filter(pa.array(keep))


def write_filter(payload, excluded, path):
    count = 0
    with pq.ParquetWriter(path, POSITION_SCHEMA, compression='zstd', compression_level=3) as writer:
        for table in filtered_batches(payload, excluded):
            writer.write_table(table, row_group_size=128)
            count += len(table)
    return count, sha256_file(path)


def verify_retained_rows(payload, excluded, path):
    """Compare every field/byte against the immutable original, after filtering."""
    expected = (row for table in filtered_batches(payload, excluded) for row in table.to_pylist())
    actual = (row for batch in pq.ParquetFile(path).iter_batches(batch_size=128) for row in batch.to_pylist())
    sentinel = object()
    count = 0
    for left, right in zip_longest(expected, actual, fillvalue=sentinel):
        if left != right:
            raise ValueError('Filtered tensor/action/label changed or row missing')
        count += 1
    return count


def position_keys(fens):
    import chess
    from tinychess.validation.parity import board_identity
    return {hashlib.sha256(board_identity(chess.Board(fen)).encode()).digest() for fen in fens}
