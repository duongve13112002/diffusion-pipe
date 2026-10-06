"""Immutable tensor references for caption/dataset configuration profiles."""

import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sqlite3

import datasets
from datasets.fingerprint import Hasher
import numpy as np
import torch


logger = logging.getLogger(__name__)


def row_key(row, columns):
    payload = {column: row[column] for column in columns}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def align_legacy_metadata(metadata, iteration_path, path):
    """Recover the actual legacy tensor order from its saved iteration mapping, not a seed guess."""
    iteration = datasets.load_from_disk(str(iteration_path))
    path = Path(path) / Hasher.hash([metadata._fingerprint, iteration._fingerprint])
    if path.joinpath('aligned').exists():
        return datasets.load_from_disk(str(path / 'aligned'))
    path.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path / 'order.sqlite') as lookup:
        lookup.execute('CREATE TABLE IF NOT EXISTS rows (key TEXT PRIMARY KEY, idx INTEGER)')
        lookup.execute('DELETE FROM rows')
        for row in iteration.select_columns(['image_spec', 'latents_idx']):
            key = json.dumps(row['image_spec'])
            existing = lookup.execute('SELECT idx FROM rows WHERE key=?', (key,)).fetchone()
            if existing is not None and existing[0] != row['latents_idx']:
                raise ValueError('Ambiguous legacy latent ordering')
            lookup.execute('INSERT OR IGNORE INTO rows VALUES (?, ?)', (key, row['latents_idx']))
        lookup.commit()

        def original_index(row):
            match = lookup.execute('SELECT idx FROM rows WHERE key=?', (json.dumps(row['image_spec']),)).fetchone()
            if match is None:
                raise ValueError('Legacy iteration order does not cover its metadata')
            return {'_legacy_index': match[0]}

        ordered = metadata.map(original_index, keep_in_memory=True,
                               new_fingerprint=Hasher.hash([metadata._fingerprint, iteration._fingerprint]))
        ordered = ordered.sort('_legacy_index', keep_in_memory=True)
        if any(value != index for index, value in enumerate(ordered['_legacy_index'])):
            raise ValueError('Legacy latent indices are not a complete contiguous mapping')
        ordered = ordered.remove_columns('_legacy_index')
        ordered.save_to_disk(str(path / 'aligned'))
    return datasets.load_from_disk(str(path / 'aligned'))


class ReadOnlyCache:
    """Read the legacy SQLite/shard format without ever clearing or claiming it."""

    def __init__(self, path, identity=None, encoder_key=None):
        self.path = Path(path)
        self.con = sqlite3.connect(self.path.joinpath('metadata.db').resolve().as_uri() + '?mode=ro', uri=True)
        self.fingerprint = self.con.execute('SELECT value FROM fingerprint').fetchone()[0]
        self.count = self.con.execute('SELECT COUNT(*) FROM items').fetchone()[0]
        self.identity = None
        self.content_digest = None
        self.encoder_key = ''
        manifest = self.path / 'cache_manifest.json'
        if manifest.exists():
            record = json.loads(manifest.read_text(encoding='utf-8'))
            self.identity = record.get('identity')
            self.content_digest = record.get('content_digest')
            self.encoder_key = record.get('encoder_key', '')
        if (identity and identity != self.identity) or (encoder_key is not None and encoder_key != self.encoder_key):
            self.con.close()
            raise ValueError(f'Incompatible encoder identity in {self.path}')
        self.open_files = {}
        self.pid = os.getpid()
        # Filled by load_index(); until then every read looks its location up in SQLite.
        self._shard = self._offset = self._size = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(con=None, open_files={}, pid=None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_process(self):
        # DataLoader workers must not share SQLite objects or seek offsets inherited by fork.
        if self.pid != os.getpid():
            self.close()
            self.pid = os.getpid()

    def _ensure_reader(self):
        self._ensure_process()
        if self.con is None:
            self.con = sqlite3.connect(self.path.joinpath('metadata.db').resolve().as_uri() + '?mode=ro', uri=True)

    def load_index(self):
        """Read the item and shard tables into arrays, as the legacy Cache keeps them in RAM.

        Otherwise every sample costs two random B-tree lookups. That is cheap on a local disk
        but means page reads and lock round trips on network storage, which made training
        reads much slower than the legacy cache. Loaded in the parent before DataLoader
        workers fork, the arrays are shared copy-on-write. About 20 bytes per item.
        """
        if self._offset is not None:
            return
        self._ensure_reader()
        shard = np.full(self.count, -1, dtype=np.int64)
        shard_index = np.empty(self.count, dtype=np.int64)
        cursor = self.con.execute('SELECT rowid, shard, shard_index FROM items')
        while rows := cursor.fetchmany(1 << 20):
            rows = np.asarray(rows, dtype=np.int64)
            position = rows[:, 0] - 1
            if position.min() < 0 or position.max() >= self.count:
                raise ValueError(f'Non-contiguous item rows in {self.path}')
            shard[position] = rows[:, 1]
            shard_index[position] = rows[:, 2]
        if (shard < 0).any():
            raise ValueError(f'Non-contiguous item rows in {self.path}')
        offset = np.empty(self.count, dtype=np.int64)
        size = np.empty(self.count, dtype=np.int64)
        for shard_id in np.unique(shard).tolist():
            table = np.asarray(self.con.execute(
                f'SELECT offset, size FROM shard_{int(shard_id)} ORDER BY rowid').fetchall(), dtype=np.int64)
            selected = shard == shard_id
            offset[selected] = table[shard_index[selected], 0]
            size[selected] = table[shard_index[selected], 1]
        self._shard, self._offset, self._size = shard.astype(np.int32), offset, size
        self.close()  # Not needed any more; workers never reopen it.

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        if self._offset is not None:
            self._ensure_process()
            shard, offset, size = int(self._shard[index]), int(self._offset[index]), int(self._size[index])
        else:
            self._ensure_reader()
            shard, shard_index = self.con.execute(
                'SELECT shard, shard_index FROM items WHERE rowid=?', (int(index) + 1,)).fetchone()
            offset, size = self.con.execute(
                f'SELECT offset, size FROM shard_{int(shard)} WHERE rowid=?', (int(shard_index) + 1,)).fetchone()
        if shard not in self.open_files:
            self.open_files[shard] = open(self.path / f'shard_{shard}.bin', 'rb')
        handle = self.open_files[shard]
        handle.seek(offset)
        return torch.load(io.BytesIO(handle.read(size)), map_location='cpu')

    def close(self):
        for handle in self.open_files.values():
            handle.close()
        self.open_files.clear()
        if self.con is not None:
            self.con.close()
            self.con = None


class IndexedCache:
    def __init__(self, references, sources):
        self.references = references
        self.sources = sources
        # Arrays, not per-sample Arrow row reads: see ReadOnlyCache.load_index.
        if len(references):
            self._source = references.data.column('source').to_numpy().astype(np.int32)
            self._index = references.data.column('index').to_numpy().astype(np.int64)
        else:
            self._source = np.empty(0, dtype=np.int32)
            self._index = np.empty(0, dtype=np.int64)
        # Only sources something points at; unmatched donors never pay for an index.
        for source in np.unique(self._source).tolist():
            if hasattr(sources[source], 'load_index'):
                sources[source].load_index()

    def __len__(self):
        return len(self._source)

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self._source)
        if not 0 <= index < len(self._source):
            raise IndexError(index)
        return self.sources[int(self._source[index])][int(self._index[index])]

    def close(self):
        for source in self.sources:
            source.close()


def _image_hash(image):
    return int.from_bytes(hashlib.blake2b(image.encode(), digest_size=8).digest(), 'little')


class IndexedTextEmbeddingDataset:
    """Persist image/caption lookup rather than materializing millions of Python lists."""

    def __init__(self, cache, metadata, path):
        self.te_dataset = cache
        self.flattened_captions = metadata
        path = Path(path)
        self.path = path
        self.pid = os.getpid()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lookup = sqlite3.connect(path)
        self.lookup.execute('CREATE TABLE IF NOT EXISTS captions (image TEXT, number INTEGER, idx INTEGER, '
                            'PRIMARY KEY(image, number))')
        self.lookup.execute('CREATE TABLE IF NOT EXISTS complete (fingerprint TEXT)')
        complete = self.lookup.execute('SELECT fingerprint FROM complete').fetchone()
        if complete != (metadata._fingerprint,):
            self.lookup.execute('DELETE FROM captions')
            previous = None
            number = 0
            for index, row in enumerate(metadata.select_columns('image_spec')):
                image = json.dumps(row['image_spec'])
                number = number + 1 if image == previous else 0
                self.lookup.execute('INSERT INTO captions VALUES (?, ?, ?)', (image, number, index))
                previous = image
            self.lookup.execute('DELETE FROM complete')
            self.lookup.execute('INSERT INTO complete VALUES (?)', (metadata._fingerprint,))
            self.lookup.commit()
        self._load_arrays(metadata)

    def _load_arrays(self, metadata):
        """Keep image -> first flattened row in RAM, so a sample costs no SQLite query.

        Sorted 64-bit hashes of the image key, searched with numpy: about 20 bytes per image
        instead of a Python dict. Every lookup comes from this metadata's own images, so the
        mapping is exact unless two of them share a hash; that is checked here, and the
        SQLite table stays the fallback for it. Persisted next to it, keyed by fingerprint.
        """
        self._hashes = self._starts = self._counts = None
        arrays_path = self.path.with_name(self.path.name + '.lookup.npz')
        try:
            with np.load(arrays_path) as stored:
                if str(stored['fingerprint']) == metadata._fingerprint:
                    self._hashes, self._starts, self._counts = (
                        stored['hashes'], stored['starts'], stored['counts'])
                    return
        except (OSError, KeyError, ValueError):
            pass
        hashes, starts, counts = [], [], []
        previous = None
        row = 0
        for batch in metadata.select_columns('image_spec').iter(batch_size=100_000):
            for image_spec in batch['image_spec']:
                image = json.dumps(image_spec)
                if image == previous:
                    counts[-1] += 1
                else:
                    hashes.append(_image_hash(image))
                    starts.append(row)
                    counts.append(1)
                previous = image
                row += 1
        hashes = np.asarray(hashes, dtype=np.uint64)
        order = np.argsort(hashes, kind='stable')
        hashes = hashes[order]
        if len(hashes) > 1 and (hashes[1:] == hashes[:-1]).any():
            logger.warning('%s: image key hash collision; text embedding lookups use SQLite.', self.path)
            return
        starts = np.asarray(starts, dtype=np.int64)[order]
        counts = np.asarray(counts, dtype=np.int32)[order]
        temporary = arrays_path.with_name(f'{arrays_path.name}.{os.getpid()}.tmp')
        with open(temporary, 'wb') as handle:
            np.savez(handle, fingerprint=np.asarray(metadata._fingerprint),
                     hashes=hashes, starts=starts, counts=counts)
        os.replace(temporary, arrays_path)
        self._hashes, self._starts, self._counts = hashes, starts, counts

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(lookup=None, pid=None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def get_text_embeddings(self, image_spec, caption_number):
        if self._hashes is not None:
            key = np.uint64(_image_hash(json.dumps(image_spec)))
            position = int(np.searchsorted(self._hashes, key))
            number = int(caption_number)
            if (position == len(self._hashes) or self._hashes[position] != key
                    or not 0 <= number < self._counts[position]):
                raise KeyError((image_spec, caption_number))
            return self.te_dataset[int(self._starts[position]) + number]
        if self.lookup is None or self.pid != os.getpid():
            self.close()
            self.lookup = sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True)
            self.pid = os.getpid()
        match = self.lookup.execute('SELECT idx FROM captions WHERE image=? AND number=?',
                                    (json.dumps(image_spec), int(caption_number))).fetchone()
        if match is None:
            raise KeyError((image_spec, caption_number))
        return self.te_dataset[match[0]]

    def close(self):
        if self.lookup is not None:
            self.lookup.close()
            self.lookup = None


def tensor_profile(metadata, map_fn, path, columns, identity, sources, map_cache, *,
                   caching_batch_size=1, regenerate_cache=False, encoder_key=None, lookup_dir=None):
    """Reference exact legacy inputs; encode only inputs absent from a compatible donor.

    The on-disk lookup is built once, avoiding a 30M-entry Python dictionary. Caption,
    video/bucket/mask/control context all remain in text keys; latent keys omit caption only.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    binding_file = path / 'references.json'
    if binding_file.exists() and not regenerate_cache:
        binding = json.loads(binding_file.read_text(encoding='utf-8'))
        if (binding['metadata'] == metadata._fingerprint and binding['identity'] == identity
                and binding['encoder_key'] == encoder_key):
            readers = []
            try:
                for item in binding['sources']:
                    reader = ReadOnlyCache(path / item['path'], identity, encoder_key)
                    readers.append(reader)
                    if reader.fingerprint != item['fingerprint'] or len(reader) != item['count']:
                        raise RuntimeError('A referenced tensor cache changed.')
                    if reader.content_digest != item['content_digest']:
                        raise RuntimeError('A referenced tensor input manifest changed.')
                references = datasets.load_from_disk(str(path / binding['references']))
                if len(references) != len(metadata):
                    raise RuntimeError('Incomplete tensor profile.')
                return IndexedCache(references, readers)
            except (RuntimeError, ValueError, OSError, sqlite3.Error, TypeError) as e:
                # A donor is not ours to keep stable: a reuse_metadata_cache = false run clears
                # and rebuilds the legacy cache it lives in whenever its fingerprint moves. The
                # binding is only an index, so rebuild it from donors that are re-validated
                # below, and encode whatever none of them still holds.
                for reader in readers:
                    reader.close()
                print(f'[CACHE] Tensor profile {path} is stale ({e}); rebuilding its references.')

    readers = []
    lookups = []
    try:
        if not regenerate_cache:
            for donor_metadata, donor_path in sources() if callable(sources) else sources:
                donor_path = Path(donor_path)
                if not (donor_path / 'metadata.db').exists():
                    continue
                try:
                    reader = ReadOnlyCache(donor_path, identity, encoder_key)
                except ValueError:
                    continue
                if len(reader) != len(donor_metadata) or not set(columns) <= set(donor_metadata.column_names):
                    logger.warning(
                        'Cache donor %s has %s entries for %s metadata rows or incompatible columns; '
                        'not reusing it, unmatched inputs will be regenerated.',
                        donor_path, len(reader), len(donor_metadata))
                    reader.close()
                    continue
                stamp = row_key({'path': str(donor_path.resolve()), 'metadata': donor_metadata._fingerprint,
                                 'fingerprint': reader.fingerprint, 'columns': columns,
                                 'content_digest': reader.content_digest},
                                ['path', 'metadata', 'fingerprint', 'columns', 'content_digest'])
                index_path = Path(lookup_dir or path.parent / 'lookup') / f'{stamp}.sqlite'
                index_path.parent.mkdir(parents=True, exist_ok=True)
                lookup = sqlite3.connect(index_path)
                lookup.execute('CREATE TABLE IF NOT EXISTS rows (key TEXT PRIMARY KEY, idx INTEGER)')
                lookup.execute('CREATE TABLE IF NOT EXISTS complete (count INTEGER)')
                if lookup.execute('SELECT count FROM complete').fetchone() is None:
                    content_column = 'caption' if 'caption' in columns else 'image_spec'
                    if reader.content_digest and reader.content_digest != Hasher.hash(list(donor_metadata[content_column])):
                        print(f'[CACHE] Donor content digest differs: {reader.path}; skipping.')
                        lookup.close()
                        reader.close()
                        continue
                    lookup.execute('DELETE FROM rows')
                    lookup.executemany('INSERT OR IGNORE INTO rows VALUES (?, ?)',
                                       ((row_key(row, columns), i) for i, row in enumerate(donor_metadata)))
                    lookup.execute('INSERT INTO complete VALUES (?)', (len(donor_metadata),))
                    lookup.commit()
                readers.append(reader)
                lookups.append(lookup)

        missing_indices = []
        references_token = hashlib.sha256(
            json.dumps([metadata._fingerprint, identity, encoder_key,
                        [(str(r.path), r.fingerprint, len(r), r.content_digest) for r in readers]], sort_keys=True).encode()
        ).hexdigest()[:16]
        references_path = path / f'references_{references_token}'

        def references_generator():
            missing_indices.clear()
            for index, row in enumerate(metadata):
                key = row_key(row, columns)
                found = None
                for source, lookup in enumerate(lookups):
                    match = lookup.execute('SELECT idx FROM rows WHERE key=?', (key,)).fetchone()
                    if match is not None:
                        found = {'source': source, 'index': match[0]}
                        break
                if found is None:
                    found = {'source': len(readers), 'index': len(missing_indices)}
                    missing_indices.append(index)
                yield found

        # Explicit local cache_dir is also needed for the generator's intermediate Arrow files.
        references = datasets.Dataset.from_generator(
            references_generator, cache_dir=str(path / 'arrow'),
            fingerprint=references_token + ('_regenerate' if regenerate_cache else ''))
        # A cached generator may not execute: reconstruct the missing input positions from its output.
        missing_indices = [i for i, ref in enumerate(references) if ref['source'] == len(readers)]
        if missing_indices:
            if map_fn is None:
                raise RuntimeError(f'{len(missing_indices)} inputs in {path} need an encoder; run caching first.')
            missing = metadata.select(missing_indices)
            generated = map_cache(missing, map_fn, path / f'generated_{references_token}',
                                  caching_batch_size=caching_batch_size, identity=identity,
                                  regenerate_cache=regenerate_cache)
            if not generated.path.joinpath('inputs').exists():
                missing.save_to_disk(str(generated.path / 'inputs'))
            generated.finalize_current_shard()
            if encoder_key is not None:
                manifest = generated.manifest_file
                record = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {'schema': 2}
                record['encoder_key'] = encoder_key
                temporary = manifest.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(record, sort_keys=True), encoding='utf-8')
                os.replace(temporary, manifest)
            generated.con.close()
            for handle in generated.open_files.values():
                handle.close()
            readers.append(ReadOnlyCache(generated.path, identity, encoder_key))
        if not references_path.exists():
            references.save_to_disk(str(references_path))
        binding = {'metadata': metadata._fingerprint, 'identity': identity, 'encoder_key': encoder_key,
                   'references': references_path.name,
                   'sources': [{'path': os.path.relpath(r.path, path), 'fingerprint': r.fingerprint,
                                'count': len(r), 'content_digest': r.content_digest} for r in readers]}
        temporary = binding_file.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(binding, sort_keys=True), encoding='utf-8')
        os.replace(temporary, binding_file)
        print(f'[CACHE] Profile: {len(metadata) - len(missing_indices)} reused, {len(missing_indices)} encoded.')
        return IndexedCache(references, readers)
    except Exception:
        for reader in readers:
            reader.close()
        raise
    finally:
        for lookup in lookups:
            lookup.close()
