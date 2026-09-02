import json
import sqlite3
from pathlib import Path
import os
import io
from collections import defaultdict

import torch


class Cache:
    def __init__(self, path: str, fingerprint: str, shard_size_gb=1,
                 keep_on_fingerprint_change=False, identity=None, content_digest=None):
        self.keep_on_fingerprint_change = keep_on_fingerprint_change
        # What produced this cache's contents -- the VAE for latents, the text encoder
        # for embeddings. None means the caller supplies no identity, which is the
        # behaviour every model had before this existed.
        self.identity = identity
        # Digest of the INPUT these contents were computed from, for caches whose input is
        # cheap to hash. identity answers "who produced this"; this answers "from what".
        # keep_on_fingerprint_change deliberately tolerates a moved fingerprint, and for the
        # text embedding cache the caption text is essentially the whole fingerprint -- so
        # without this, keeping is exactly the same thing as ignoring an edit to the captions.
        self.content_digest = content_digest
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.metadata_db = self.path / 'metadata.db'
        self.shard_size_gb = shard_size_gb
        os.makedirs(self.path, exist_ok=True)

        self.init()


    def __len__(self):
        return len(self.items)


    def __getitem__(self, idx):
        assert isinstance(idx, int)
        shard_id, shard_index = self.items[idx]
        offset, size = self.shard_metadata[shard_id][shard_index]
        if shard_id not in self.open_files:
            self.open_files[shard_id] = open(self.path / f'shard_{shard_id}.bin', 'rb')
        f = self.open_files[shard_id]
        f.seek(offset)
        byte_string = f.read(size)
        buffer = io.BytesIO(byte_string)
        item = torch.load(buffer, map_location='cpu')
        return item


    def init(self):
        print('[CACHE] Initializing')
        # create database
        self.con = sqlite3.connect(self.metadata_db, autocommit=False)

        # check fingerprint, clear cache if different
        self.con.execute('CREATE TABLE IF NOT EXISTS fingerprint(value)')
        existing_fingerprint = self.con.execute('SELECT value FROM fingerprint').fetchone()
        if existing_fingerprint is not None:
            existing_fingerprint = existing_fingerprint[0]
            print(f'[CACHE] Existing cache has fingerprint {existing_fingerprint}')
            compatible, reason = self.check_identity()
            if not compatible:
                # An incompatible cache is rebuilt no matter what keep_* says. keep_* exists to
                # avoid recaching UNNECESSARILY, not to reuse contents produced by something
                # else -- reusing latents from a different VAE trains on silently wrong data.
                print(f'[CACHE] {reason} Rebuilding.')
                self.clear()
                return
            if self.fingerprint != existing_fingerprint:
                if self.keep_on_fingerprint_change:
                    # Compatible, so the contents are still what this model produces; only the
                    # fingerprint moved, which is what a caption setting change does.
                    print(
                        f'[CACHE] Fingerprint changed ({existing_fingerprint} -> '
                        f'{self.fingerprint}) but the cache is compatible with this run and '
                        'keep is set, so the existing files are reused.'
                    )
                else:
                    print('[CACHE] Fingerprint changed, deleting existing cache files')
                    self.clear()
                    return
        else:
            print(f'[CACHE] Storing new fingerprint: {self.fingerprint}')
            self.con.execute('INSERT INTO fingerprint VALUES(?)', (self.fingerprint,))

        # items table, current length, next shard index
        self.con.execute('CREATE TABLE IF NOT EXISTS items(shard, shard_index)')
        self.items = self.con.execute('SELECT shard, shard_index FROM items').fetchall() or []
        max_existing_shard = -1
        for shard, _ in self.items:
            max_existing_shard = max(max_existing_shard, shard)
        self.shard = max_existing_shard + 1  # current shard to write to
        self.shard_file = None
        print(f'[CACHE] Existing cache length: {len(self)}')

        # shard metadata
        self.shard_metadata = defaultdict(list)
        for table_name, in self.con.execute('SELECT name FROM sqlite_master').fetchall():
            if table_name.startswith('shard_'):
                shard_id = int(table_name.split('_')[-1])
                for entry in self.con.execute(f'SELECT offset, size FROM {table_name}').fetchall():
                    self.shard_metadata[shard_id].append(entry)
        self.open_files = {}

        # commit
        self.con.commit()


    @property
    def manifest_file(self):
        return self.path / 'cache_manifest.json'

    def check_identity(self):
        """Is this cache's content compatible with what the caller is about to ask for?

        Returns (compatible, reason). Compatible in three cases: the caller supplies no
        identity, no manifest exists, or the manifest agrees. The middle case is what keeps
        every cache written before this existed valid -- nothing is known about it, so nothing
        is claimed, and it gets a manifest the next time it is written in full.
        """
        if not self.identity:
            return True, ''
        if not self.manifest_file.exists():
            return True, ''
        try:
            with open(self.manifest_file, encoding='utf-8') as f:
                recorded = json.load(f).get('identity', None)
        except (OSError, ValueError):
            # An unreadable manifest says nothing either way; do not destroy a cache over it.
            return True, ''
        if recorded is None or recorded == self.identity:
            return True, ''
        return False, (
            f'This cache was built by {recorded!r} but this run is {self.identity!r}, so its '
            'contents do not belong to this model.'
        )

    def recorded_content_digest(self):
        """The content digest this cache was last completed with, or None if unknown.

        None covers a cache written before this existed as well as one with no manifest at
        all. Both mean nothing is claimed about the input, so the caller must not treat a
        mismatch it cannot see as proof of anything.
        """
        if not self.manifest_file.exists():
            return None
        try:
            with open(self.manifest_file, encoding='utf-8') as f:
                return json.load(f).get('content_digest', None)
        except (OSError, ValueError):
            return None

    def write_manifest(self):
        """Record what produced this cache, and from what. Called once contents are complete."""
        if not self.identity and not self.content_digest:
            return
        os.makedirs(self.path, exist_ok=True)
        record = {'schema': 2}
        if self.identity:
            record['identity'] = self.identity
        if self.content_digest:
            record['content_digest'] = self.content_digest
        tmp = self.manifest_file.with_name(self.manifest_file.name + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(record, f, sort_keys=True)
        os.replace(tmp, self.manifest_file)

    def clear(self):
        '''Deletes all cache files from disk. Calls init() again.'''
        self.con.close()
        os.remove(self.metadata_db)
        for bin_path in self.path.glob('*.bin'):
            os.remove(bin_path)
        # The manifest describes contents that no longer exist.
        self.manifest_file.unlink(missing_ok=True)
        self.init()


    def create_new_shard(self):
        self.shard_file = open(self.path / f'shard_{self.shard}.bin', 'wb')
        self.shard_table = f'shard_{self.shard}'
        print(f'[CACHE] Creating new shard: {self.shard_table}')
        self.con.execute(f'CREATE TABLE {self.shard_table}(offset, size)')
        self.shard_index = 0
        self.offset = 0


    def finalize_current_shard(self):
        if self.shard_file is None:
            # no-op if already finalized
            return
        self.shard_file.close()
        self.shard_file = None
        self.shard += 1
        self.con.commit()


    def add(self, item):
        if self.shard_file is None:
            self.create_new_shard()
        buffer = io.BytesIO()
        torch.save(item, buffer)
        bytes_view = buffer.getbuffer()
        self.shard_file.write(bytes_view)

        # update items metadata
        item = (self.shard, self.shard_index)
        self.items.append(item)
        self.con.execute('INSERT INTO items VALUES(?, ?)', item)
        self.shard_index += 1

        # update shard metadata
        size = len(bytes_view)
        entry = (self.offset, size)
        self.shard_metadata[self.shard].append(entry)
        self.con.execute(f'INSERT INTO {self.shard_table} VALUES (?, ?)', entry)
        self.offset += size

        # create new shard when existing one is large enough
        current_size_gb = self.shard_file.tell() / 1_000_000_000
        if current_size_gb >= self.shard_size_gb:
            self.finalize_current_shard()


# for testing
if __name__ == '__main__':
    cache = Cache('/home/anon/tmp/cache_test', 'foo', shard_size_gb=0.001)

    tensor = torch.zeros((100_000,))
    for _ in range(10):
        cache.add({'key1': tensor})
    cache.finalize_current_shard()

    print(cache[0])
    print(cache[1])
