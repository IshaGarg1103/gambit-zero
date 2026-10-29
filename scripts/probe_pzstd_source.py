"""Verify one advertised PZstandard frame with a bounded suffix request.

This is a format probe, not a random-access game sampler or full-source hash.
No new compressed/decoded game data is written to disk.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

import zstandard as zstd
from tinychess.data.archive_scan import PZSTD_MAGIC


def main(month,output):
    frozen = json.loads(Path('reports/m1_50k_preflight.json').read_text())
    source = next(s for s in frozen['sources'] if s['month']==month)
    prefix_path = Path('data/raw/m1_prefixes')/f'{month}.pgn.zst.prefix'
    prefix = prefix_path.read_bytes()
    meta = json.loads(prefix_path.with_suffix('.prefix.json').read_text())
    if hashlib.sha256(prefix).hexdigest() != meta['prefix_sha256']:
        raise RuntimeError('Cached prefix changed')
    magic,size,frame_size = struct.unpack('<III',prefix[:12])
    if magic != PZSTD_MAGIC or size != 4 or frame_size > 8*2**20:
        raise RuntimeError('Unsupported or oversized PZstandard frame')
    needed = 12+frame_size+12
    start = len(prefix)
    network_bytes = 0
    if needed > start:
        if needed-start > 8*2**20:
            raise RuntimeError('Bounded probe budget exceeded')
        request = urllib.request.Request(source['url'],headers={'Range':f'bytes={start}-{needed-1}', 'Accept-Encoding':'identity'})
        with urllib.request.urlopen(request,timeout=30) as response:
            expected_range = f'bytes {start}-{needed-1}/{source["archive_size_bytes"]}'
            if response.status != 206 or response.headers.get('Content-Range') != expected_range:
                raise RuntimeError('Unexpected bounded range response')
            if response.headers.get('ETag') != source['headers'].get('ETag'):
                raise RuntimeError('Source changed since preflight')
            suffix = response.read(needed-start+1)
            if len(suffix) != needed-start:
                raise RuntimeError('Short/oversized probe')
            network_bytes = len(suffix)
            prefix += suffix
    frame = prefix[12:12+frame_size]
    parameters = zstd.get_frame_parameters(frame)
    if parameters.window_size > 64*2**20 or parameters.dict_id:
        raise RuntimeError('Decoder budget/dictionary unsupported')
    if parameters.content_size not in [zstd.CONTENTSIZE_UNKNOWN,zstd.CONTENTSIZE_ERROR] and parameters.content_size > 64*2**20:
        raise RuntimeError('Decoded frame exceeds budget')
    decoded = zstd.ZstdDecompressor().decompress(frame,max_output_size=64*2**20,allow_extra_data=False)
    next_magic,next_size,next_frame_size = struct.unpack('<III',prefix[12+frame_size:needed])
    if next_magic != PZSTD_MAGIC or next_size != 4:
        raise RuntimeError('Next advertised frame does not follow observed layout')
    report = dict(month=month,url=source['url'],new_network_payload_bytes=network_bytes,
        compressed_data_retained_bytes=0,decoded_data_retained_bytes=0,
        first_skippable_metadata_bytes=12,first_data_frame_compressed_bytes=frame_size,
        first_data_frame_decoded_bytes=len(decoded),window_bytes=parameters.window_size,
        frame_has_checksum=parameters.has_checksum,first_frame_decode_passed=True,
        first_frame_sha256=hashlib.sha256(frame).hexdigest(),next_frame_metadata_offset=12+frame_size,
        next_advertised_compressed_frame_bytes=next_frame_size,archive_wide_structure_verified=False,
        archive_sha256_verified=False,random_access_game_index_available=False)
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--month',default='2026-01')
    parser.add_argument('--output',default='reports/m1_pzstd_source_probe.json')
    args=parser.parse_args()
    main(args.month,args.output)
