"""Offline cached-prefix scanner cost; never a representative sampling run."""
import json
from pathlib import Path
import resource
import sys
import time

from tinychess.data.archive_scan import DaySampler
from tinychess.data.sources import game_blocks,open_text
from tinychess.data.evidence import read_json_evidence


def main():
    frozen=read_json_evidence('reports/m1_50k_preflight.json')
    sizes={s['month']:s['archive_size_bytes'] for s in frozen['sources']}
    rows=[]
    for path in sorted(Path('data/raw/m1_prefixes').glob('*.pgn.zst.prefix')):
        month=path.name[:7]
        sampler=DaySampler(month,5000 if month<='2026-07' else 7500,20261025)
        started=time.perf_counter()
        with open_text(path) as text:
            for ordinal,(block,bad) in enumerate(game_blocks(text),1):
                sampler.offer(block,ordinal,bad)
        elapsed=time.perf_counter()-started
        rows.append(dict(month=month,compressed_prefix_bytes=path.stat().st_size,
            elapsed_seconds=elapsed,game_blocks=sampler.counts['source_game_blocks'],
            header_eligible_occurrences=sampler.counts['header_eligible_occurrences'],
            header_rejections=dict(sampler.rejections),retained_candidates=len(sampler.retained),
            candidate_payload_bytes=sampler.candidate_bytes,
            compressed_bytes_per_second=path.stat().st_size/elapsed,
            estimated_full_month_cpu_seconds=sizes[month]/(path.stat().st_size/elapsed),
            representative_sample=False,full_source_verified=False,
            tail_may_be_partial=True))
    report=dict(status='BOUNDED_PREFIX_PROFILE_ONLY',rows=rows,
                estimated_nine_month_cpu_seconds=sum(r['estimated_full_month_cpu_seconds'] for r in rows),
                peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                limitations='Cached prefix, same UTC day. Extrapolation is not measured full-archive runtime; network excluded.')
    Path('reports/m1_streaming_header_profile.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
