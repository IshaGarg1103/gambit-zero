"""Do not launch a second extractor before its predecessor has fully finished."""
from pathlib import Path
import runpy
import pytest

READY = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/run_after_m1_month.py'))['predecessor_ready']


@pytest.mark.parametrize('status,games,session,expected', [
    ('RUNNING', 7500, ['2026-08'], False),
    ('MONTH_COMPLETE', 0, ['2026-08'], False),
    ('MONTH_COMPLETE', 7500, ['2026-07'], False),
    ('MONTH_COMPLETE', 7500, ['2026-08'], True),
])
def test_requires_final_verified_predecessor(status, games, session, expected):
    state = dict(status=status, session_months=session,
                 months={'2026-08': dict(cloud_reproduced_games=games)})
    assert READY(state, '2026-08', 7500) is expected


def test_failure_stops_waiter():
    with pytest.raises(RuntimeError, match='Generation stopped'):
        READY(dict(status='STOPPED', error={'message': 'timeout'}), '2026-08', 7500)
