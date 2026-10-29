"""Catalog retries reuse an already verified independent reproduction."""
import json
from types import SimpleNamespace
import pytest
from tinychess.data.hf_destination import HfMonthlyDestination


@pytest.mark.parametrize('already_reproduced', [False, True])
def test_completed_month_resume_only_rebuilds_when_needed(tmp_path, already_reproduced):
    month = '2026-01'
    complete = dict(source=dict(published_sha256='abc'), accepted_games=5000, dataset_d_sha256=None)
    (tmp_path / f'run-{month}.complete.json').write_text(json.dumps(complete))
    destination = object.__new__(HfMonthlyDestination)
    destination.run_id = 'run'
    destination.config = dict(months={month: dict(accepted_games=5000)})
    destination.d_sha = None
    destination.reproduced = {month} if already_reproduced else set()
    uploads, builds = [], []
    destination.uploader = SimpleNamespace(root=tmp_path, publish=lambda *a, **k: uploads.append(a))
    destination._second_build = lambda value: builds.append(value)
    assert destination.resume_completed(dict(month=month, published_sha256='abc'))
    assert len(uploads) == 1
    assert builds == ([] if already_reproduced else [complete])
