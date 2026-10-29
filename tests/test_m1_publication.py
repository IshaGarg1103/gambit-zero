from types import SimpleNamespace

import pytest

from tinychess.data.publication import check_card_paths, check_object


def test_final_lfs_content_id_and_size_must_match():
    expected = dict(path_in_repo='data/example.parquet', bytes=123, sha256='a'*64)
    obj = SimpleNamespace(path=expected['path_in_repo'], size=123,
                          lfs=SimpleNamespace(size=123, sha256='a'*64))
    assert check_object(obj, expected)
    obj.lfs.sha256 = 'b'*64
    with pytest.raises(ValueError, match='SHA256'):
        check_object(obj, expected)
    obj.lfs = None
    assert check_object(obj, expected) is False  # Never confuse Git SHA1 with payload SHA256.
    obj.size = 122
    with pytest.raises(ValueError, match='size'):
        check_object(obj, expected)


def test_published_card_cannot_point_at_old_training_shard():
    expected = dict(positions={'train': ['filtered.parquet']}, games={'train': ['original.games.parquet']})
    metadata = dict(configs=[dict(config_name='human_v3', default=True,
                                 data_files=[dict(split='train', path=['filtered.parquet'])]),
                             dict(config_name='games', default=False,
                                  data_files=[dict(split='train', path=['original.games.parquet'])])])
    check_card_paths(metadata, expected)
    metadata['configs'][0]['data_files'][0]['path'] = ['old-unfiltered.parquet']
    with pytest.raises(ValueError, match='paths disagree'):
        check_card_paths(metadata, expected)


def test_completed_exclusion_invocation_is_idempotent_without_network(tmp_path, monkeypatch, capsys):
    import json
    from pathlib import Path
    import runpy
    script = Path(__file__).parents[1]/'scripts/apply_dataset_d_exclusions.py'
    namespace = runpy.run_path(str(script))
    monkeypatch.chdir(tmp_path)
    Path('reports').mkdir()
    frozen = dict(sha256='a'*64)
    completed = dict(status='PASS', temporary_cleanup_completed=True,
                     parent_revision=namespace['PIN'], frozen_receipt=frozen,
                     final_revision='b'*40, counts={'train': {'games': 35000}})
    fixtures = {'m1_50k_run_status.json': {},
                'm1_dataset_d_freeze.json': dict(frozen_receipt=frozen),
                'm1_50k_diagnostics.json': dict(status='PASS', temporary_cleanup_completed=True),
                'm1_dataset_d_exclusions.json': completed,
                'm1_dataset_d_publication_verification.json': dict(status='PASS', revision='b'*40)}
    for name, value in fixtures.items():
        (Path('reports')/name).write_text(json.dumps(value))
    fn = namespace['run']
    monkeypatch.setitem(fn.__globals__, 'HubTransport', lambda *a, **k: pytest.fail('Unexpected network setup'))
    fn(SimpleNamespace(execute=True, output='reports/m1_dataset_d_exclusions.json'))
    assert 'ALREADY_VERIFIED' in capsys.readouterr().out
    assert not Path('data').exists()
