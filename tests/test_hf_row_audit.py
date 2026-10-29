"""A row audit must compare bytes and labels, not just remote receipts."""
from pathlib import Path
import runpy
import pytest


def test_audit_config_survives_cleanup_and_checks_heldout_quotas():
    import yaml
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / 'configs/audit_human_50k.yaml').read_text())
    expected_games = runpy.run_path(str(root / 'scripts/audit_hf_rows.py'))['expected_games']
    assert (root / config['source_manifest']).is_file()
    assert expected_games(config, '2026-07') == 5000
    assert expected_games(config, '2026-08') == 7500
    assert expected_games(config, '2026-09') == 7500
    config['months']['2026-08']['split'] = 'train'
    with pytest.raises(ValueError, match='split'):
        expected_games(config, '2026-08')


@pytest.mark.parametrize('field,value', [('action', 2), ('planes', b'changed'), ('legal_mask', b'changed'), ('outcome_stm', -1)])
def test_row_audit_rejects_changed_field(field, value):
    fn = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/audit_hf_rows.py'))['compare_row']
    expected = dict(game_id='example', ply=0, action=1, planes=b'original', legal_mask=b'original', outcome_stm=1)
    actual = dict(expected)
    fn(actual, expected)
    actual[field] = value
    with pytest.raises(ValueError, match='field='+field):
        fn(actual, expected)


def test_row_audit_accepts_parquet_list_name_roundtrip(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tinychess.data.storage import GAME_SCHEMA
    fn = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/audit_hf_rows.py'))['parquet_rows']
    p = tmp_path / 'games.parquet'
    record = dict(game_id='example', metadata='{}', actions=[1, 2])
    pq.write_table(pa.Table.from_pylist([record], schema=GAME_SCHEMA), p)
    assert list(fn(p, GAME_SCHEMA)) == [record]
    changed = GAME_SCHEMA.with_metadata({b'schema': b'wrong-version'})
    with pytest.raises(ValueError, match='Schema mismatch'):
        list(fn(p, changed))
