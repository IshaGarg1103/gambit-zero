"""Charger enforcement defaults on; explicit battery runs can opt out."""
from pathlib import Path
import runpy

import pytest


@pytest.mark.parametrize('config',[{'require_external_power':False}])
def test_battery_operation_allowed_without_hardware_query(config,monkeypatch):
    fn=runpy.run_path(str(Path(__file__).parents[1]/'scripts/run_dataset_a_pilot.py'))['startup_power_status']
    def forbidden():pytest.fail('Charger hardware must not be queried when enforcement is off')
    monkeypatch.setitem(fn.__globals__,'external_power_status',forbidden)
    assert fn(config)==dict(required=False,monitoring_enabled=False)


@pytest.mark.parametrize('connected',[False,True])
@pytest.mark.parametrize('config',[{}, {'require_external_power':True}])
def test_external_power_enforced_by_default_or_explicit_setting(connected,config,monkeypatch):
    module=runpy.run_path(str(Path(__file__).parents[1]/'scripts/run_dataset_a_pilot.py'))
    fn=module['startup_power_status']
    monkeypatch.setitem(fn.__globals__,'external_power_status',lambda:dict(required=True,external_connected=connected))
    if connected:
        assert fn(config)['external_connected']
    else:
        with pytest.raises(module['ScanError'],match='explicitly required'):
            fn(config)
