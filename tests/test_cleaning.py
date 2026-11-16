import pandas as pd

from rdt.data.cleaning import apply_arrival_rules, clean_training_readings, passes_arrival_rules


def test_arrival_rules():
    assert passes_arrival_rules("discharge", 100.0)
    assert not passes_arrival_rules("turbidity", -1.7)
    assert not passes_arrival_rules("spm_daily", 0.0)
    assert not passes_arrival_rules("spm_cd", float("nan"))
    assert passes_arrival_rules("spm_cd", 0.5)


def test_training_only_qc_rules():
    r = pd.DataFrame({
        "sensor_id": ["lobith_discharge"] * 5 + ["eijsden_turbidity"],
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, -1.0],
        "qc": ["00", "50", "20", "25", "1031", "00"],
    })
    assert len(apply_arrival_rules(r)) == 5                       # only the negative turbidity goes
    assert clean_training_readings(r)["qc"].tolist() == ["00", "1031"]
