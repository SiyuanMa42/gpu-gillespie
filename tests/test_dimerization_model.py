import pytest

from gpu_gillespie.models.basic_models import DimerizationModel


def test_equilibrium_constant_value():
    """get_equilibrium_constant() should return k_dimerization / k_dissociation"""
    model = DimerizationModel(k_dimerization=0.1, k_dissociation=0.01)
    expected = 0.1 / 0.01
    assert pytest.approx(model.get_equilibrium_constant(), rel=1e-9) == expected


def test_equilibrium_constant_type_and_scaling():
    """Return type is float and scales correctly for other parameters"""
    model = DimerizationModel(k_dimerization=2.0, k_dissociation=0.5)
    val = model.get_equilibrium_constant()
    assert isinstance(val, float)
    assert val == pytest.approx(4.0)
