# tests/test_calculator.py
import pytest
from src.calculator import Calculator

calc = Calculator()

def test_add():
    assert calc.add(2, 3) == 5

def test_multiply():
    assert calc.multiply(2, 3) == 6

@pytest.mark.parametrize("a,b", [(None, 1), (1, None), (None, None)])
def test_add_invalid_inputs(a, b):
    with pytest.raises(TypeError):
        calc.add(a, b)
