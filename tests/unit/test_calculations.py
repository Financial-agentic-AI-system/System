from src.utils.test import calculate_profit
# TODO: remove further - currently for testing workflow

def test_calculate_profit():
    assert calculate_profit(100.0, 150.0) == 50.0

def test_calculate_loss():
    assert calculate_profit(100.0, 80.0) == -20.0