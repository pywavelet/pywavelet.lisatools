import os
import pytest

@pytest.fixture
def outdir():
    d = 'out_test'
    os.makedirs(d, exist_ok=True)
    return d