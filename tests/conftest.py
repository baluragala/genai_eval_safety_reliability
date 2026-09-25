import pytest

import agentlab as al
from fake_openai import FakeOpenAI


@pytest.fixture(autouse=True)
def fake_client():
    al.llm.set_client(FakeOpenAI())
    al.METER.__init__()
    yield
    al.llm.set_client(None)
