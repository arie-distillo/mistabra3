"""Unit tier: offline, deterministic, fake provider. MISTABRA_FORCE_MOCK is set so a key
in the environment or a .env can never route a unit test to a paid model."""
import os
import pytest

os.environ["MISTABRA_FORCE_MOCK"] = "1"

from mistabra import Mistabra, Settings          # noqa: E402
from mistabra.llm import FakeProvider           # noqa: E402

TESTDATA = os.path.join(os.path.dirname(__file__), "..", "..", "testdata")
DOC = ("risk_register.md",
       "Risk R1: Hateful radicals plan a violent attack on members of Community X.\n\n"
       "Risk R2: Burglary of homes in the Riverside neighbourhood.\n\n"
       "Term: Community institution: a facility used by Community X.\n\n"
       "Term: Community Centre (the centre): main venue, a kind of Community institution.\n\n"
       "Entity type: person (name, aliases)\n\nEntity type: institution (name, address)\n\nEntity type: location (name)\n\n"
       "Source: Field reports (credibility 0.8)")


@pytest.fixture
def m(tmp_path):
    inst = Mistabra(tmp_path, settings=Settings(data_dir=tmp_path, force_mock=True), provider=FakeProvider())
    yield inst
    inst.close()


@pytest.fixture
def published(m):
    did = m.ontology.build_draft("default", [DOC], actor="de")
    m.ontology.accept_all(did, "de")
    return m.ontology.publish(did, "de")


@pytest.fixture
def source(m):
    return m.sources.register("Field reports", "manual", 0.8, "admin")
