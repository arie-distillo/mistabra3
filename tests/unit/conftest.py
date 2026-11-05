"""Unit tier: offline, deterministic. MISTABRA_FORCE_FAKE guarantees no key can route a test to a paid model."""
import os, pytest
os.environ["MISTABRA_FORCE_FAKE"] = "1"
from mistabra import Mistabra, Settings          # noqa: E402
from mistabra.llm import FakeProvider           # noqa: E402

TESTDATA = os.path.join(os.path.dirname(__file__), "..", "..", "testdata")
HOUSE = "A passer-by photographed the front of Jens Hofer's house on Elm Street while walking past in the morning."
HOODED = "From a grey Volkswagen with covered plates parked on Maple Road, a hooded person filmed the entrances of the Community Centre for over an hour at night through a telephoto lens."
AGENT = "An estate agent from Northfield Homes Ltd, wearing a company lanyard, photographed the front of a house on Elm Street that is listed for sale."
FLYTHROUGH = "Someone posted a 3D fly-through model of the Community Centre interior reconstructed from publicly shared event photos."
GROCERIES = "A shopper bought groceries at the Maple Road supermarket."


@pytest.fixture
def m(tmp_path):
    inst = Mistabra(tmp_path, Settings(data_dir=tmp_path), provider=FakeProvider())
    yield inst
    inst.close()


@pytest.fixture
def ready(m):
    """Context, one risk with derived radars, one source."""
    m.context.create("Community X in Northfield; institutions: Community Centre on Maple Road, Place of Worship on Church Lane.", "a", title="about")
    risk = m.risks.create("Hateful radicals attack members of Community X.", "a")
    m.radars.derive(risk.id, "a")
    m.sources.create("Field reports", "a", credibility=0.8)
    return risk
