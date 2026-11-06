"""Unit tier: offline and deterministic. MISTABRA_FORCE_FAKE guarantees no key can route a test to a paid model."""
import os

os.environ["MISTABRA_FORCE_FAKE"] = "1"

import pytest  # noqa: E402

from mistabra_core import (Core, Document, Episode, Hypothesis, MissionContext, OperationalContext, Risk,  # noqa: E402
                           Settings, StoredTrajectory)
from mistabra_core.llm import FakeProvider  # noqa: E402

HOUSE = "A passer-by photographed the front of Jens Hofer's house on Elm Street while walking past in the morning."
HOODED = ("From a grey Volkswagen with covered plates parked on Maple Road, a hooded person filmed the entrances of the "
          "Community Centre for over an hour at night through a telephoto lens.")
AGENT = "An estate agent from Northfield Homes Ltd, wearing a company lanyard, photographed the front of a house on Elm Street that is listed for sale."
GROCERIES = "A shopper bought groceries at the Maple Road supermarket."
DOORS = "A volunteer at the Community Centre reports that an unknown visitor asked which side doors stay unlocked during events."


def ep(text, eid="e1", **kw):
    return Episode(id=eid, text=text, source=kw.pop("source", "Field reports"), credibility=kw.pop("credibility", 0.8),
                   occurred=kw.pop("occurred", "2026-08-03T10:00:00Z"), **kw)


@pytest.fixture
def mission():
    return MissionContext(version="m1", documents=(
        Document(id="about", title="about", text="Community X in Northfield: Community Centre on Maple Road, Place of Worship on Church Lane, homes on Elm Street."),),
        risks=(Risk(id="attack", text="Hateful radicals attack members of Community X, their homes, or community institutions."),
               Risk(id="burglary", text="Burglars target homes in the Riverside neighbourhood."),
               Risk(id="old", text="An old concern nobody tracks any more.", status="retired")))


@pytest.fixture
def op():
    return OperationalContext(hypotheses=(
        Hypothesis(id="H_elm", risk_id="attack", claim="Homes on Elm Street are a target of hostile preparation", origin="analyst"),
        Hypothesis(id="H_centre", risk_id="attack", claim="The Community Centre is a target of hostile preparation", origin="analyst"),
        Hypothesis(id="B_listing", risk_id="attack", claim="the person has a property interest", role="baseline", baseline_of=("H_elm",)),
        Hypothesis(id="H_worship", risk_id="attack", claim="The Place of Worship is a target of hostile preparation", status="rejected"),
    ))


def make_core(overrides=None, **settings):
    """Tests exercise the judged path unless they say otherwise (it has the most moving parts); the product default is
    noisy_or (test_default_standing_is_noisy_or)."""
    settings.setdefault("standing", "judged")
    return Core(Settings(force_fake=True, **settings), provider=FakeProvider(overrides or {}))


@pytest.fixture
def core():
    return make_core()


def traj(tid, hid, score, status="proposed", eid=None):
    return StoredTrajectory(id=tid, episode_id=eid or f"ep-{tid}", hypothesis_id=hid, risk_id="attack", score=score, fit=score, reach=0.3, status=status)
