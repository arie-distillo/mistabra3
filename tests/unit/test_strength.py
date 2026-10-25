from mistabra.engine import strength_from


def test_strength_monotone_and_bounded():
    base = dict(reach=0.3, fit=0.4, gap=0.0, specificity=1.0, credibility=0.8)
    s0 = strength_from(**base)
    assert 0 <= s0 <= 1
    assert strength_from(**{**base, "fit": 0.9}) > s0 and strength_from(**{**base, "gap": 0.5}) > s0 and strength_from(**{**base, "reach": 0.9}) > s0
    assert strength_from(**{**base, "gap": -0.5}) < s0 and strength_from(**{**base, "credibility": 0.4}) < s0
    assert strength_from(reach=1, fit=1, gap=1, specificity=1, credibility=1) == 1.0
