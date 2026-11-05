# Mistabra Core — Conceptual API

Analyst's co-pilot that turns intuition into reasoning · Layer specification v3 · 5 November 2026

Supersedes v2. One call, `assess`, takes one episode with any part of its trajectory, completes what is missing, and assesses it against its alternatives — on its own, and together with what is already attached to the same hypothesis. Changes from v2 follow the review of 5 November: one episode per call; combination through shared hypotheses; all trajectories in the operational context; Lambda defined by how an element came to be.

## 1. What this layer is

Mistabra Core is the reasoning layer. Given what the mission cares about, what is currently believed, and an episode — with or without the analyst's own reading of it — it returns trajectories from the episode to hypotheses and risks, each assessed against its baseline alternatives.

The layer is defined by this API. Its implementation (Python library, LLM-based agent, fine-tuned model, or a combination) is a later decision and may change without changing the API.

## 2. Boundary

| In this layer | In another layer |
| --- | --- |
| Completing a partial trajectory: articulating the signal, finding or proposing the hypothesis, finding the risk | Authoring the mission context; deriving risks |
| Assessing each trajectory against its baselines | Storing hypotheses and every trajectory raised, with status |
| Weighing an episode together with the trajectories already attached to its hypothesis | Accepting, editing or rejecting hypotheses; confirming or rejecting trajectories |
| Reusing stored hypotheses and baselines; matching articulated claims to them | Deciding what enters the operational context |
| Proposing Lambda hypotheses and new baselines | Alerting, dashboards, analyst workflow, access control |
| Radars, as an internal efficiency device | Audit storage and verification |
| Provenance of every assessment | |

The layer reads both contexts and writes neither. Everything it creates — signals, trajectories, new hypotheses, new baselines — is returned; the other layer decides what to keep.

## 3. Concepts

### 3.1 Mission context

What the analyst or organization gives the system to define the mission. Changes rarely and deliberately; versioned.

| Item | Content |
| --- | --- |
| Documents | goals, rules, organization, sources and their credibility, history and past cases — free text, any number of documents |
| Risks | `id`, sentence, polarity (`negative`: to prevent \| `positive`: to achieve), status (`active` \| `retired`) |
| Version | a version id; every change to documents or risks produces a new version |

Risks are derived from the documents by another layer and stored here; this layer never derives risks on its own.

### 3.2 Operational context

What the system and analyst currently believe. Changes during work, after almost every assessment; may be scoped to a case or investigation, archived or reset without touching the mission.

| Item | Content |
| --- | --- |
| Hypotheses | `id`, `version`, `risk_id`, claim, entities, `origin` (`analyst` \| `system`), `role` (`primary` \| `baseline`), `lambda` (true when the layer articulated it, section 3.4), `status` (`proposed` \| `open` \| `supported` \| `rejected`) |
| Relations | `baseline_of(B, H)`: B is an ordinary explanation competing with primary H for the same observations |
| Trajectories | every trajectory raised and kept by the other layer, in the shape of `AssessedTrajectory` (section 4.3), each with `status` (`proposed` \| `confirmed` \| `rejected`) |

The other layer keeps trajectories even when weak, and keeps proposed hypotheses with status `proposed`: combination across episodes (section 5.5) works only on what is kept. Confirmed trajectories also serve as **precedents**: worked examples of how this mission reasons. Trajectories an organization wants to keep beyond a case are promoted by the other layer into mission documents.

Episodes themselves are not part of the operational context; trajectories carry their `episode_id` and the extracted signal.

### 3.3 Trajectory elements

**Episode** — one unit of external data: `id`, text, source and its credibility, occurrence (instant, range or unknown).

**Signal** — what in the episode is relevant to a hypothesis: structured content (actor, behaviour, target, means, time, modifiers, entities, one-sentence summary).

**Hypothesis** — a claim about the world that, if true, connects the signal to the risk.

**Risk** — as in mission context.

**Trajectory** — episode → signal → hypothesis → risk, with the reasoning that connects them.

**Baseline trajectory** — the same episode leading to a baseline hypothesis and ending in an ordinary outcome.

### 3.4 How an element came to be, and Lambda

Every signal and hypothesis in an output carries its `source`:

| Element | `given` | `predefined` | `articulated` |
| --- | --- | --- | --- |
| Signal | the analyst wrote it | one of the layer's radars asked for exactly this | neither — a **Lambda signal** |
| Hypothesis | the analyst or caller named it | matched to a stored hypothesis | neither — a **Lambda hypothesis** |
| Risk | the caller named it | always: risks are stored | never |

Lambda means *articulated by the layer*: something nobody stated in advance and the layer put into words. Who triggered the call does not matter:

- A stream episode no radar matched yields a Lambda signal.
- An analyst who submits an episode and a hypothesis without saying what connects them — sensing the link without stating it — yields a Lambda signal. The layer turns that intuition into reasoning.
- An analyst who submits an episode and a signal without a hypothesis — "this matters, I don't know what it means" — yields a Lambda hypothesis, unless the layer matches a stored hypothesis (then the hypothesis is `predefined`; see open question 2).
- A stream episode no stored hypothesis explains yields a Lambda hypothesis.

Never a bare "Lambda".

## 4. The call

```
assess(mission, operational, trajectory, *, scope="given", detail="full") -> Assessment
```

### 4.1 Input trajectory

| Element | Required | Form |
| --- | --- | --- |
| `episode` | yes, exactly one | episode record |
| `signal` | no | a draft: content fields and/or a free-text note of what matters |
| `hypothesis` | no | a reference (`id` in operational context) or a draft (free-text claim) |
| `risk` | no | a reference (`id` in mission context) |

Elements provided by the caller are **given**: the layer assesses them and never replaces them. Elements not provided are completed by the layer, as `predefined` or `articulated`.

### 4.2 The eight input patterns

| # | Given | The layer completes | Typical use |
| --- | --- | --- | --- |
| 1 | E | signal, hypothesis, risk — every reasonable trajectory to any active risk | stream of incoming episodes |
| 2 | E, S | hypothesis, risk | analyst: "this matters, I don't know what it means" |
| 3 | E, H | signal, risk (the hypothesis's own risk if stored; derived if a draft) | analyst: "this bears on H, I can't say how" |
| 4 | E, R | signal, hypothesis | "does this bear on risk R?" |
| 5 | E, S, H | risk | analyst's hunch, with the hypothesis |
| 6 | E, S, R | hypothesis | analyst's hunch, with the risk |
| 7 | E, H, R | signal | checking a known path |
| 8 | E, S, H, R | nothing — assess only | the analyst's full trajectory |

When given elements do not fit together — a stored hypothesis belongs to another risk than the one given, or the signal does not appear in the episode — the layer returns the group with an `inconsistency` note and does not alter the given elements.

### 4.3 Output

```
Assessment:
  groups:     [AssessmentGroup]     # empty: the episode bears on no active risk
  provenance: Provenance

AssessmentGroup:
  target:     AssessedTrajectory    # role primary
  baselines:  [AssessedTrajectory]  # role baseline; competing ordinary explanations
  standing:   Standing              # section 5.2
  inconsistency: str | None

AssessedTrajectory:
  episode_id
  signal:     {content, source: given | predefined | articulated}
  hypothesis: HypothesisRef
  risk:       {risk_id, source: given | predefined}
  score:      [0, 1]                # the episode in isolation, section 5.1
  reach:      [0, 1]                # how far along the path to the risk the episode sits
  fit:        [0, 1]                # how well the episode supports the hypothesis
  missing:    [str]                 # information that would change the assessment
  rationale:  str                   # omitted when detail="scores"
  ordinary_outcome: str | None      # baselines only
  stance:     supports | weakens     # weakens: counter-evidence against the hypothesis (v3.1)
  lambda_signal:     bool           # signal.source == articulated
  lambda_hypothesis: bool           # hypothesis.source == articulated

HypothesisRef:
  source: given | predefined | articulated
  given:        {id, version} or {claim}
  predefined:   {id, version, match_confidence}
  articulated:  {claim, entities}
  status: proposed | open | supported | rejected         # of the stored hypothesis, if any

Standing:
  value:      [0, 1]                # the hypothesis given all attached trajectories and this one
  without:    [0, 1]                # the same, without this episode
  delta:      value - without       # what this episode adds
  used:       [{trajectory_id, status, weight}]
  rationale:  str                   # omitted when detail="scores"
```

### 4.4 Options

`scope` — which hypotheses the layer may report when the caller gave a hypothesis or a risk:

- `given` (default): only groups consistent with the given elements — with a given hypothesis, exactly one group; with a given risk, groups under that risk.
- `open`: also any other group the episode supports, under any active risk. The groups for the given elements come first.

With nothing given beyond the episode or the signal (patterns 1 and 2), `scope` has no effect.

`detail` — how much to return:

- `full` (default): everything in section 4.3.
- `scores`: the same groups, with every `rationale` omitted. For streams: call with `scores`, then again with `full` for the groups worth reading. Both calls return identical values for identical inputs.

## 5. Semantics

### 5.1 Score: the episode in isolation

`score` of a primary trajectory answers: *how much does this episode, on its own, support this hypothesis against its baselines?*

- `0`: irrelevant to the hypothesis, or fully explained by a baseline; `1`: the risk-bearing behaviour itself, with no ordinary explanation.
- Independent of every other trajectory in the operational context. This keeps it comparable from one episode to the next and protects it from the layer's own earlier guesses.
- Independent across groups: scores do not sum to anything; one episode may support several hypotheses strongly, or none.
- Relative within a group: computed from the primary's `fit` against the best baseline's `fit`, its `reach`, signal specificity and source credibility. The function is documented, deterministic given those values, and the same for every implementation of this API version.
- A baseline trajectory's `score` is its own `fit`, reported so the analyst can see what the primary was weighed against.

Because a hypothesis is always scored against its stored baselines, its scores are comparable across episodes.

### 5.2 Standing: the hypothesis given everything attached

`standing.value` answers: *how well is this hypothesis supported, against its baselines, by all trajectories attached to it plus this one?* It is where individually weak episodes combine.

- The layer judges the attached trajectories jointly, not by summing scores: two weak trajectories may together make the hypothesis likely, and two strong ones may be the same fact reported twice.
- Trajectories are weighted by status: `confirmed` fully; `proposed` less (a documented weight below 1); `rejected` not at all — the analyst judged that episode does not support the hypothesis — but listed in `used` so the exclusion is visible.
- `without` is the standing computed the same way but without this episode; `delta` is what this episode adds. A small `score` with a large `delta` is the signature of a weak signal that matters in combination.
- Standing is returned for every group, including a Lambda hypothesis (then `without` is 0 unless earlier trajectories match it).

Two safeguards against the system talking itself into a hypothesis: `score` never uses other trajectories, and `proposed` trajectories — the layer's own unconfirmed outputs — carry reduced weight in standing.

### 5.2a Counter-evidence (v3.1)

An episode can make a stored hypothesis less likely — "the drains were cleared last week" against "uncleared drains will flood the
lower districts". For every stored hypothesis of the risk the layer states a stance: supports, weakens or unrelated. A weakening
episode is returned as a trajectory to that hypothesis with `stance: weakens` and `score` 0; it is kept like any other trajectory,
and standing weighs it as evidence against the hypothesis. Without this, counter-evidence would be judged irrelevant and dropped,
and a hypothesis could only ever gain standing.

### 5.3 Reuse of stored hypotheses and baselines

- The layer matches every articulated claim against the stored hypotheses of the relevant risk — including `proposed` ones — and reuses a match when it is the same claim, reporting `match_confidence`. Returning `articulated` for a claim a stored hypothesis already expresses is a defect and is measured (section 8).
- For a matched or given primary, the layer uses its stored `baseline_of` hypotheses. It adds a new baseline only when the episode suggests an ordinary explanation none of the stored ones covers.
- Rejected hypotheses are never articulated again. If the episode supports a rejected hypothesis, the group is returned with `status: rejected`, under `scope="open"` or when that hypothesis was given.

### 5.4 Lambda hypotheses

When the episode plausibly leads to an active risk and no stored primary hypothesis explains it, the layer articulates one, with its trajectory and baselines. It is a proposal; the other layer stores it with status `proposed` or discards it. Stored, it can be matched by later episodes — which is how a hypothesis no one stated accumulates support one episode at a time.

### 5.5 Combination across episodes

There is no call that combines episodes, and no one chooses which episodes to combine. Combination happens through the hypothesis: when an episode's trajectory reaches a hypothesis that already has trajectories attached, `standing` weighs them together.

This depends on two things the other layer does: keeping trajectories even when weak, and keeping proposed hypotheses. It also depends on matching: two weak episodes whose trajectories reach differently worded hypotheses never meet. Matching quality is therefore measured (A4).

### 5.6 Mission context use

The layer retrieves from mission documents what each judgment needs (sources' credibility, rules, history). Risks are used exactly as stored: by id, with their sentence and polarity. A retired risk is never a completion target.

### 5.7 Provenance

Every assessment carries:

| Field | Content |
| --- | --- |
| `implementation` | name and version |
| `models`, `prompts` | ids and versions used |
| `mission_version` | the mission context version read |
| `operational_items` | ids and versions of every hypothesis and trajectory read |
| `input_hash` | hash of the input trajectory and options |
| `deterministic` | whether the models ran deterministically (e.g. temperature 0) |

Identical inputs, identical context versions and the same implementation version return identical results. This is what makes replay, audit and evaluation possible in the other layer.

### 5.8 Feedback

Feedback never calls this layer. It reaches the layer through the operational context on later calls: hypothesis statuses, trajectory statuses, and confirmed trajectories as precedents. An implementation that learns from feedback does so behind this API.

## 6. Examples

**Stream episode, Lambda hypothesis.** Pattern 1. Episode: "A passer-by photographed the front of Jens Hofer's house on Elm Street." No stored hypothesis about Elm Street. One group under the attack risk: hypothesis `articulated` ("the house on Elm Street is a target of hostile preparation"), score 0.3; baselines `articulated` ("property listing", "hobby album"), fit about 0.5; standing 0.3. Signal `predefined` (a radar for attention to members' homes matched).

**Same hypothesis, two days later.** Pattern 1. "The same man photographed the house again, from a parked car." Hypothesis `predefined` (matched to the stored proposal, `match_confidence` 0.9), same two baselines, score 0.55; standing `without` 0.3, `value` 0.65, `delta` 0.35.

**Analyst senses a link — Lambda signal.** Pattern 3. Episode: a community-centre volunteer reports someone asked which doors stay unlocked during events; hypothesis given: "the Community Centre is a target of hostile preparation". No signal given, no radar asked for door access. Signal `articulated` (actor: unknown visitor; behaviour: asking about unlocked entrances; target: Community Centre), `lambda_signal` true, score 0.6.

**Analyst notices something — Lambda hypothesis.** Pattern 2. Episode: a supplier quote; signal given: "the same three-person board appears on two unrelated bidders". No stored hypothesis explains it. Hypothesis `articulated` ("the bidders are a single entity bidding through fronts"), `lambda_hypothesis` true.

**Conjunction through a shared hypothesis.** Two pattern 4 calls under the urban-flood risk. Call 1, heavy rain forecast: hypothesis `articulated` ("drainage capacity is below expected runoff"), score 0.3, standing 0.3; stored as `proposed`. Call 2, drain maintenance budget cut: hypothesis `predefined` (matched), score 0.25; standing `without` 0.3 (the proposed rain trajectory, reduced weight), `value` 0.65, `delta` 0.35.

**Cluster Lambda, one episode at a time.** Stored hypotheses: expired catalyst; low mixing speed. Call 1, ventilation intake blocked by wrap: both stored hypotheses score low; a third group `articulated` ("contamination through a ventilation failure"), score 0.35. Calls 2–4 (pressure drop, dust spike, cellulose fibres) each match it; standing rises to about 0.8 by the fourth.

## 7. Out of scope for v1

- Expected-but-missing evidence and belief decay over time.
- Actions: which radar to run next, what to ask the analyst, when to alert.
- Re-assessing all hypotheses of a risk at a point in time without a new episode (a possible later call, reusing `standing`).
- Entity resolution beyond what extraction yields.
- Deriving risks.

## 8. Evaluation

The project's case files map onto sequences of calls:

| Case file element | API |
| --- | --- |
| `risks` / analytical context | mission context |
| `claims` | stored primary hypotheses in operational context |
| `episodes` | one call each, in order, with the returned trajectories kept as `proposed` |
| `ground_truth.claims[]` verdicts and score ranges | targets for `standing.value` after the last episode |
| `phase1_cases` `conj_*` | data points as a sequence under the stated hypothesis: no single member lowers its standing much, the completed refuting set does; `conj_ctrl_*` no such drop |
| `phase1_cases` `miss_*` | data points as a sequence with the given hypotheses stored: the planted cluster should produce a Lambda hypothesis matching `missing_hypothesis`, whose standing rises as members arrive; `miss_ctrl_*` none |

| Id | Measurement |
| --- | --- |
| A1 | Ranking and calibration of `score` against labelled bands; hard negatives low |
| A2 | Lambda hypotheses articulated when the true explanation is absent, not in controls |
| A3 | `standing.delta` separates jointly-strong sequences from controls |
| A4 | Matching: share of articulated claims that should have matched a stored hypothesis (false `articulated` rate); wrong matches |
| A5 | Score stability: the same hypothesis scored on paraphrased episodes varies within a tolerance |
| A6 | Self-reinforcement: on noise sequences, standing does not rise from `proposed` trajectories alone |

Each is measured first for the reference implementation against the pairwise baseline, then for any later implementation against the reference.

## 9. Implementation notes (non-normative)

**Reuse from the PoC.** LLM gateway and provider abstraction; versioned prompts; fake provider; the abducer (`trajectory.risk`, `trajectory.baseline`), which already completes hypotheses and baselines and reuses existing ones; the decision models — `pairwise` already computes `score`, and the set-wise model's use of attached signals is the starting point for `standing`; the PoC's replay of state as of a signal's arrival gives `standing.without`. New: completion for the eight patterns, `source` on each element, matching with confidence, status weighting, `scope` and `detail`, provenance for both contexts. The PoC's store and facade APIs become the other layer.

**Radars.** Derived inside the layer from the active risks and the stored primary hypotheses, cached per (mission version, operational hash), used to skip full reasoning for episodes that bear on nothing. A radar match makes the signal `predefined`; no match makes it `articulated`.

**Implementations.** `reference` (the reasoning model for every judgment; the teacher), `distilled` (a small model trained on the reference for `detail="scores"` on streams), `adapted` (distilled, further trained on a tenant's feedback). One API; provenance names which one answered.

**Bounding standing.** A heavily supported hypothesis may have many attached trajectories; the layer may summarise or sample them, listing in `used` exactly what it weighed.

## 10. Open questions

1. Whether a new baseline must be accepted before it is reused, or is reused as soon as the other layer stores it.
2. Analyst gives episode and signal; the layer matches a stored hypothesis. v3 treats the hypothesis as `predefined`, so "Lambda hypothesis" always means a new hypothesis. The alternative is to treat every hypothesis the analyst did not name as Lambda.
3. The weight of `proposed` trajectories in standing, and whether the other layer may set it.
4. Whether `scope="open"` should also search under risks other than the given one, or only for other hypotheses under it.
