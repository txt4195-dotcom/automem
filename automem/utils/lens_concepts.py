"""Canonical concept dimensions for User lens and Memory stance.

Storage: named properties per concept on each node.

  User:   lens_<category>_<concept> = [a, b]  — Beta distribution pair.
          e.g. lens_culture_individualism_collectivism = [2.0, 1.0]
          a = evidence for plus pole, b = evidence for minus pole.
          p_user = a / (a + b)  →  direction.
          a + b = total evidence  →  confidence.
          [1, 1] = uniform prior (no opinion, skipped in scoring).
          Property absent = unobserved (skipped in scoring).

  Memory: lor_<category>_<concept> = float  — signed log-odds of content's stance.
          e.g. lor_culture_individualism_collectivism = 1.2
          0.0 = neutral on that axis.
          >0 = leans toward plus pole, <0 = leans toward minus pole.
          Property absent = unscored (no evidence for that axis).

Scoring in Cypher (log-space, no exp/sigmoid needed for ranking):
  alignment = (a - b) * lor  — positive when user and content agree.
  Sum across dimensions → profile_score.

56 concepts across 7 categories, derived from:
- Hofstede's cultural dimensions
- World Values Survey (Inglehart)
- Moral Foundations Theory (Haidt)
- Pew Research religiosity/spirituality decomposition
- Document type emphasis (unipolar, like moral foundations)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple


# Each concept: (key, plus_pole_label, minus_pole_label)
# plus_pole (+lor) = left in the name, minus_pole (-lor) = right in the name.
LENS_CATEGORIES: Dict[str, List[str]] = OrderedDict([
    ("culture", [
        "individualism_collectivism",
        "hierarchy_egalitarian",
        "uncertainty_tolerance",
        "longterm_shortterm",
        "indulgence_restraint",
        "highcontext_lowcontext",
        "competition_cooperation",
        "social_trust_distrust",
    ]),
    ("polity", [
        "liberty_authority",
        "tradition_change",
        "nationalism_cosmopolitanism",
        "universalism_particularism",
        "pluralism_orthodoxy",
        "security_freedom",
        "populism_elitism",
        "institutional_trust",
        "gender_role_traditionalism",
        "sexual_permissiveness",
    ]),
    ("economy", [
        "market_redistribution",
        "merit_equity",
        "growth_sustainability",
        "consumerism_minimalism",
        "achievement_balance",
        "anthropocentrism_ecocentrism",
    ]),
    ("epistemic", [
        "empiricism_revelation",
        "rational_intuitive",
        "certainty_ambiguity",
        "agency_fatalism",
        "human_tech_orientation",
        "novelty_familiarity",
        "abstraction_concreteness",
        "systems_personalism",
    ]),
    ("moral", [
        "care_emphasis",
        "fairness_emphasis",
        "loyalty_emphasis",
        "authority_emphasis",
        "sanctity_emphasis",
        "liberty_emphasis",
    ]),
    ("religion", [
        "religiosity",
        "spirituality",
        "institutional_religion_affinity",
        "doctrinal_certainty",
        "transcendence_orientation",
        "afterlife_orientation",
        "ritual_formality",
        "sacred_boundary",
        "spirit_world_belief",
        "exclusivism_pluralism",
    ]),
    ("doctype", [
        "decision_emphasis",
        "pattern_emphasis",
        "preference_emphasis",
        "style_emphasis",
        "habit_emphasis",
        "insight_emphasis",
        "context_emphasis",
        "factual_emphasis",
    ]),
])

# Bipolar labels: concept_key -> (plus_label, minus_label)
# nano sees these to judge which direction the content leans.
CONCEPT_POLES: Dict[str, Tuple[str, str]] = {
    # culture
    "individualism_collectivism": ("individualism, personal autonomy", "collectivism, group harmony"),
    "hierarchy_egalitarian": ("hierarchy, vertical structure", "egalitarian, flat/equal"),
    "uncertainty_tolerance": ("tolerates uncertainty, embraces ambiguity", "avoids uncertainty, seeks rules"),
    "longterm_shortterm": ("long-term orientation, delayed gratification", "short-term, immediate results"),
    "indulgence_restraint": ("indulgence, personal freedom to enjoy", "restraint, duty and discipline"),
    "highcontext_lowcontext": ("high-context, implicit communication", "low-context, explicit communication"),
    "competition_cooperation": ("competition, winning matters", "cooperation, working together"),
    "social_trust_distrust": ("social trust, people are trustworthy", "social distrust, people are unreliable"),
    # polity
    "liberty_authority": ("liberty, individual freedom", "authority, strong governance"),
    "tradition_change": ("tradition, preserve existing order", "change, reform and progress"),
    "nationalism_cosmopolitanism": ("nationalism, national identity first", "cosmopolitanism, global citizen"),
    "universalism_particularism": ("universalism, same rules for all", "particularism, context-dependent rules"),
    "pluralism_orthodoxy": ("pluralism, many valid worldviews", "orthodoxy, one correct worldview"),
    "security_freedom": ("security, safety and stability", "freedom, risk and openness"),
    "populism_elitism": ("populism, power to the people", "elitism, defer to experts"),
    "institutional_trust": ("trusts institutions", "distrusts institutions"),
    "gender_role_traditionalism": ("traditional gender roles", "gender equality / fluid roles"),
    "sexual_permissiveness": ("sexually permissive / open", "sexually conservative / restrictive"),
    # economy
    "market_redistribution": ("free market, minimal regulation", "redistribution, government intervention"),
    "merit_equity": ("meritocracy, reward performance", "equity, equal outcomes"),
    "growth_sustainability": ("economic growth, GDP first", "sustainability, ecological limits"),
    "consumerism_minimalism": ("consumerism, more is better", "minimalism, less is more"),
    "achievement_balance": ("achievement, career success", "balance, work-life harmony"),
    "anthropocentrism_ecocentrism": ("anthropocentrism, human needs first", "ecocentrism, nature has value"),
    # epistemic
    "empiricism_revelation": ("empiricism, evidence and data", "revelation, faith and authority"),
    "rational_intuitive": ("rational, logical analysis", "intuitive, gut feeling and instinct"),
    "certainty_ambiguity": ("certainty, clear answers", "ambiguity, comfort with not knowing"),
    "agency_fatalism": ("agency, I shape my fate", "fatalism, fate/luck determines outcomes"),
    "human_tech_orientation": ("human-centered, technology serves people", "tech-oriented, technology leads progress"),
    "novelty_familiarity": ("novelty, seek new experiences", "familiarity, prefer the known"),
    "abstraction_concreteness": ("abstraction, theory and models", "concreteness, practical and tangible"),
    "systems_personalism": ("systems thinking, see structures", "personalism, focus on individuals"),
    # moral (Haidt's Moral Foundations — each is an independent emphasis)
    "care_emphasis": ("high care emphasis, compassion matters", "low care emphasis, toughness valued"),
    "fairness_emphasis": ("high fairness emphasis, justice matters", "low fairness emphasis, pragmatism over fairness"),
    "loyalty_emphasis": ("high loyalty emphasis, in-group loyalty", "low loyalty emphasis, universal loyalty"),
    "authority_emphasis": ("high authority emphasis, respect hierarchy", "low authority emphasis, question authority"),
    "sanctity_emphasis": ("high sanctity emphasis, purity matters", "low sanctity emphasis, nothing is sacred"),
    "liberty_emphasis": ("high liberty emphasis, resist oppression", "low liberty emphasis, accept constraints"),
    # religion
    "religiosity": ("religious, faith is central", "secular, religion is irrelevant"),
    "spirituality": ("spiritual, inner transcendence", "non-spiritual, materialist worldview"),
    "institutional_religion_affinity": ("institutional religion, organized worship", "non-institutional, personal practice"),
    "doctrinal_certainty": ("doctrinally certain, scripture is truth", "doctrinally flexible, open to interpretation"),
    "transcendence_orientation": ("transcendence-oriented, beyond material", "immanence-oriented, this-worldly"),
    "afterlife_orientation": ("afterlife-oriented, eternal matters", "present-oriented, this life matters"),
    "ritual_formality": ("formal ritual, liturgy and ceremony", "informal, spontaneous worship"),
    "sacred_boundary": ("strong sacred boundary, holy vs profane", "weak sacred boundary, all is ordinary"),
    "spirit_world_belief": ("believes in spirit world", "no spirit world, naturalism"),
    "exclusivism_pluralism": ("religious exclusivism, one true faith", "religious pluralism, many paths"),
    # doctype (document character — unipolar emphasis, like moral foundations)
    "decision_emphasis": ("decision, strategic choice with rationale, why X over Y", "not a decision, no choice being made"),
    "pattern_emphasis": ("pattern, recurring behavior or regularity across situations", "not a pattern, one-off occurrence"),
    "preference_emphasis": ("preference, user preference or setting, likes/dislikes", "not a preference, no personal taste"),
    "style_emphasis": ("style, coding or writing style, conventions and approach", "not about style, no stylistic content"),
    "habit_emphasis": ("habit, regular practice or workflow, routine behavior", "not a habit, no routine described"),
    "insight_emphasis": ("insight, discovery or learned understanding, a-ha moment", "not an insight, no new understanding"),
    "context_emphasis": ("context, environmental or situational background info", "not context, not background information"),
    "factual_emphasis": ("factual record, what happened, raw data, concrete event", "not factual, abstract or speculative"),
}

LENS_PREFIX = "lens_"      # User node: lens_culture_individualism_collectivism, ...
LOR_PREFIX = "lor_"        # Memory node: lor_culture_individualism_collectivism, ...

CATEGORY_KEYS: List[str] = list(LENS_CATEGORIES.keys())

NUM_CONCEPTS = sum(len(v) for v in LENS_CATEGORIES.values())  # 56

ALL_CONCEPTS: List[str] = []
for _concepts in LENS_CATEGORIES.values():
    ALL_CONCEPTS.extend(_concepts)

CONCEPT_LOCATION: Dict[str, Tuple[str, int]] = {}
for _cat, _concepts in LENS_CATEGORIES.items():
    for _i, _name in enumerate(_concepts):
        CONCEPT_LOCATION[_name] = (_cat, _i)

PRIORITY_CONCEPTS: List[str] = [
    "individualism_collectivism",
    "hierarchy_egalitarian",
    "uncertainty_tolerance",
    "tradition_change",
    "liberty_authority",
    "pluralism_orthodoxy",
    "institutional_trust",
    "gender_role_traditionalism",
    "sexual_permissiveness",
    "market_redistribution",
    "growth_sustainability",
    "consumerism_minimalism",
    "empiricism_revelation",
    "rational_intuitive",
    "agency_fatalism",
    "novelty_familiarity",
    "abstraction_concreteness",
    "religiosity",
    "spirituality",
    "afterlife_orientation",
]

# ---------------------------------------------------------------------------
# Named property helpers (v5: one property per concept)
# ---------------------------------------------------------------------------


def lor_concept_property(concept: str) -> str:
    """Named lor property for a single concept.

    e.g. 'individualism_collectivism' -> 'lor_culture_individualism_collectivism'
    """
    cat, _idx = CONCEPT_LOCATION[concept]
    return f"{LOR_PREFIX}{cat}_{concept}"


def lens_concept_property(concept: str) -> str:
    """Named lens property for a single concept.

    e.g. 'individualism_collectivism' -> 'lens_culture_individualism_collectivism'
    """
    cat, _idx = CONCEPT_LOCATION[concept]
    return f"{LENS_PREFIX}{cat}_{concept}"


# Pre-built lookup: concept_name -> lor property name
ALL_LOR_PROPERTIES: Dict[str, str] = {
    concept: lor_concept_property(concept) for concept in ALL_CONCEPTS
}

# Pre-built lookup: concept_name -> lens property name
ALL_LENS_PROPERTIES: Dict[str, str] = {
    concept: lens_concept_property(concept) for concept in ALL_CONCEPTS
}

# Flat list of all lor property names (for schema discovery, needs_scoring, etc.)
LOR_PROPERTY_NAMES: List[str] = list(ALL_LOR_PROPERTIES.values())
LENS_PROPERTY_NAMES: List[str] = list(ALL_LENS_PROPERTIES.values())


def lor_properties_for_category(category: str) -> List[Tuple[str, str]]:
    """Return [(concept_name, property_name), ...] for a category.

    e.g. lor_properties_for_category('culture') ->
         [('individualism_collectivism', 'lor_culture_individualism_collectivism'), ...]
    """
    return [
        (concept, ALL_LOR_PROPERTIES[concept])
        for concept in LENS_CATEGORIES[category]
    ]


def lens_properties_for_category(category: str) -> List[Tuple[str, str]]:
    """Return [(concept_name, property_name), ...] for a category."""
    return [
        (concept, ALL_LENS_PROPERTIES[concept])
        for concept in LENS_CATEGORIES[category]
    ]


# ---------------------------------------------------------------------------
# Default values (named property format)
# ---------------------------------------------------------------------------


def make_default_lens() -> Dict[str, List[float]]:
    """All [1,1] = uniform prior. No effect on scoring (skipped).

    Returns {property_name: [1.0, 1.0], ...} for all 56 concepts.
    """
    return {prop: [1.0, 1.0] for prop in LENS_PROPERTY_NAMES}


def make_default_lor() -> Dict[str, float]:
    """All 0.0 = neutral lor. {lor_culture_individualism_collectivism: 0.0, ...}."""
    return {prop: 0.0 for prop in LOR_PROPERTY_NAMES}


# ---------------------------------------------------------------------------
# Legacy helpers (category-level property names — for migration/compat)
# ---------------------------------------------------------------------------


def lens_property_name(category: str) -> str:
    """LEGACY: e.g. 'culture' -> 'lens_culture'. Use lens_concept_property() instead."""
    return f"{LENS_PREFIX}{category}"


def lor_property_name(category: str) -> str:
    """LEGACY: e.g. 'culture' -> 'lor_culture'. Use lor_concept_property() instead."""
    return f"{LOR_PREFIX}{category}"


def res_property_name(category: str) -> str:
    """Deprecated: use lor_property_name. Maps to lor_ prefix."""
    return lor_property_name(category)
