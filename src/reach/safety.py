"""Deterministic refusal for sharp and hot tabletop requests."""

import re

BLOCKED_TERMS = (
    'knife', 'knives', 'blade', 'razor', 'scissors', 'needle', 'cutter', 'saw',
    'glass shard', 'broken glass', 'lighter', 'match', 'matches', 'candle',
    'stove', 'oven', 'pan', 'pot', 'kettle', 'iron', 'hot', 'boiling', 'burner', 'grill',
)
REFUSAL = "I won't guide your hand to sharp or hot objects."


def blocked_term(phrase, cfg=None):
    words = re.findall(r'\w+', (phrase or '').casefold())
    terms = (cfg or {}).get('safety', {}).get('blocked_terms', BLOCKED_TERMS)
    for term in terms:
        tokens = re.findall(r'\w+', str(term).casefold())
        if tokens and any(words[i:i+len(tokens)] == tokens for i in range(len(words)-len(tokens)+1)):
            return str(term)
    return None
