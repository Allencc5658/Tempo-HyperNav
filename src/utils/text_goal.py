import re

from configs.categories import name2index


def format_text_goal_description(text_goal, intrinsic_only=False):
    """Normalize TextNav/description-goal records into one text contract."""
    if isinstance(text_goal, dict):
        if intrinsic_only:
            return str(text_goal.get('intrinsic_attributes', '') or '').strip()
        parts = []
        for key in ('intrinsic_attributes', 'extrinsic_attributes'):
            value = text_goal.get(key)
            if value:
                parts.append(str(value).strip())
        if parts:
            return ' '.join(parts)
    return str(text_goal or '').strip()


def _canonical_text(text):
    text = str(text or '').lower().replace('_', ' ').replace('-', ' ')
    text = re.sub(r'[^a-z0-9 ]+', ' ', text)
    return ' '.join(text.split())


def infer_text_goal_category_id(text):
    """Infer the closed-set detector channel requested by a text goal."""
    text = _canonical_text(text)
    if not text:
        return None

    aliases = [
        ('tv_monitor', ('tv monitor', 'television', 'tv')),
        ('potted plant', ('potted plant', 'plant')),
        ('sofa', ('sofa', 'couch')),
        ('chair', ('chair',)),
        ('toilet', ('toilet',)),
        ('bed', ('bed',)),
    ]
    matches = []
    for label, terms in aliases:
        for term in terms:
            match = re.search(rf'\b{re.escape(term)}\b', text)
            if match:
                matches.append((match.start(), label))
                break
    if not matches:
        return None
    _, label = min(matches, key=lambda item: item[0])
    return name2index.get(label)
