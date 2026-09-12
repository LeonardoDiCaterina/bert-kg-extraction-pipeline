import re
from typing import List, Dict, Set, Tuple


def is_informative_chunk(text: str, min_words: int = 50) -> bool:
    """
    Filters out empty tables, legal boilerplate, and short snippets from SEC filings.
    """
    words = text.split()
    if len(words) < min_words:
        return False
    # Drop checkbox-heavy administrative blocks
    if "indicate by check mark" in text.lower():
        return False
    return True


def extract_rebel_triplets(text: str) -> List[Dict[str, str]]:
    """
    Parses REBEL-style formatted triplet strings into head/type/tail dictionaries.
    Format: <s> <triplet> subject <subj> object <obj> relation </s>
    """
    triplets: List[Dict[str, str]] = []
    relation, subject, object_ = "", "", ""
    text = text.strip()
    current = "x"

    for token in (
        text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split()
    ):
        if token == "<triplet>":
            current = "t"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token

    if subject != "" and relation != "" and object_ != "":
        triplets.append(
            {"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()}
        )

    return triplets


def parse_triplet_string(text: str) -> Set[Tuple[str, str, str]]:
    """
    Uses regex to extract (subject, relation, object) tuples from structured output strings.
    Format: <triplet> subject <subj_type> TYPE <relation> REL <obj> object <obj_type> TYPE
    """
    triplets: Set[Tuple[str, str, str]] = set()
    pattern = r"<triplet>\s*(.*?)\s*<subj_type>.*?<relation>\s*(.*?)\s*<obj>\s*(.*?)\s*<obj_type>"
    matches = re.findall(pattern, text)

    for sub, rel, obj in matches:
        if sub and rel and obj:
            triplets.add((sub.strip(), rel.strip(), obj.strip()))

    return triplets


def clean_json_string(raw_text: str) -> str:
    """
    Strips markdown code fences (e.g. ```json ... ```) and leading/trailing whitespace
    to prepare an LLM response for json.loads.
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()
