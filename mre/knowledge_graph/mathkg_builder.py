"""
mre.knowledge_graph.mathkg_builder
────────────────────────────────────
Build a real mathematical knowledge graph from ProofWiki.

Source priority
---------------
1. Pre-uploaded Kaggle dataset (/kaggle/input/proofwiki-dump/latest.xml.gz)
2. Auto-download XML dump from proofwiki.org/xmldump/latest.xml.gz
3. MediaWiki API with correct category names (Category:Proven Results, etc.)
4. Synthetic fallback — always works offline

Usage
-----
    builder = MathKGBuilder(data_dir='data/mathkg', max_entities=3000)
    builder.crawl_entities()
    builder.extract_relations()
    builder.filter_and_clean()
    builder.make_splits()
    builder.save()
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import random
import re
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

from mre.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    entity_id:     int
    name:          str
    entity_type:   str
    description:   str = ""
    proofwiki_url: str = ""


@dataclass
class RawTriple:
    head:       str
    relation:   str
    tail:       str
    confidence: float = 1.0
    source:     str   = ""


# ─────────────────────────────────────────────────────────────────────────────
# ProofWiki API helpers
# ─────────────────────────────────────────────────────────────────────────────

PROOFWIKI_API      = "https://proofwiki.org/w/api.php"
PROOFWIKI_DUMP_URL = "https://proofwiki.org/xmldump/latest.xml.gz"
HEADERS = {
    "User-Agent": "MathKG-Builder/1.0 (research)",
    "Accept-Encoding": "gzip",
}

CATEGORIES_TO_CRAWL = [
    ("Proven Results", "theorem",    0),
    ("Definitions",    "definition", 106),
    ("Axioms",         "axiom",      108),
    ("Lemmas",         "lemma",      0),
    ("Corollaries",    "corollary",  0),
]

TARGET_RELATIONS = [
    "depends_on", "generalizes", "equivalent_to", "applied_in", "corollary_of"
]

_last_req: float = 0.0


def _api_get(params: dict, retries: int = 3) -> dict:
    global _last_req
    wait = 0.35 - (time.time() - _last_req)
    if wait > 0:
        time.sleep(wait)
    for attempt in range(retries):
        try:
            r = requests.get(PROOFWIKI_API, params=params,
                             headers=HEADERS, timeout=25)
            r.raise_for_status()
            _last_req = time.time()
            return r.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def _api_category_members(category: str, namespace: int = 0, max_pages: int = 2000) -> list:
    pages, params = [], {
        "action": "query", "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmnamespace": namespace, "cmlimit": 500, "format": "json",
    }
    while len(pages) < max_pages:
        data = _api_get(params)
        pages.extend(data.get("query", {}).get("categorymembers", []))
        cont = data.get("continue", {})
        if not cont:
            break
        params.update(cont)
    return pages[:max_pages]


# ─────────────────────────────────────────────────────────────────────────────
# Relation extractor
# ─────────────────────────────────────────────────────────────────────────────

def _links_in_section(wikitext_section: str) -> List[str]:
    targets = []
    for m in re.finditer(r"\[\[([^\]|#\n]+)(?:\|[^\]]+)?\]\]", wikitext_section):
        target = m.group(1).strip().split("#")[0].strip()
        if target:
            targets.append(target)
    return targets


def _split_sections(wikitext: str) -> Dict[str, str]:
    sections: Dict[str, str] = {"__lead__": ""}
    current, buf = "__lead__", []
    for line in wikitext.split("\n"):
        m = re.match(r"^(={2,4})\s*(.+?)\s*\1\s*$", line)
        if m:
            sections[current] = "\n".join(buf)
            current = m.group(2).strip().lower()
            buf = []
        else:
            buf.append(line)
    sections[current] = "\n".join(buf)
    return sections


def extract_relations_from_wikitext(title: str, wikitext: str) -> List[RawTriple]:
    """Extract all (head, relation, tail) triples from ProofWiki wikitext."""
    if not wikitext:
        return []

    triples: List[RawTriple] = []
    sections = _split_sections(wikitext)

    def add(h, r, t, c, src):
        if h != t:
            triples.append(RawTriple(h, r, t, c, src))

    # Corollaries → generalizes / corollary_of
    for sec in ("corollaries", "corollary"):
        for tgt in _links_in_section(sections.get(sec, "")):
            add(title, "generalizes",  tgt,   0.95, "corollaries_section")
            add(tgt,   "corollary_of", title, 0.95, "corollaries_section")

    # Generalizations
    for sec in ("generalizations", "generalization", "general result"):
        for tgt in _links_in_section(sections.get(sec, "")):
            add(tgt,   "generalizes", title, 0.90, "generalizations_section")
            add(title, "generalizes", tgt,   0.90, "generalizations_section")

    # Special cases
    for sec in ("special cases", "special case", "particular cases"):
        for tgt in _links_in_section(sections.get(sec, "")):
            add(title, "generalizes", tgt, 0.90, "special_cases_section")

    # Also see → depends_on (weak)
    for sec in ("also see", "see also"):
        for tgt in _links_in_section(sections.get(sec, "")):
            add(title, "depends_on", tgt, 0.60, "also_see_section")

    # Proof links
    proof_body = "\n".join(
        body for sec, body in sections.items()
        if re.match(r"proof\b", sec, re.IGNORECASE)
    )
    for tgt in _links_in_section(proof_body):
        if tgt == title or tgt.startswith(("File:", "Category:", "Template:")):
            continue
        if tgt.startswith("Definition:"):
            add(tgt, "applied_in", title, 0.70, "proof_def_link")
        else:
            add(title, "depends_on", tgt, 0.75, "proof_link")

    # Equivalent statements
    for sec in ("equivalent statements", "equivalents", "equivalent forms",
                "also presented as", "also known as"):
        for tgt in _links_in_section(sections.get(sec, "")):
            add(title, "equivalent_to", tgt,   0.90, "equiv_section")
            add(tgt,   "equivalent_to", title, 0.90, "equiv_section")

    # In-text patterns
    body = (sections.get("theorem", "") + sections.get("statement", "") +
            sections.get("__lead__", ""))
    for m in re.finditer(r"is a (?:special|particular) case of\s+\[\[([^\]|#]+)", body, re.I):
        tgt = m.group(1).strip()
        add(tgt, "generalizes", title, 0.95, "text_special_case")
    for m in re.finditer(r"(?:generalizes|is a generalization of)\s+\[\[([^\]|#]+)", body, re.I):
        tgt = m.group(1).strip()
        add(title, "generalizes", tgt, 0.90, "text_generalizes")
    for m in re.finditer(r"if and only if[^.]{0,120}\[\[([^\]|#]+)", body, re.I):
        tgt = m.group(1).strip()
        add(title, "equivalent_to", tgt,   0.80, "text_iff")
        add(tgt,   "equivalent_to", title, 0.80, "text_iff")

    # Deduplicate — keep highest confidence per (h, rel, t)
    best: Dict[tuple, Tuple[float, str]] = {}
    for t in triples:
        key = (t.head, t.relation, t.tail)
        if key not in best or t.confidence > best[key][0]:
            best[key] = (t.confidence, t.source)
    return [
        RawTriple(k[0], k[1], k[2], v[0], v[1])
        for k, v in best.items()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_entities(n: int, seed: int = 42) -> Tuple[List[Entity], Dict[str, str]]:
    rng = random.Random(seed)
    topics  = ["Cauchy","Riemann","Banach","Hilbert","Fourier","Lagrange",
               "Gauss","Euler","Fermat","Pythagoras","Cantor","Zorn",
               "Bolzano","Weierstrass","Heine","Borel","Lebesgue","Baire"]
    objects = ["Theorem","Lemma","Corollary","Proposition","Definition",
               "Criterion","Inequality","Identity","Formula","Axiom"]
    areas   = ["Analysis","Algebra","Topology","Number Theory","Geometry",
               "Combinatorics","Probability","Functional Analysis","Set Theory"]
    etype_map = {
        "Theorem":"theorem","Lemma":"lemma","Corollary":"corollary",
        "Definition":"definition","Proposition":"theorem","Criterion":"theorem",
        "Inequality":"theorem","Identity":"theorem","Formula":"theorem","Axiom":"axiom",
    }
    entities = []
    seen_names: set = set()
    for i in range(n):
        topic  = rng.choice(topics)
        obj    = rng.choice(objects)
        area   = rng.choice(areas)
        suffix = f" ({area})" if rng.random() < 0.3 else ""
        base_name = f"{topic}'s {obj}{suffix}"
        # Guarantee uniqueness: append a numeric discriminator when name collides
        name = base_name
        disambig = 2
        while name in seen_names:
            name = f"{base_name} {disambig}"
            disambig += 1
        seen_names.add(name)
        entities.append(Entity(
            entity_id=i,
            name=name,
            entity_type=etype_map.get(obj, "theorem"),
        ))
    # wikitext_map keyed by name — now guaranteed collision-free
    wikitext_map = {e.name: "" for e in entities}
    assert len(wikitext_map) == n, f"wikitext_map collision: {len(wikitext_map)} != {n}"
    return entities, wikitext_map


def _generate_synthetic_triples(
    entities: List[Entity],
    edges_per_relation: int = 600,
    seed: int = 42,
) -> List[RawTriple]:
    rng = random.Random(seed)
    type_rules = {
        "depends_on":   ({"theorem","lemma","corollary"}, {"theorem","lemma","axiom","definition"}),
        "generalizes":  ({"theorem","lemma"}, {"theorem","lemma","corollary"}),
        "equivalent_to":({"theorem","lemma"}, {"theorem","lemma"}),
        "applied_in":   ({"theorem","lemma","definition"}, {"theorem"}),
        "corollary_of": ({"corollary","lemma"}, {"theorem","lemma"}),
    }
    type_index: Dict[str, List[int]] = defaultdict(list)
    for e in entities:
        type_index[e.entity_type].append(e.entity_id)

    triples: List[RawTriple] = []
    for rel in TARGET_RELATIONS:
        head_types, tail_types = type_rules.get(rel, ({"theorem"}, {"theorem"}))
        heads = [eid for t in head_types for eid in type_index.get(t, [])]
        tails = [eid for t in tail_types for eid in type_index.get(t, [])]
        if not heads or not tails:
            heads = tails = list(range(len(entities)))
        rng.shuffle(heads)
        count = 0
        for h_id in heads:
            if count >= edges_per_relation:
                break
            candidates = rng.choices(tails, k=20)
            for t_id in candidates:
                if t_id != h_id:
                    triples.append(RawTriple(
                        head=entities[h_id].name, relation=rel,
                        tail=entities[t_id].name,
                        confidence=round(rng.uniform(0.75, 1.0), 2),
                        source="synthetic",
                    ))
                    count += 1
                    break
    return triples


# ─────────────────────────────────────────────────────────────────────────────
# Main builder class
# ─────────────────────────────────────────────────────────────────────────────

class MathKGBuilder:
    """
    Orchestrates entity crawling, relation extraction, filtering, and saving.
    All intermediate state is stored on the instance so steps can be run
    individually (useful in a notebook).
    """

    def __init__(
        self,
        data_dir:    str = "data/mathkg",
        max_entities: int = 3000,
        seed:        int = 42,
        local_dump:  str = "/kaggle/input/proofwiki-dump/latest.xml.gz",
    ):
        self.data_dir     = Path(data_dir)
        self.max_entities = max_entities
        self.seed         = seed
        self.local_dump   = local_dump

        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "splits").mkdir(exist_ok=True)
        (self.data_dir / "cache").mkdir(exist_ok=True)

        self.entities:     List[Entity]    = []
        self.wikitext_map: Dict[str, str]  = {}
        self.raw_triples:  List[RawTriple] = []
        self.clean_entities: List[Entity]  = []
        self.clean_triples:  List[RawTriple] = []
        self.splits_data:  Dict[str, Dict[str, List[RawTriple]]] = {}

    # ── Step 1: entity collection ─────────────────────────────────────────────

    def crawl_entities(self) -> None:
        """Try XML dump → API → synthetic; populate self.entities and self.wikitext_map."""
        # 1. Local dump
        if Path(self.local_dump).exists():
            logger.info("Loading from local XML dump: %s", self.local_dump)
            ents, wmap = self._load_xml_dump(self.local_dump)
            if ents:
                self.entities, self.wikitext_map = ents, wmap
                return

        # 2. Download XML dump
        logger.info("Attempting XML dump download...")
        try:
            ents, wmap = self._load_xml_dump(PROOFWIKI_DUMP_URL)
            if ents:
                self.entities, self.wikitext_map = ents, wmap
                return
        except Exception as exc:
            logger.warning("XML download failed: %s", exc)

        # 3. API
        logger.info("Trying MediaWiki API...")
        try:
            ents, wmap = self._load_from_api()
            if ents:
                self.entities, self.wikitext_map = ents, wmap
                return
        except Exception as exc:
            logger.warning("API failed: %s", exc)

        # 4. Synthetic fallback
        logger.warning("All live sources unavailable — using synthetic entities.")
        self.entities, self.wikitext_map = _generate_synthetic_entities(
            self.max_entities, seed=self.seed
        )

    def _load_xml_dump(self, source: str) -> Tuple[List[Entity], Dict[str, str]]:
        import xml.etree.ElementTree as ET

        if source.startswith("http"):
            logger.info("Downloading XML dump (~80 MB)...")
            req = urllib.request.Request(
                source, headers={"User-Agent": HEADERS["User-Agent"]}
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = resp.read()
            raw = gzip.decompress(raw) if source.endswith(".gz") else raw
            src = io.BytesIO(raw)
        elif source.endswith(".gz"):
            src = gzip.open(source, "rb")
        else:
            src = open(source, "rb")

        NS = "{http://www.mediawiki.org/xml/export-0.11/}"
        skip_pfx = ("Talk:","User:","Help:","ProofWiki:","Template:",
                    "Category:","File:","MediaWiki:")

        def detect_type(title, text):
            if title.startswith("Definition:"): return "definition"
            if title.startswith("Axiom:"):      return "axiom"
            if "[[Category:Corollaries"  in text: return "corollary"
            if "[[Category:Lemmas"       in text: return "lemma"
            if "[[Category:Proven"       in text: return "theorem"
            if re.search(r"==\s*Proof\s*==", text): return "theorem"
            return None

        entities, wmap, eid = [], {}, 0
        try:
            for _, elem in ET.iterparse(src, events=("end",)):
                if elem.tag != f"{NS}page":
                    continue
                te = elem.find(f"{NS}title")
                xe = elem.find(f".//{NS}text")
                if te is None or xe is None:
                    elem.clear(); continue
                title = te.text or ""
                text  = xe.text or ""
                if any(title.startswith(p) for p in skip_pfx):
                    elem.clear(); continue
                if title.split(":")[-1].count("/") > 1:
                    elem.clear(); continue
                etype = detect_type(title, text)
                if etype is None:
                    elem.clear(); continue
                dm = re.search(
                    r"==\s*(?:Theorem|Definition|Statement)\s*==\s*([^=]{20,250})",
                    text, re.DOTALL,
                )
                desc = dm.group(1)[:200].strip().replace("\n", " ") if dm else ""
                entities.append(Entity(
                    entity_id=eid, name=title, entity_type=etype,
                    description=desc,
                    proofwiki_url=f"https://proofwiki.org/wiki/{title.replace(' ','_')}",
                ))
                wmap[title] = text
                eid += 1
                if eid % 2000 == 0:
                    logger.info("  %d entities parsed...", eid)
                if eid >= self.max_entities:
                    break
                elem.clear()
        finally:
            src.close()

        logger.info("XML dump: loaded %d entities", len(entities))
        return entities, wmap

    def _load_from_api(self) -> Tuple[List[Entity], Dict[str, str]]:
        seen, entities, wmap, eid = set(), [], {}, 0
        cache_dir = self.data_dir / "cache"

        for cat, etype, ns in CATEGORIES_TO_CRAWL:
            if eid >= self.max_entities:
                break
            want = self.max_entities - eid
            logger.info("API: Category:%s (ns=%d, want %d)...", cat, ns, want)
            try:
                pages = _api_category_members(cat, namespace=ns, max_pages=min(want, 1000))
            except Exception as exc:
                logger.warning("Category:%s failed: %s", cat, exc)
                continue

            for p in pages:
                title = p["title"]
                if title in seen:
                    continue
                seen.add(title)

                # Try cache first
                key  = hashlib.md5(title.encode()).hexdigest()
                path = cache_dir / f"{key}.txt"
                if path.exists():
                    text = path.read_text(encoding="utf-8")
                else:
                    text = ""
                    try:
                        data = _api_get({
                            "action":"query","titles":title,"prop":"revisions",
                            "rvprop":"content","rvslots":"main","format":"json",
                        })
                        for page in data.get("query",{}).get("pages",{}).values():
                            text = (page.get("revisions") or [{}])[0]\
                                       .get("slots",{}).get("main",{}).get("*","")
                        path.write_text(text, encoding="utf-8")
                    except Exception:
                        pass

                dm = re.search(
                    r"==\s*(?:Theorem|Definition|Statement)\s*==\s*([^=]{20,250})",
                    text, re.DOTALL,
                )
                desc = dm.group(1)[:200].strip().replace("\n"," ") if dm else ""
                entities.append(Entity(
                    entity_id=eid, name=title, entity_type=etype,
                    description=desc,
                    proofwiki_url=f"https://proofwiki.org/wiki/{title.replace(' ','_')}",
                ))
                wmap[title] = text
                eid += 1
                if eid >= self.max_entities:
                    break

        logger.info("API: loaded %d entities", len(entities))
        return entities, wmap

    # ── Step 2: relation extraction ───────────────────────────────────────────

    def extract_relations(self) -> None:
        title_to_id = {e.name: e.entity_id for e in self.entities}
        real_texts  = sum(1 for v in self.wikitext_map.values() if v.strip())

        if real_texts == 0:
            logger.info("No wikitext — generating synthetic triples.")
            self.raw_triples = _generate_synthetic_triples(self.entities, seed=self.seed)
            return

        all_triples: List[RawTriple] = []
        logger.info("Extracting relations from %d pages with wikitext...", real_texts)
        for entity in self.entities:
            wikitext = self.wikitext_map.get(entity.name, "")
            if not wikitext:
                continue
            try:
                for t in extract_relations_from_wikitext(entity.name, wikitext):
                    if t.tail in title_to_id and t.relation in TARGET_RELATIONS:
                        all_triples.append(t)
            except Exception:
                continue

        logger.info("Raw triples extracted: %d", len(all_triples))
        if len(all_triples) < 50:
            logger.warning("Too few real triples — supplementing with synthetic.")
            all_triples.extend(_generate_synthetic_triples(self.entities, seed=self.seed))
        self.raw_triples = all_triples

    # ── Step 3: filter & clean ────────────────────────────────────────────────

    def filter_and_clean(
        self,
        conf_threshold: float = 0.65,
        min_degree:     int   = 1,
    ) -> None:
        title_to_id = {e.name: e.entity_id for e in self.entities}

        # Convert to id-based, drop self-loops and low-confidence
        id_triples = []
        for t in self.raw_triples:
            h = title_to_id.get(t.head)
            tl = title_to_id.get(t.tail)
            if h is None or tl is None or h == tl:
                continue
            if t.confidence < conf_threshold:
                continue
            id_triples.append((h, t.relation, tl, t.confidence, t.source))

        # Deduplicate
        best: Dict[tuple, Tuple[float, str]] = {}
        for h, rel, tl, conf, src in id_triples:
            key = (h, rel, tl)
            if key not in best or conf > best[key][0]:
                best[key] = (conf, src)

        deduped = [
            RawTriple(self.entities[k[0]].name, k[1], self.entities[k[2]].name, v[0], v[1])
            for k, v in best.items()
        ]

        # Enforce symmetry for equivalent_to
        existing = {(t.head, t.relation, t.tail) for t in deduped}
        sym = []
        for t in deduped:
            sym.append(t)
            if t.relation == "equivalent_to":
                rev = (t.tail, "equivalent_to", t.head)
                if rev not in existing:
                    sym.append(RawTriple(t.tail, "equivalent_to", t.head, t.confidence, "symmetry"))
        deduped = sym

        # Entity degree filter
        degree: Dict[str, int] = defaultdict(int)
        for t in deduped:
            degree[t.head] += 1
            degree[t.tail] += 1

        active = {n for n, d in degree.items() if d >= min_degree}
        kept   = [e for e in self.entities
                  if e.name in active or e.entity_type in ("axiom", "definition")]
        new_id = {e.name: i for i, e in enumerate(kept)}
        for i, e in enumerate(kept):
            e.entity_id = i

        final = [t for t in deduped if t.head in new_id and t.tail in new_id]
        self.clean_entities = kept
        self.clean_triples  = final
        logger.info("After filtering: %d entities, %d triples", len(kept), len(final))

    # ── Step 4: splits ────────────────────────────────────────────────────────

    def make_splits(self, train_ratio: float = 0.70, val_ratio: float = 0.15) -> None:
        rng = random.Random(self.seed)
        self.splits_data = {}
        for rel in TARGET_RELATIONS:
            rel_triples = [t for t in self.clean_triples if t.relation == rel]
            rng.shuffle(rel_triples)
            n = len(rel_triples)
            n_train = int(n * train_ratio)
            n_val   = int(n * val_ratio)
            self.splits_data[rel] = {
                "train": rel_triples[:n_train],
                "val":   rel_triples[n_train : n_train + n_val],
                "test":  rel_triples[n_train + n_val :],
            }

    # ── Step 5: save ──────────────────────────────────────────────────────────

    def save(self) -> None:
        import pandas as pd

        new_id = {e.name: e.entity_id for e in self.clean_entities}

        # entities.tsv
        ent_df = pd.DataFrame([
            {"entity_id": e.entity_id, "name": e.name,
             "type": e.entity_type, "description": e.description,
             "proofwiki_url": e.proofwiki_url}
            for e in self.clean_entities
        ])
        ent_df.to_csv(self.data_dir / "entities.tsv", sep="\t", index=False)

        # relations.tsv
        rows = []
        for t in self.clean_triples:
            h = new_id.get(t.head)
            tl = new_id.get(t.tail)
            if h is not None and tl is not None:
                rows.append({"head_id": h, "head_name": t.head,
                             "relation": t.relation,
                             "tail_id": tl, "tail_name": t.tail,
                             "confidence": t.confidence, "source": t.source})
        rel_df = pd.DataFrame(rows)
        rel_df.to_csv(self.data_dir / "relations.tsv", sep="\t", index=False)

        # Split files
        split_summary: Dict[str, Dict[str, int]] = {}
        for rel in TARGET_RELATIONS:
            for split_name in ("train", "val", "test"):
                split_rows = []
                for t in self.splits_data.get(rel, {}).get(split_name, []):
                    h = new_id.get(t.head)
                    tl = new_id.get(t.tail)
                    if h is not None and tl is not None:
                        split_rows.append({"head_id": h, "head_name": t.head,
                                           "relation": t.relation,
                                           "tail_id": tl, "tail_name": t.tail,
                                           "confidence": t.confidence, "source": t.source})
                pd.DataFrame(split_rows).to_csv(
                    self.data_dir / "splits" / f"{rel}_{split_name}.tsv",
                    sep="\t", index=False,
                )
            split_summary[rel] = {
                s: len(self.splits_data.get(rel, {}).get(s, []))
                for s in ("train", "val", "test")
            }

        # stats.json
        from collections import Counter
        rel_counts = Counter(t.relation for t in self.clean_triples)
        stats = {
            "n_entities": len(self.clean_entities),
            "n_triples":  len(self.clean_triples),
            "relations":  TARGET_RELATIONS,
            "splits":     split_summary,
            "per_relation_counts": {r: rel_counts[r] for r in TARGET_RELATIONS},
            "entity_type_counts":  dict(Counter(e.entity_type for e in self.clean_entities)),
            "seed": self.seed,
        }
        with open(self.data_dir / "stats.json", "w") as fh:
            json.dump(stats, fh, indent=2)

        logger.info("MathKG saved to %s", self.data_dir)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        from collections import Counter
        rel_counts = Counter(t.relation for t in self.clean_triples)
        lines = [f"MathKGBuilder (entities={len(self.clean_entities)}, triples={len(self.clean_triples)})"]
        for rel in TARGET_RELATIONS:
            cnt  = rel_counts[rel]
            sp   = self.splits_data.get(rel, {})
            n_tr = len(sp.get("train", []))
            n_v  = len(sp.get("val",   []))
            n_te = len(sp.get("test",  []))
            lines.append(f"  {rel:20s}: {cnt:4d} total  train={n_tr}  val={n_v}  test={n_te}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset_stats(builder: MathKGBuilder) -> "plt.Figure":  # type: ignore
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from collections import Counter

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#0f0f1a")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)
    PALETTE = ["#FF6B6B","#FFD93D","#6BCB77","#4FC3F7","#C77DFF","#FF9FF3"]

    def dark_ax(ax):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.grid(True, alpha=0.15, color="white")

    rel_counts = Counter(t.relation for t in builder.clean_triples)

    # Plot 1: triples per relation
    ax1 = fig.add_subplot(gs[0, 0])
    dark_ax(ax1)
    rels   = [r for r in TARGET_RELATIONS if rel_counts[r] > 0]
    counts = [rel_counts[r] for r in rels]
    bars = ax1.barh(rels, counts, color=PALETTE[:len(rels)], alpha=0.85)
    for bar, cnt in zip(bars, counts):
        ax1.text(cnt + 1, bar.get_y() + bar.get_height()/2,
                 f"{cnt:,}", va="center", color="white", fontsize=9)
    ax1.set_xlabel("# triples")
    ax1.set_title("Triples per Relation", fontweight="bold")

    # Plot 2: entity type pie
    ax2 = fig.add_subplot(gs[0, 1])
    dark_ax(ax2)
    tc = Counter(e.entity_type for e in builder.clean_entities)
    ax2.pie(list(tc.values()), labels=list(tc.keys()),
            colors=PALETTE[:len(tc)], autopct="%1.1f%%",
            textprops={"color":"white"},
            wedgeprops={"linewidth":0.5,"edgecolor":"#0f0f1a"})
    ax2.set_title("Entity Type Distribution", fontweight="bold")

    # Plot 3: degree distribution
    ax3 = fig.add_subplot(gs[0, 2])
    dark_ax(ax3)
    degree: Counter = Counter()
    for t in builder.clean_triples:
        degree[t.head] += 1
        degree[t.tail] += 1
    if degree:
        deg_hist = Counter(degree.values())
        dx, dy = zip(*sorted(deg_hist.items()))
        ax3.loglog(dx, dy, "o", color="#FFD93D", markersize=4, alpha=0.7)
        ax3.set_xlabel("Degree (log)")
        ax3.set_ylabel("# Entities (log)")
        ax3.set_title("Degree Distribution", fontweight="bold")

    # Plot 4: confidence histogram
    ax4 = fig.add_subplot(gs[1, 0])
    dark_ax(ax4)
    confs = [t.confidence for t in builder.clean_triples]
    if confs:
        ax4.hist(confs, bins=20, color="#6BCB77", alpha=0.85, edgecolor="#0f0f1a")
        ax4.axvline(np.mean(confs), color="#FFD93D", linestyle="--",
                    label=f"mean={np.mean(confs):.2f}")
        ax4.set_xlabel("Confidence score")
        ax4.set_ylabel("# Triples")
        ax4.set_title("Confidence Distribution", fontweight="bold")
        ax4.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    # Plot 5: split sizes
    ax5 = fig.add_subplot(gs[1, 1])
    dark_ax(ax5)
    split_rels = [r for r in TARGET_RELATIONS if builder.splits_data.get(r, {}).get("train")]
    if split_rels:
        x = np.arange(len(split_rels))
        width = 0.28
        for i, (sn, col) in enumerate(zip(["train","val","test"],
                                          ["#6BCB77","#FFD93D","#FF6B6B"])):
            vals = [len(builder.splits_data[r][sn]) for r in split_rels]
            ax5.bar(x + i*width, vals, width, label=sn, color=col, alpha=0.85)
        ax5.set_xticks(x + width)
        ax5.set_xticklabels([r.replace("_","\n") for r in split_rels], fontsize=7)
        ax5.set_ylabel("# Triples")
        ax5.set_title("Split Sizes per Relation", fontweight="bold")
        ax5.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    # Plot 6: summary box
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#1a1a2e")
    ax6.axis("off")
    avg_deg = np.mean(list(degree.values())) if degree else 0
    avg_conf = np.mean(confs) if confs else 0
    summary = "\n".join([
        "  MathKG Summary",
        "  " + "─" * 22,
        f"  Entities    : {len(builder.clean_entities):,}",
        f"  Triples     : {len(builder.clean_triples):,}",
        f"  Relations   : {len(TARGET_RELATIONS)}",
        f"  Avg degree  : {avg_deg:.1f}",
        f"  Avg conf    : {avg_conf:.2f}",
        "",
        "  Source: ProofWiki.org",
        "  Ready for NRO pipeline ✓",
    ])
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             va="top", ha="left", fontsize=11, color="white",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                       edgecolor="#6BCB77", linewidth=1.5))

    fig.suptitle("MathKG Dataset Statistics", color="white",
                 fontsize=15, fontweight="bold", y=0.98)
    return fig
