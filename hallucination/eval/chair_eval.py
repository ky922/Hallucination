"""
CHAIR (Caption Hallucination Assessment with Image Relevance) evaluation.

Metrics:
  CHAIRs — fraction of captions that mention at least one hallucinated object
  CHAIRi — fraction of all mentioned objects that are hallucinated

Pipeline per caption:
  1. Extract nouns via NLTK POS tagging
  2. Lemmatise and match against COCO synonym dict → set of COCO categories
  3. Compare with GT COCO annotations → hallucinated = mentioned ∖ GT
"""

import json
import os
from typing import Dict, List, Set

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


def _ensure_nltk() -> None:
    """Download NLTK resources on first run (silent if already present)."""
    for pkg in ("punkt", "punkt_tab", "averaged_perceptron_tagger",
                "averaged_perceptron_tagger_eng", "wordnet"):
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg, quiet=True)


# ── COCO synonym mapping ───────────────────────────────────────────────────────
# Maps every COCO category to a list of surface forms (lower-case).
# Covers all 80 COCO object categories.

COCO_SYNONYMS: Dict[str, List[str]] = {
    "person":        ["person", "people", "man", "men", "woman", "women", "child",
                      "children", "boy", "girl", "human", "individual", "lady",
                      "gentleman", "kid", "baby", "infant", "toddler", "pedestrian",
                      "player", "skater", "surfer", "skier", "rider", "cyclist",
                      "biker", "athlete", "crowd"],
    "bicycle":       ["bicycle", "bike", "cycle"],
    "car":           ["car", "automobile", "vehicle", "sedan", "suv", "taxi", "cab",
                      "coupe"],
    "motorcycle":    ["motorcycle", "motorbike", "moped", "scooter"],
    "airplane":      ["airplane", "plane", "aircraft", "jet", "airliner"],
    "bus":           ["bus", "coach", "minibus"],
    "train":         ["train", "locomotive"],
    "truck":         ["truck", "lorry", "van", "pickup"],
    "boat":          ["boat", "ship", "vessel", "sailboat", "canoe", "kayak",
                      "yacht", "ferry"],
    "traffic light": ["traffic light", "stoplight", "signal light"],
    "fire hydrant":  ["fire hydrant", "hydrant"],
    "stop sign":     ["stop sign"],
    "parking meter": ["parking meter"],
    "bench":         ["bench"],
    "bird":          ["bird", "pigeon", "seagull", "duck", "eagle", "dove",
                      "parrot", "sparrow", "crow", "robin", "pelican", "swan",
                      "owl", "hawk", "chicken", "rooster"],
    "cat":           ["cat", "kitten", "feline", "kitty", "tabby"],
    "dog":           ["dog", "puppy", "canine", "hound", "poodle", "labrador",
                      "retriever"],
    "horse":         ["horse", "pony", "stallion", "mare", "foal"],
    "sheep":         ["sheep", "lamb", "ram", "ewe"],
    "cow":           ["cow", "cattle", "bull", "calf", "ox", "buffalo", "steer"],
    "elephant":      ["elephant"],
    "bear":          ["bear", "grizzly"],
    "zebra":         ["zebra"],
    "giraffe":       ["giraffe"],
    "backpack":      ["backpack", "bag", "rucksack", "knapsack", "bookbag",
                      "satchel"],
    "umbrella":      ["umbrella", "parasol"],
    "handbag":       ["handbag", "purse", "clutch"],
    "tie":           ["tie", "necktie"],
    "suitcase":      ["suitcase", "luggage", "baggage"],
    "frisbee":       ["frisbee", "disc"],
    "skis":          ["ski", "skis"],
    "snowboard":     ["snowboard"],
    "sports ball":   ["ball", "football", "soccer ball", "basketball", "baseball",
                      "tennis ball", "golf ball", "volleyball"],
    "kite":          ["kite"],
    "baseball bat":  ["baseball bat", "bat"],
    "baseball glove":["baseball glove", "glove", "mitt"],
    "skateboard":    ["skateboard"],
    "surfboard":     ["surfboard"],
    "tennis racket": ["tennis racket", "racket", "racquet"],
    "bottle":        ["bottle", "jar", "flask", "jug"],
    "wine glass":    ["wine glass", "wineglass", "goblet"],
    "cup":           ["cup", "mug"],
    "fork":          ["fork"],
    "knife":         ["knife"],
    "spoon":         ["spoon"],
    "bowl":          ["bowl", "dish"],
    "banana":        ["banana"],
    "apple":         ["apple"],
    "sandwich":      ["sandwich", "burger", "sub", "wrap"],
    "orange":        ["orange", "tangerine", "mandarin"],
    "broccoli":      ["broccoli"],
    "carrot":        ["carrot"],
    "hot dog":       ["hot dog", "hotdog", "sausage", "wiener"],
    "pizza":         ["pizza"],
    "donut":         ["donut", "doughnut"],
    "cake":          ["cake"],
    "chair":         ["chair", "stool"],
    "couch":         ["couch", "sofa", "settee", "loveseat"],
    "potted plant":  ["potted plant", "plant", "flower pot", "flowerpot",
                      "cactus", "succulent"],
    "bed":           ["bed", "bunk", "mattress", "crib"],
    "dining table":  ["table", "dining table", "desk", "counter"],
    "toilet":        ["toilet", "commode", "lavatory"],
    "tv":            ["tv", "television", "monitor", "screen", "display"],
    "laptop":        ["laptop", "notebook", "computer", "macbook"],
    "mouse":         ["mouse"],
    "remote":        ["remote", "remote control"],
    "keyboard":      ["keyboard"],
    "cell phone":    ["phone", "cell phone", "smartphone", "mobile", "iphone"],
    "microwave":     ["microwave"],
    "oven":          ["oven", "stove", "range"],
    "toaster":       ["toaster"],
    "sink":          ["sink", "basin"],
    "refrigerator":  ["refrigerator", "fridge"],
    "book":          ["book", "novel", "textbook", "magazine"],
    "clock":         ["clock", "watch"],
    "vase":          ["vase"],
    "scissors":      ["scissors", "shears"],
    "teddy bear":    ["teddy bear", "teddy", "stuffed animal", "plush"],
    "hair drier":    ["hair drier", "hair dryer", "hairdryer"],
    "toothbrush":    ["toothbrush"],
}

# Reverse map: surface form → canonical COCO category name
_WORD_TO_CAT: Dict[str, str] = {}
for _cat, _syns in COCO_SYNONYMS.items():
    for _s in _syns:
        _WORD_TO_CAT[_s.lower()] = _cat

# Multi-word phrases sorted longest-first for greedy prefix matching
_MULTI_WORD_PHRASES = sorted(
    [s for s in _WORD_TO_CAT if " " in s], key=len, reverse=True
)


# ── COCO annotation loading ────────────────────────────────────────────────────

def load_coco_annotations(annotation_file: str) -> Dict[int, Set[str]]:
    """
    Load COCO instances_val2014.json.

    Returns:
        Dict mapping image_id (int) → set of COCO category names present.
    """
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(
            f"COCO annotation file not found: {annotation_file}\n"
            "Run download_data.sh first."
        )
    with open(annotation_file) as f:
        data = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    image_to_cats: Dict[int, Set[str]] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        image_to_cats.setdefault(img_id, set()).add(
            cat_id_to_name[ann["category_id"]]
        )
    return image_to_cats


# ── Object extraction ──────────────────────────────────────────────────────────

def extract_objects(caption: str, lemmatizer: WordNetLemmatizer) -> Set[str]:
    """
    Extract COCO category names mentioned in *caption*.

    Strategy:
      1. Greedy multi-word phrase matching (longest first)
      2. NLTK POS-tag noun extraction + lemmatisation for single words
    """
    _ensure_nltk()
    caption_lc = caption.lower()
    found: Set[str] = set()

    # Pass 1: multi-word phrases
    working = caption_lc
    for phrase in _MULTI_WORD_PHRASES:
        if phrase in working:
            found.add(_WORD_TO_CAT[phrase])
            working = working.replace(phrase, " ")  # avoid double-counting

    # Pass 2: single-word nouns via POS tagging
    tokens = word_tokenize(working)
    tagged = pos_tag(tokens)
    for word, tag in tagged:
        if not tag.startswith("NN"):
            continue
        # try raw form
        if word in _WORD_TO_CAT:
            found.add(_WORD_TO_CAT[word])
            continue
        # try lemmatised noun form
        lemma = lemmatizer.lemmatize(word, pos="n")
        if lemma in _WORD_TO_CAT:
            found.add(_WORD_TO_CAT[lemma])

    return found


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_chair(
    captions: List[str],
    image_ids: List[int],
    gt_annotations: Dict[int, Set[str]],
) -> Dict:
    """
    Compute CHAIRs and CHAIRi over a set of generated captions.

    CHAIRs = |captions with ≥1 hallucinated object| / |captions|
    CHAIRi = |hallucinated object mentions| / |all object mentions|

    Args:
        captions:       generated captions (one per image)
        image_ids:      corresponding COCO image IDs
        gt_annotations: output of load_coco_annotations()

    Returns:
        Dict with CHAIRs, CHAIRi, and supporting counts.
    """
    _ensure_nltk()
    lemmatizer = WordNetLemmatizer()

    n_caps              = len(captions)
    caps_with_halluc    = 0
    total_mentioned     = 0
    total_hallucinated  = 0

    for caption, img_id in zip(captions, image_ids):
        gt   = gt_annotations.get(img_id, set())
        pred = extract_objects(caption, lemmatizer)
        hall = pred - gt

        total_mentioned    += len(pred)
        total_hallucinated += len(hall)
        if hall:
            caps_with_halluc += 1

    chairs = caps_with_halluc   / n_caps          if n_caps         else 0.0
    chairi = total_hallucinated / total_mentioned if total_mentioned else 0.0

    return {
        "CHAIRs":                    round(chairs * 100, 2),
        "CHAIRi":                    round(chairi * 100, 2),
        "total_captions":            n_caps,
        "captions_with_hallucination": caps_with_halluc,
        "total_objects_mentioned":   total_mentioned,
        "total_objects_hallucinated": total_hallucinated,
    }
