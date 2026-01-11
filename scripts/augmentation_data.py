"""
Hard Pos -> 1000 sku = sku
Soft Pos -> 1000 sku ~ sku (remove one entity)
Hard Neg -> 1000 sku != sku (reversed)
+ Hard Neg -> 1000 sku != sku (shifted by 1) ###
Soft Neg -> 5000 sku !~ sku (replace one entity with another value of same type)
"""
import json
import random
import pandas as pd
from copy import deepcopy
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
INPUT_FILE = r"C:/Users/raad2/Downloads/Product Matching/data/Extract_entities_spacy_SKU_data.json"
OUTPUT_DIR = Path(r"C:/Users/raad2/Downloads/Product Matching/data/labeled_datasets_specified")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# types order used for soft_neg generation (5 variants per record)
SOFT_NEG_TYPES = ['FORM','QUANTITY','DOSAGE_UNIT','DOSAGE_VALUE','BRAND']
# Replacement vocabularies (expanded from old code)
BRAND_REPLACEMENTS = [
    "Flector", "Atomorelax", "Bio Balance", "Atrozemb", "Gluco Dr", "Dettol",
    "Sebanoin", "Trypsalin", "Vatika", "Uno Plus", "Adidas", "Zola", "Bio Energy",
    "Itolon", "Natal", "Lasilactone", "Isis", "Cansin", "Acti Flu", "Ponoforte",
    "Systane", "Allerest", "Emper", "Lozaty", "Maalox", "Aprovel", "Manuka Doctor",
    "Cbo", "Cornetears", "Lamsa", "Varnova", "Fine Baby", "Chicco", "Lorius",
    "Lipodar", "Jidoo", "Koleston", "Co-Tareg", "Iybao", "Rigenforte", "Nan Optipro",
    "Roxone", "Freshdays", "Doxiproct", "Limitless", "Exofen", "Erastapex", "Moximax",
    "Venus", "Baby Life", "Tresemme", "Candistan", "Hi Dee", "Tabocine", "Vatika Menz",
    "Essential Vitamin D3", "Sebamed", "Dorofen", "Curam", "Cal-Heparine",
    "Dentissimo", "Dermovate", "Zinc", "Sofy", "Ascensia Elite", "Powerecta", "Unac",
    "BT Pharma", "Garamycin", "Ozone", "Anoxicam", "Nestogen", "Bronson", "Lecoxen",
    "Loreal", "Bambini", "Butacid", "Vatika Oilfusion", "Vichy", "Locoid", "Prevalin",
    "Pasante", "Citrobiotic", "Bariatric Fusion", "Unifungi", "Sensodyne", "Aptamil",
    "Titania", "Farlin", "Salibet", "Palette", "Fluorodine", "Celia Deluxe",
    "Pigeon", "Hidrasec", "Jasper", "Titania Decubits", "Rotahelex", "Allerfin"
]

DOSAGE_VALUES = ['5', '10', '20', '25', '50', '100', '150', '200', '250', '400', '500']
DOSAGE_UNITS = ['mg', 'mcg', 'g', 'kg', 'ml', 'l', 'mg/ml', 'meq', 'iu', '%', 'pk']
FORM_REPLACEMENTS = [
    'tablet', 'capsule', 'syrup', 'suspension', 'solution', 'cream', 'gel', 
    'lotion', 'paste', 'drops', 'sachet', 'powder', 'granule', 'patch', 'spray',
    'film', 'lozenge', 'injection', 'ampoule', 'vial', 'softgel', 'inhaler',
    'suppository', 'ovule', 'pessary', 'shampoo', 'conditioner', 'serum', 'mask',
    'oil', 'balm', 'soap', 'wash', 'condom', 'hand sanitizer', 'milk', 'spry',
    'salt', 'effervescent tablet', 'foam', 'wax', 'sanitizer', 'wipes', 'pad',
    'diaper', 'toothbrush', 'toothpaste', 'thermometer', 'nebulizer', 'glucometer',
    'razor', 'bottle', 'kit', 'device', 'makeup', 'lip makeup', 'eye makeup',
    'nail product', 'makeup remover', 'cleaner', 'fragrance', 'medical accessory'
]
QUANTITIES = ['1', '2', '3', '5', '10', '12', '15', '20', '24', '30', '48', '50', '100']


# -------------------------
# Helpers (text + entities manipulation)
# Entities format assumed: list of [start, end, label]
# Each input item: {"text": "...", "entities": [[s,e,label], ...]}
# -------------------------
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_entity_by_label(ents, label):
    for idx, (s,e,l) in enumerate(ents):
        if l == label:
            return idx, s, e, l
    return None

def replace_entity_in_text(text, ents, ent_idx, new_value):
    s,e,l = ents[ent_idx]
    new_text = text[:s] + new_value + text[e:]
    # update entities positions
    delta = len(new_value) - (e - s)
    new_ents = []
    for i,(s2,e2,l2) in enumerate(ents):
        if i == ent_idx:
            new_ents.append([s2, s2 + len(new_value), l2])
        else:
            if s2 >= e:
                new_ents.append([s2 + delta, e2 + delta, l2])
            else:
                new_ents.append([s2, e2, l2])
    return new_text, new_ents

def remove_entity_from_text(text, ents, ent_idx):
    s,e,l = ents[ent_idx]
    new_text = (text[:s].strip() + " " + text[e:].strip()).strip()
    delta = s - e
    new_ents = []
    for i,(s2,e2,l2) in enumerate(ents):
        if i == ent_idx:
            continue
        if s2 > s:
            new_ents.append([s2 + delta, e2 + delta, l2])
        else:
            new_ents.append([s2, e2, l2])
    return new_text, new_ents

def validate_entities(text, ents):
    if not ents:
        return False
    ents_sorted = sorted(ents, key=lambda x: x[0])
    for i,(s,e,l) in enumerate(ents_sorted):
        if s < 0 or e > len(text) or s >= e:
            return False
        if not text[s:e].strip():
            return False
        if i < len(ents_sorted)-1 and e > ents_sorted[i+1][0]:
            return False
    return True

# safe random word replace fallback
def fallback_replace_random_word(text, replacements):
    words = text.split()
    if not words:
        return None
    idx = random.randrange(len(words))
    new_words = words.copy()
    new_words[idx] = random.choice(replacements)
    return " ".join(new_words)

# -------------------------
# Dataset creators (matching required sizes)
# -------------------------
def create_hard_pos(data):
    # alias = sku (original text), label = 1
    rows = []
    for item in data:
        rows.append({'alias': item['text'], 'sku': item['text'], 'label': 1})
    df = pd.DataFrame(rows)
    df.to_excel(OUTPUT_DIR / "hard_pos.xlsx", index=False)
    return df

def create_soft_pos(data):
    # want exactly N rows (one modified version per original)
    rows = []
    for item in data:
        text = item['text']
        ents = deepcopy(item['entities'])
        # attempt to remove ONE non-BRAND entity; if none, remove a random non-empty token
        labels_available = [ent[2] for ent in ents if ent[2] != 'BRAND']
        modified = None
        if labels_available:
            # pick one to remove
            remove_label = random.choice(labels_available)
            ent_info = get_entity_by_label(ents, remove_label)
            if ent_info:
                ent_idx, s, e, l = ent_info
                new_text, new_ents = remove_entity_from_text(text, ents, ent_idx)
                if validate_entities(new_text, new_ents):
                    modified = {'alias': new_text, 'sku': text, 'label': 1}
        if modified is None:
            # fallback: remove a random word (not ideal but ensures count)
            words = text.split()
            if len(words) > 1:
                idx = random.randrange(len(words))
                new_words = words[:idx] + words[idx+1:]
                new_text = " ".join(new_words).strip()
                if new_text:
                    modified = {'alias': new_text, 'sku': text, 'label': 1}
        # if still None (rare), keep original as soft_pos variant
        if modified is None:
            modified = {'alias': text, 'sku': text, 'label': 1}
        rows.append(modified)
    df = pd.DataFrame(rows).drop_duplicates()
    # if duplicates reduced length, we resample from originals to reach N
    N = len(data)
    if len(df) < N:
        needed = N - len(df)
        # sample originals and generate simple deletions for them
        samples = random.choices(data, k=needed)
        for sitem in samples:
            t = sitem['text']
            w = t.split()
            if len(w) > 1:
                idx = random.randrange(len(w))
                new_t = " ".join(w[:idx] + w[idx+1:])
            else:
                new_t = t
            df = pd.concat([df, pd.DataFrame([{'alias': new_t, 'sku': t, 'label': 1}])], ignore_index=True)
    # finally ensure length == N
    if len(df) > N:
        df = df.sample(n=N, random_state=42).reset_index(drop=True)
    df.to_excel(OUTPUT_DIR / "soft_pos.xlsx", index=False)
    return df

def create_hard_neg_reversed(data):
    # requirement: hard_neg length == N (use reversed original list so length preserved)
    texts = [item['text'] for item in data]
    reversed_texts = texts[::-1]
    rows = []
    for alias_text, sku_text in zip(texts, reversed_texts):
        rows.append({'alias': alias_text, 'sku': sku_text, 'label': 0})
    df = pd.DataFrame(rows)
    df.to_excel(OUTPUT_DIR / "hard_neg_reversed.xlsx", index=False)
    return df

def create_hard_neg_shifted(data):
    """Hard negatives by shifting texts by 1 position, saved as Excel"""
    texts = [item['text'] for item in data]
    shifted_texts = texts[1:] + [texts[0]]  # shift by 1
    rows = []
    for alias_text, sku_text in zip(texts, shifted_texts):
        rows.append({'alias': alias_text, 'sku': sku_text, 'label': 0})
    df = pd.DataFrame(rows)
    df.to_excel(OUTPUT_DIR / "hard_neg_shifted.xlsx", index=False)
    return df


def create_soft_neg(data, factor=5):
    # requirement: for each original produce exactly 5 modified negatives
    rows = []
    for item in data:
        original = item['text']
        ents = deepcopy(item['entities'])
        produced = 0
        tries = 0
        # try to produce one variant per SOFT_NEG_TYPES in order
        for typ in SOFT_NEG_TYPES:
            if produced >= factor:
                break
            tries += 1
            mod = None
            # try replace the requested type
            if typ == 'FORM':
                mod = replace_entity_by_label_helper(item, 'FORM', FORM_REPLACEMENTS)
            elif typ == 'QUANTITY':
                mod = replace_entity_by_label_helper(item, 'QUANTITY', QUANTITIES)
            elif typ == 'DOSAGE_UNIT':
                mod = replace_entity_by_label_helper(item, 'DOSAGE_UNIT', DOSAGE_UNITS)
            elif typ == 'DOSAGE_VALUE':
                mod = replace_entity_by_label_helper(item, 'DOSAGE_VALUE', DOSAGE_VALUES)
            elif typ == 'BRAND':
                mod = replace_entity_by_label_helper(item, 'BRAND', BRAND_REPLACEMENTS)
            # fallback: if mod is None, try other labels or fallback replace random word
            if mod is None:
                # try any other replaceable label
                for alt_label, pool in [('FORM', FORM_REPLACEMENTS), ('DOSAGE_VALUE', DOSAGE_VALUES),
                                       ('DOSAGE_UNIT', DOSAGE_UNITS), ('BRAND', BRAND_REPLACEMENTS),
                                       ('QUANTITY', QUANTITIES)]:
                    if alt_label == typ:
                        continue
                    mod = replace_entity_by_label_helper(item, alt_label, pool)
                    if mod:
                        break
            if mod is None:
                # fallback to random word replacement with brand pool
                fallback = fallback_replace_random_word(original, BRAND_REPLACEMENTS)
                if fallback and fallback != original:
                    mod = {'alias': fallback, 'sku': original, 'label': 0}
            if mod:
                rows.append(mod)
                produced += 1
        # if produced < factor, we keep trying by random replacements until reach factor
        extra_tries = 0
        while produced < factor and extra_tries < 10:
            extra_tries += 1
            # random replace any label
            alt = random.choice([
                ('FORM', FORM_REPLACEMENTS),
                ('DOSAGE_VALUE', DOSAGE_VALUES),
                ('DOSAGE_UNIT', DOSAGE_UNITS),
                ('BRAND', BRAND_REPLACEMENTS),
                ('QUANTITY', QUANTITIES)
            ])
            mod = replace_entity_by_label_helper(item, alt[0], alt[1])
            if mod:
                rows.append(mod); produced += 1
            else:
                fallback = fallback_replace_random_word(original, BRAND_REPLACEMENTS)
                if fallback and fallback != original:
                    rows.append({'alias': fallback, 'sku': original, 'label': 0}); produced += 1
        # if still short (very unlikely) duplicate last produced to reach factor
        while produced < factor:
            if rows:
                rows.append(rows[-1].copy())
            else:
                rows.append({'alias': original + " x", 'sku': original, 'label': 0})
            produced += 1

    df = pd.DataFrame(rows)
    # ensure size == factor * N
    N = len(data)
    expected = factor * N
    if len(df) > expected:
        df = df.sample(n=expected, random_state=42).reset_index(drop=True)
    df.to_excel(OUTPUT_DIR / "soft_neg.xlsx", index=False)
    return df

# small helper used inside create_soft_neg
def replace_entity_by_label_helper(item, label, replacements_pool):
    text = item['text']; ents = deepcopy(item['entities'])
    info = get_entity_by_label(ents, label)
    if info is None:
        return None
    ent_idx, s,e,l = info
    old = text[s:e]
    choices = [r for r in replacements_pool if r.lower() != old.lower()]
    if not choices:
        return None
    new_value = random.choice(choices)
    try:
        new_text, new_ents = replace_entity_in_text(text, ents, ent_idx, new_value)
    except Exception:
        return None
    if validate_entities(new_text, new_ents):
        return {'alias': new_text, 'sku': text, 'label': 0}
    # if replacement produced invalid spans, try fallback simple word replace
    fb = fallback_replace_random_word(text, replacements_pool)
    if fb and fb != text:
        return {'alias': fb, 'sku': text, 'label': 0}
    return None

# -------------------------
# MAIN
# -------------------------
def main():
    data = load_json(INPUT_FILE)
    N = len(data)
    print(f"Loaded {N} records")

    hp = create_hard_pos(data)           # should be N
    sp = create_soft_pos(data)           # should be N
    hnr = create_hard_neg_reversed(data) # should be N
    hns = create_hard_neg_shifted(data)  # should be N
    sn = create_soft_neg(data, factor=5) # should be 5*N

    print("Files created and saved to:", OUTPUT_DIR)
    print(f"hard_pos: {len(hp)} rows (expected {N})")
    print(f"soft_pos: {len(sp)} rows (expected {N})")
    print(f"hard_neg_reversed: {len(hnr)} rows (expected {N})")
    print(f"hard_neg_shifted: {len(hns)} rows (expected {N})")
    print(f"soft_neg: {len(sn)} rows (expected {5*N})")

    # create combined file (concatenate all)
    combined = pd.concat([hp, sp, hnr, hns, sn], ignore_index=True)
    combined_path = OUTPUT_DIR / "all_combined_labeled_specified.xlsx"
    combined.to_excel(combined_path, index=False)
    print("Combined dataset saved:", combined_path)

if __name__ == "__main__":
    main()