import pandas as pd
import numpy as np
import re
import json
from preprocessing_data import unit, FORMULATIONS


def extract_all_entities_fixed(text):
    """Extract all possible entities from text"""
    entities = [] 
    # Extract DOSAGE (VALUE + UNIT)
    units_pattern = '|'.join([re.escape(u) for u in unit])
    dosage_pattern = rf'(\d+(?:\.\d+)?)\s*({units_pattern})'
    
    dosage_match = re.search(dosage_pattern, text, flags=re.IGNORECASE)
    
    # Extract FORM + QUANTITY
    forms_pattern = '|'.join([re.escape(f.rstrip('s')) for f in FORMULATIONS])
    form_pattern = rf'(?:(\d+)\s*({forms_pattern})|({forms_pattern})\s*(\d+)|({forms_pattern}))'
    
    form_match = re.search(form_pattern, text.lower())
    
    # Extract BRAND
    brand_end = len(text)
    first_entity_start = len(text)
    
    if dosage_match:
        first_entity_start = min(first_entity_start, dosage_match.start())
    
    if form_match:
        first_entity_start = min(first_entity_start, form_match.start())
    
    brand_end = first_entity_start
    
    brand_text = text[:brand_end].strip()
    if brand_text:
        words = brand_text.split()
        if words:
            first_word_pos = text.find(words[0])
            last_word_pos = text.find(words[-1], first_word_pos) + len(words[-1])
            entities.append([first_word_pos, last_word_pos, 'BRAND'])
    
    if dosage_match:
        entities.append([dosage_match.start(1), dosage_match.end(1), 'DOSAGE_VALUE'])
        entities.append([dosage_match.start(2), dosage_match.end(2), 'DOSAGE_UNIT'])
    
    if form_match:
        if form_match.group(1) and form_match.group(2):
            entities.append([form_match.start(1), form_match.end(1), 'QUANTITY'])
            entities.append([form_match.start(2), form_match.end(2), 'FORM'])
        elif form_match.group(3) and form_match.group(4):
            entities.append([form_match.start(3), form_match.end(3), 'FORM'])
            entities.append([form_match.start(4), form_match.end(4), 'QUANTITY'])
        elif form_match.group(5):
            entities.append([form_match.start(5), form_match.end(5), 'FORM'])
    
    return entities


def validate_and_fix_entities(text, entities):
    """Validate entities and fix any overlaps"""
    if not entities:
        return []
    
    # Sort entities by position
    entities_sorted = sorted(entities, key=lambda x: x[0])
    valid_entities = []
    
    for i, ent in enumerate(entities_sorted):
        start, end, label = ent
        
        # Check for valid range
        if start < 0 or end > len(text) or start >= end:
            continue
        
        # Check for overlap with previous entity
        if valid_entities:
            prev_start, prev_end, prev_label = valid_entities[-1]
            if start < prev_end:
                continue
        
        valid_entities.append([start, end, label])
    
    return valid_entities


def create_training_data(df, column_name):
    """Create training data from DataFrame"""
    training_data = []
    skipped_count = 0
    
    for idx, row in df.iterrows():
        text = row[column_name]
        
        # Extract entities
        entities = extract_all_entities_fixed(text)
        
        # Validate and fix overlaps
        valid_entities = validate_and_fix_entities(text, entities)
        
        # Count skipped entities due to overlaps
        if len(entities) > len(valid_entities):
            skipped_count += 1
        
        # Add only valid entities
        if valid_entities:
            training_data.append({
                "text": text,
                "entities": valid_entities
            })
    
    if skipped_count > 0:
        print(f"{skipped_count} entities skipped due to overlaps")
    
    return training_data

# ========== Main Execution ==========

df = pd.read_excel(r"C:/Users/raad2/Downloads/Product Matching/data/Cleaned_data.xlsx")
df = df[["alias", "SKU_Name"]].dropna().reset_index(drop=True)
print(f"Loaded {len(df)} rows")


training_data_SKU = create_training_data(df, "SKU_Name")
training_data_alias = create_training_data(df, "alias")

print(f"Created {len(training_data_SKU)} training examples")
print(f"Created {len(training_data_alias)} training examples")

# Save file
with open(r'C:/Users/raad2/Downloads/Product Matching/data/Extract_entities_alias_data.json', 'w', encoding='utf-8') as f:
    json.dump(training_data_alias, f, ensure_ascii=False, indent=2)
    
with open(r'C:/Users/raad2/Downloads/Product Matching/data/Extract_entities_sku_data.json', 'w', encoding='utf-8') as f:
    json.dump(training_data_SKU, f, ensure_ascii=False, indent=2)