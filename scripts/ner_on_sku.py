import pandas as pd
import spacy
import json

df = pd.read_excel(r"C:/Users/raad2/Downloads/Product Matching/data/Cleaned_data.xlsx")

nlp = spacy.load(r"C:/Users/raad2/Downloads/Product Matching/models/ner_model")
print("Model loaded successfully")

# ================================================================
# Run predictions and collect entities
# ================================================================
json_results = []

for text in df["SKU_Name"]:
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append([ent.start_char, ent.end_char, ent.label_])
    json_results.append({
        "text": text,
        "entities": entities
    })

# ================================================================
# Save results in JSON file
# ================================================================
output_path = r"C:/Users/raad2/Downloads/Product Matching/data/Extract_entities_spacy_SKU_data"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(json_results, f, ensure_ascii=False, indent=2)

# print len json file
print(f"Total processed: {len(json_results)}")