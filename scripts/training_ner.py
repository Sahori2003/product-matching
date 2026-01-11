import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import json
import random
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# === Function to load data from JSON file ===

def load_training_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# === Convert data to spaCy format ===

def convert_to_spacy_format(training_data):
    spacy_data = []
    
    for item in training_data:
        text = item['text']
        entities = item['entities']
        
        if validate_entities(text, entities):
            spacy_data.append((text, {"entities": entities}))
        else:
            print(f"Warning: Invalid data - {text}")
    
    print(f"Converted {len(spacy_data)} valid examples")
    return spacy_data


# === Validate entity positions ===

def validate_entities(text, entities):
    for start, end, label in entities:
        if start < 0 or end > len(text) or start >= end:
            return False
        if not text[start:end].strip():
            return False
    return True


# === Create or load base model ===

def create_model(base_model="en_core_web_md"):
    print(f"\n{'='*60}")
    print(f"Creating model based on: {base_model}")
    print(f"{'='*60}\n")
    
    try:
        nlp = spacy.load(base_model)
        print(f"Loaded base model: {base_model}")
    except OSError:
        print(f"Model {base_model} not found")
        return None
    
    
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
        print("Added NER component")
    else:
        ner = nlp.get_pipe("ner")
        print("Found existing NER component")
    
    return nlp, ner


# === Add labels to model ===

def add_labels(ner, training_data):
    labels = set()
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            labels.add(ent[2])
    
    for label in labels:
        ner.add_label(label)
        print(f"Added label: {label}")
    
    return labels


# === Train model ===

def train_model(nlp, training_data, n_iter=45, dropout=0.5, batch_size=8):
    print(f"\n{'='*60}")
    print(f"Starting training")
    print(f"{'='*60}")
    print(f"Number of examples: {len(training_data)}")
    print(f"Number of epochs: {n_iter}")
    print(f"Dropout rate: {dropout}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    examples = []
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    optimizer = nlp.resume_training()
    
    with nlp.disable_pipes(*other_pipes):
        losses_history = []
        
        for epoch in range(n_iter):
            random.shuffle(examples)
            losses = {}
            batches = minibatch(examples, size=compounding(4.0, batch_size, 1.001))
            
            for batch in batches:
                nlp.update(
                    batch,
                    drop=dropout,
                    losses=losses,
                    sgd=optimizer
                )
            
            losses_history.append(losses['ner'])
            
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:3d}/{n_iter} - Loss: {losses['ner']:.4f}")
        
        print(f"\n{'='*60}")
        print(f"✓ Training finished!")
        print(f"✓ Final Loss: {losses_history[-1]:.4f}")
        print(f"✓ Initial Loss: {losses_history[0]:.4f}")
        print(f"✓ Improvement: {((losses_history[0] - losses_history[-1]) / losses_history[0] * 100):.2f}%")
        print(f"{'='*60}\n")
    
    return nlp, losses_history


# === Evaluate model performance ===

def evaluate_model_detailed(nlp, test_data):
    print(f"\n{'='*60}")
    print(f"Evaluating model - Detailed Metrics")
    print(f"{'='*60}\n")
    
    entity_stats = {}
    
    for text, annotations in test_data:
        doc = nlp(text)
        predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        gold_entities = annotations['entities']
        
        pred_set = set(predicted_entities)
        gold_set = set(map(tuple, gold_entities))
        
        for pred_ent in pred_set:
            label = pred_ent[2]
            if label not in entity_stats:
                entity_stats[label] = {'tp': 0, 'fp': 0, 'fn': 0}
            
            if pred_ent in gold_set:
                entity_stats[label]['tp'] += 1
            else:
                entity_stats[label]['fp'] += 1
        
        for gold_ent in gold_set:
            label = gold_ent[2]
            if label not in entity_stats:
                entity_stats[label] = {'tp': 0, 'fp': 0, 'fn': 0}
            
            if gold_ent not in pred_set:
                entity_stats[label]['fn'] += 1
    
    total_tp = total_fp = total_fn = 0
    print(f"{'Entity Type':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"{'-'*60}")
    
    for label in sorted(entity_stats.keys()):
        tp = entity_stats[label]['tp']
        fp = entity_stats[label]['fp']
        fn = entity_stats[label]['fn']
        total_tp += tp; total_fp += fp; total_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn
        
        print(f"{label:<15} {precision:>10.4f} {recall:>10.4f} {f1_score:>10.4f} {support:>10}")
    
    print(f"{'-'*60}")
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
    
    print(f"{'TOTAL':<15} {total_precision:>10.4f} {total_recall:>10.4f} {total_f1:>10.4f}")
    print(f"\n{'='*60}\n")
    
    return {
        'precision': total_precision,
        'recall': total_recall,
        'f1_score': total_f1
    }


# === Save trained model ===

def save_model(nlp, output_dir="./trained_model"):
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    nlp.to_disk(output_path)
    print(f"Model saved to: {output_path}")
    return output_path


# === Save NER results to JSON file ===
def save_ner_predictions(nlp, input_texts, output_path):
    results = []
    for text in input_texts:
        doc = nlp(text)
        entities_dict = {}
        for ent in doc.ents:
            entities_dict[ent.label_] = ent.text
        results.append({
            "Text": text,
            **entities_dict
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nNER predictions saved to: {output_path}")



# === Test model ===

def test_model(nlp, test_texts):
    print(f"\n{'='*60}")
    print(f"Testing model on new texts")
    print(f"{'='*60}\n")
    
    for text in test_texts:
        doc = nlp(text)
        print(f"Text: {text}")
        print(f"Entities:")
        if doc.ents:
            for ent in doc.ents:
                print(f"  [{ent.start_char:3d}, {ent.end_char:3d}] {ent.label_:15s} → '{ent.text}'")
        else:
            print("  (No entities found)")
        print()


# === Main function ===

def main():
    print("=" * 60)
    print("STEP 1: Load training data")
    print("=" * 60)
    alias_data_raw = load_training_data(r"C:/Users/raad2/Downloads/Product Matching/data/Extract_entities_alias_data.json")
    sku_data_raw = load_training_data(r"C:/Users/raad2/Downloads/Product Matching/data/Extract_entities_sku_data.json")
        
    # --- Combine them ---
    training_data_raw = alias_data_raw + sku_data_raw
    print(f"Total loaded examples: {len(training_data_raw)}")

    training_data = convert_to_spacy_format(training_data_raw)
    
    random.shuffle(training_data)
    split_index = int(len(training_data) * 0.8)
    train_data = training_data[:split_index]
    test_data = training_data[split_index:]
    
    print(f"\Data split:")
    print(f"Training: {len(train_data)} examples")
    print(f"Testing:  {len(test_data)} examples")
    
    # create el model
    nlp, ner = create_model("en_core_web_md")
    labels = add_labels(ner, train_data)
    
    # training
    nlp, losses = train_model(nlp, train_data)
    
    # evaluation
    metrics = evaluate_model_detailed(nlp, test_data)
    
    # save
    model_path = save_model(nlp, r"C:/Users/raad2/Downloads/Product Matching/models/ner_model")

    # test
    test_texts = [
        "panadol 500mg tablet 20",
        "augmentin 1g injection",
        "ventolin 100mcg inhaler",
        "aspirin 100mg tablet 30"
    ]
    test_model(nlp, test_texts)
    
    print("\nModel saved to:", model_path)


if __name__ == "__main__":
    main()