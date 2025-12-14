import json
import csv

# Full category name to short name mapping
full_to_short = {
    "Kidnapping / Abduction / Missing Person (BNS 140–151)": "kidnapping",
    "Sexual Offences (BNS 63–70)": "sexual_offence",
    "Assault / Hurt / Violence (BNS 115–140)": "assault",
    "Women & Child Safety (BNS 86 + POCSO)": "women_child_safety",
    "Harassment / Threats / Stalking (BNS 351–353)": "harassment",
    "Accident / Hit & Run (BNS 106, 112, 279)": "accident",
    "Cybercrime (IT Act + BNS mapping)": "cybercrime",
    "Fraud / Cheating / Financial Crimes (BNS 318–324)": "fraud",
    "Theft & Robbery (BNS 303–309)": "theft",
    "Trespass / Housebreaking / Property Disputes (BNS 332–335)": "trespass",
    "Defamation / Public Order Offences (BNS 356–357, 147–150)": "defamation",
    "Other / Cannot Classify": "other"
}

# Short name to numeric label
short_to_label = {
    "kidnapping": 0,
    "sexual_offence": 1,
    "assault": 2,
    "women_child_safety": 3,
    "harassment": 4,
    "accident": 5,
    "cybercrime": 6,
    "fraud": 7,
    "theft": 8,
    "trespass": 9,
    "defamation": 10,
    "other": 11
}

def update_jsonl_with_shortnames(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            full_category = data['category']
            short_category = full_to_short[full_category]
            data['category'] = short_category
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

def convert_jsonl_to_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as jsonl_file, \
         open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['text', 'label'])
        
        for line in jsonl_file:
            data = json.loads(line.strip())
            text = data['text']
            category = data['category']
            label = short_to_label[category]
            csv_writer.writerow([text, label])

# Update JSONL files with short names
print("Updating JSONL files with short category names...")
update_jsonl_with_shortnames('train.jsonl', 'train_temp.jsonl')
update_jsonl_with_shortnames('validate.jsonl', 'validate_temp.jsonl')
update_jsonl_with_shortnames('test.jsonl', 'test_temp.jsonl')

import os
os.replace('train_temp.jsonl', 'train.jsonl')
os.replace('validate_temp.jsonl', 'validate.jsonl')
os.replace('test_temp.jsonl', 'test.jsonl')

print("✓ Updated JSONL files with short names")

# Convert to CSV
print("\nConverting to CSV format...")
convert_jsonl_to_csv('train.jsonl', 'train.csv')
convert_jsonl_to_csv('validate.jsonl', 'validate.csv')
convert_jsonl_to_csv('test.jsonl', 'test.csv')

print("✓ Converted train.jsonl to train.csv")
print("✓ Converted validate.jsonl to validate.csv")
print("✓ Converted test.jsonl to test.csv")
print("\nDone!")
