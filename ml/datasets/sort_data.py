import csv

# Read all rows
rows = []
with open('crime_data.csv', 'r', encoding='utf-8') as f:
    for line in f:
        # Split from the right to get label (last comma-separated value)
        parts = line.strip().rsplit(',', 1)
        if len(parts) == 2:
            text, label = parts
            try:
                rows.append((text, int(label)))
            except ValueError:
                print(f"Warning: Could not parse label from line: {line[:50]}...")

print(f"Original data: {len(rows)} rows")

# Sort by label (second element of tuple)
rows_sorted = sorted(rows, key=lambda x: x[1])

# Count labels
from collections import Counter
label_counts = Counter(label for _, label in rows_sorted)
print(f"Label distribution:")
for label in sorted(label_counts.keys()):
    print(f"  Label {label}: {label_counts[label]} rows")

# Write back to file
with open('crime_data.csv', 'w', encoding='utf-8', newline='') as f:
    for text, label in rows_sorted:
        f.write(f"{text},{label}\n")

print(f"\n✅ File sorted by label in ascending order (0-11)")
print(f"Total rows: {len(rows_sorted)}")
