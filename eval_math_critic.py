import json

data = [json.loads(l) for l in open("solve.jsonl")]

total = len(data)
correct = sum(1 for d in data if any(d.get("correct", [])))

print("Accuracy:", correct / total * 100)