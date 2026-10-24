import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "Rag-1"))

from agents import ask


dataset_path = Path(__file__).resolve().parent / "eval_dataset.json"
results_path = Path(__file__).resolve().parent / "eval_result1.json"

with dataset_path.open(encoding="utf-8") as dataset_file:
    data = json.load(dataset_file)

results = []

for sample in data:
    question = sample["question"]
    expected_answer = sample["expected_answer"]

    actual_response = ask(question)

    result = {
        "question": question,
        "expected_answer": expected_answer,
        "actual_answer": actual_response["ans"],
        "context_used": actual_response["context"],
    }

    results.append(result)
    print(f"Answer from Rag: {actual_response['ans']}")

with results_path.open("w", encoding="utf-8") as results_file:
    json.dump(results, results_file, ensure_ascii=False, indent=2)

print(f"Evaluation completed. Results saved to {results_path.name}")
