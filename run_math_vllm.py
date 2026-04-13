import argparse
import json
import re
from tqdm import tqdm

from local_model import LocalLLM
from grader import grade_answer

llm = LocalLLM()

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str)
parser.add_argument('--src', type=str)
parser.add_argument('--tgt', type=str)
parser.add_argument('--max_instance', type=int, default=100)

args = parser.parse_args()


def generate(prompt):
    return llm.generate(prompt)


# =========================
# SOLVE
# =========================
def solve(instance):

    prompt = f"""Solve this math problem step by step.

{instance['problem']}

Give final answer in \\boxed{{}} format.
"""

    out = generate(prompt)

    try:
        # Extract steps
        steps = re.findall(r"Step \d+:.*", out)
        steps = steps if steps else [out]

        # Extract answer properly
        match = re.findall(r"\\boxed\{(.*?)\}", out)

        if match:
            ans = match[-1]
        else:
            numbers = re.findall(r"\d+", out)

            # Remove small step numbers
            numbers = [n for n in numbers if int(n) > 5]

            ans = numbers[-1] if numbers else ""

        correct = grade_answer(ans, instance["answer"])

    except:
        steps, ans, correct = [], "", False

    instance["pre_generated_steps"] = [steps]
    instance["predict_answer"] = [ans]
    instance["correct"] = [correct]

    return instance


# =========================
# MAIN
# =========================
def main():

    data = [json.loads(line) for line in open(args.src)][:args.max_instance]

    results = []

    for inst in tqdm(data):
        inst = solve(inst)
        results.append(inst)

    with open(args.tgt, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()