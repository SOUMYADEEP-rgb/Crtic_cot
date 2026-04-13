# Members who contributed to the case study:
SOUMYADEEP DASGUPTA (23BDS0134)
AMAN HOODA (23BDS0157)
ISHITA SINGH (23BDS0189)
HUSSAIN DIYAB (23BDS0201)
# 🧠 Critic-CoT Pipeline using Small Language Model

This project implements a Chain-of-Thought (CoT) reasoning pipeline using a small language model to solve mathematical problems from the GSM8K dataset.

Instead of directly predicting answers, the model generates step-by-step reasoning, improving accuracy and interpretability.

---

## 🚀 Overview

Pipeline Flow:

Input Problem → Reasoning → Answer Extraction → Evaluation

---

## 📂 Project Structure

- local_model.py → Loads model and handles generation  
- run_math_vllm.py → Main pipeline (solve problems)  
- utils.py → Helper functions for parsing  
- grader.py → Checks correctness of answers  
- eval_math_critic.py → Calculates accuracy  
- train.jsonl → Dataset (GSM8K subset)  

---

## ⚙️ How It Works

1. Input  
   Problems are loaded from train.jsonl  

2. Solve Stage  
   Model generates step-by-step reasoning:

   Step 1: ...  
   Step 2: ...  
   Step 3: ...  
   \boxed{answer}  

3. Answer Extraction  
   Extracts final answer from boxed format  

4. Evaluation  
   Compares predicted answer with actual answer  

---

## 🧠 Model Used

- Model: microsoft/phi-2  
- Type: Small Language Model (SLM)  
- Framework: HuggingFace Transformers  

---

## 📊 Results

- Initial Accuracy: ~14%  
- Final Accuracy: ~55–65%  

---

## 🛠️ Installation

pip install torch transformers tqdm

---

## ▶️ How to Run

Run Solve Pipeline:

python run_math_vllm.py --mode solve --src train.jsonl --tgt solve.jsonl

Run Evaluation:

python eval_math_critic.py

---

## 📈 Key Improvements

- Chain-of-Thought prompting using SLM ( Microsoft phi-2)
- Prompt engineering  
- Answer extraction optimization  
- Deterministic decoding  

---

## ⚠️ Limitations

- Small model → limited reasoning ability  
- Accuracy lower than large models  
- Some incorrect reasoning still occurs  

---

## 🔮 Future Work

- Improve accuracy using multi-sampling  
- Fine-tune model on math dataset  
- Use larger models  
- Add critic/refinement stage  

---

## 🎯 Conclusion

- Step-by-step reasoning improves performance  
- Small models can solve complex problems with proper prompting  
- Modular pipelines help build interpretable AI systems  

---
