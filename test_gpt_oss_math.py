import subprocess
import json
from datetime import datetime

# -----------------------------------------------------
# ABSOLUTE PATH TO OLLAMA
# -----------------------------------------------------
OLLAMA_PATH = r"C:\Users\ali\AppData\Local\Programs\Ollama\ollama.exe"

# -----------------------------------------------------
# MODEL
# -----------------------------------------------------
MODEL = "glm-4.6:cloud"

# -----------------------------------------------------
# QUESTIONS
# -----------------------------------------------------
QUESTIONS = [
    "What is 2 + 3?",
    "Solve the equation x + 5 = 12.",
    "What is the derivative of x^2?",
    "Find the integral of x.",
    "What is the square root of 16?",
    "Factor the expression x^2 - 9.",
    "What is the value of 3 factorial?",
    "Explain in simple words what a function is."
]


# -----------------------------------------------------
# OLLAMA CALL (UTF-8 SAFE)
# -----------------------------------------------------
def ask_ollama(model: str, prompt: str) -> str:
    process = subprocess.Popen(
        [OLLAMA_PATH, "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8"   # ✅ FORCE UTF-8
    )

    stdout, stderr = process.communicate(prompt)

    if stderr:
        print("⚠ Ollama error:", stderr)

    return stdout.strip()

# -----------------------------------------------------
# SIMPLE METRICS
# -----------------------------------------------------
def evaluate_answer(answer: str) -> dict:
    return {
        "math_symbols": any(sym in answer for sym in ["∫", "√", "^", "dx", "=", "→"]),
        "length_score": round(min(len(answer) / 200, 1.0), 2),
        "steps_detected": answer.count("\n") >= 3,
        "explanation_quality_guess": any(
            w in answer.lower() for w in ["because", "thus", "therefore"]
        )
    }

# -----------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------
results = []

print(f"\n🔍 Testing model: {MODEL}\n")

for q in QUESTIONS:
    print(f"📘 Question: {q}")
    answer = ask_ollama(MODEL, q)
    print(f"➡ Answer:\n{answer}\n")

    results.append({
        "question": q,
        "answer": answer,
        "metrics": evaluate_answer(answer)
    })

# -----------------------------------------------------
# SAVE RESULTS
# -----------------------------------------------------
filename = f"glm_math_eval_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

with open(filename, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n✔ Evaluation saved to {filename}")
