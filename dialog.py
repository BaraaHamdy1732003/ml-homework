import subprocess
import json
OLLAMA_PATH = r"C:\Users\ali\AppData\Local\Programs\Ollama\ollama.exe"
MODEL_NAME = "glm-4.6:cloud"
MAX_CONTEXT_LENGTH = 5   # how many last messages to keep

history = []  # list of {"role": "user"/"assistant", "content": "..."} dictionaries

def ask_ollama(prompt, context):
    """Send message + context to Ollama using subprocess."""

    # Prepare the combined prompt text
    context_text = ""
    for msg in context:
        prefix = "User" if msg["role"] == "user" else "Assistant"
        context_text += f"{prefix}: {msg['content']}\n"

    full_prompt = context_text + f"User: {prompt}\nAssistant:"

    # Call Ollama CLI
    result = subprocess.run(
        [OLLAMA_PATH, "run", MODEL_NAME],
        input=full_prompt.encode("utf-8"),
        stdout=subprocess.PIPE
    )

    response = result.stdout.decode("utf-8").strip()
    return response


print("💬 Диалог начат! Напишите 'закончить диалог' чтобы выйти.\n")

while True:
    user_input = input("Вы: ").strip()

    if user_input.lower() == "exit":
        print("👋 Диалог завершён.")
        break

    # Add user message to history
    history.append({"role": "user", "content": user_input})

    # Trim context to last N messages
    if len(history) > MAX_CONTEXT_LENGTH:
        history = history[-MAX_CONTEXT_LENGTH:]

    # Ask model
    answer = ask_ollama(user_input, history)

    # Save assistant reply
    history.append({"role": "assistant", "content": answer})

    print("AI:", answer)
                            