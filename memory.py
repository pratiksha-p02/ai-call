conversation = []

def add_message(role, text):
    conversation.append({"role": role, "text": text})

def get_context(last_n=5):
    return "\n".join(
        [f"{msg['role']}: {msg['text']}" for msg in conversation[-last_n:]]
    )