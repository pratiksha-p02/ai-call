from llm import call_llm

def detect_intent(text):
    prompt = f"Classify the customer intent in one phrase: {text}"
    return call_llm(prompt)

def detect_sentiment(text):
    prompt = f"Is the customer angry, neutral, or happy: {text}"
    return call_llm(prompt)

def suggest_response(context):
    prompt = f"""
    You are a call center assistant.
    Suggest the best response for the agent.

    Conversation:
    {context}
    """
    return call_llm(prompt)