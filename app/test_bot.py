import requests

API_URL = "http://127.0.0.1:8000/query"

test_cases = [
    {"question": "What is microplastic pollution?"},  # Dataset
    {"question": "Tell me about blue whales"},        # LLaMA fallback
    {"question": "Who is the president of USA?"}      # Filtered
]

for case in test_cases:
    print(f"\n📝 Question: {case['question']}")
    response = requests.post(API_URL, json=case)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"🤖 Answer: {data['answer']}\n")
    else:
        print(f"❌ Error: {response.status_code}")
