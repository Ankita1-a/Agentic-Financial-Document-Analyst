import os
from mistralai.client import Mistral
from mistralai.client.errors import SDKError

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

for model in ["mistral-small-latest", "open-mistral-nemo", "ministral-3b-latest", "mistral-medium-latest"]:
    print(f"=== {model} ===")
    try:
        response = client.chat.complete(model=model, messages=[{"role": "user", "content": "Say OK"}])
        print("SUCCESS:", response.choices[0].message.content)
    except SDKError as e:
        h = e.raw_response.headers
        print(f"FAILED status={e.raw_response.status_code} "
              f"limit_per_min={h.get('x-ratelimit-limit-req-minute')} "
              f"remaining_per_min={h.get('x-ratelimit-remaining-req-minute')}")
    print()