from groq import Groq

# Test Groq API and available models
client = Groq(api_key="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")

# Test different model names
models_to_test = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile", 
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

for model in models_to_test:
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, just testing!"}],
            model=model,
            max_tokens=10
        )
        print(f"✅ {model} - WORKS")
        print(f"   Response: {response.choices[0].message.content}")
        break  # Use the first working model
    except Exception as e:
        print(f"❌ {model} - Error: {str(e)[:100]}...")

print("\nTesting complete!")