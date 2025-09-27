import google.generativeai as genai

# Configure with your API key
genai.configure(api_key="AIzaSyBw5esgD-JqZbPDV1xrK7ZB1xj7NdG6D_o")

print("Available Gemini models:")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"- {model.name}")