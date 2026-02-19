from google import genai

client = genai.Client(api_key='AIzaSyCPtbtlUsZwvb0Jqv-rGCCFt0f3QxccJgw')

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="How does AI work?"
)
print(response.text)