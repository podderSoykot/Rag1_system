# rag_system/test_api.py
# Simple script to test OpenAI API key

from openai import OpenAI
from config.settings import OPENAI_API_KEY

def test_openai_api():
    """Test if the OpenAI API key is working"""
    print("🔑 Testing OpenAI API key...")
    print(f"API Key loaded: {'Yes' if OPENAI_API_KEY else 'No'}")
    print(f"API Key length: {len(OPENAI_API_KEY) if OPENAI_API_KEY else 0}")
    
    if not OPENAI_API_KEY:
        print("❌ No API key found!")
        return False
    
    try:
        # Create OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Test with a simple request
        print("🔄 Testing API connection...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello, API test successful!'"}],
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ API test successful!")
        print(f"🤖 Response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_openai_api()
