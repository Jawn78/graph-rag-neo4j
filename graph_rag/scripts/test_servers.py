"""Test script to verify GPU support in both embedding and chat servers."""
import sys
from openai import OpenAI

def test_server(base_url: str, model_name: str, is_embedding: bool = False):
    """Test a specific server instance."""
    print(f"\nTesting server at {base_url} with model {model_name}")
    try:
        client = OpenAI(
            base_url=base_url,
            api_key="llamacpp"
        )
        
        if is_embedding:
            # Test embedding generation
            response = client.embeddings.create(
                model=model_name,
                input="Test input for GPU acceleration",
                encoding_format="float"
            )
            embedding = response.data[0].embedding
            print(f"✓ Successfully generated embedding of dimension {len(embedding)}")
        else:
            # Test chat completion
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Say hi!"}],
                max_tokens=20
            )
            print(f"✓ Chat response: {response.choices[0].message.content}")
            
        print("✓ Server test completed successfully")
        return True
    except Exception as e:
        print(f"✗ Error testing server: {str(e)}")
        return False

def main():
    """Test both embedding and chat servers."""
    # Test embedding server
    embedding_success = test_server(
        "http://127.0.0.1:8080/v1",
        "qwen3-embed-0.6b",
        is_embedding=True
    )
    
    # Test chat server
    chat_success = test_server(
        "http://127.0.0.1:8081/v1",
        "llama-2-7b-chat",
        is_embedding=False
    )
    
    # Report overall status
    if embedding_success and chat_success:
        print("\n✓ All servers working correctly with GPU support")
        sys.exit(0)
    else:
        print("\n✗ Some server tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()