#!/usr/bin/env python3
"""
Test script to verify OpenAI CLIP installation
"""

def test_clip_import():
    """Test if CLIP can be imported"""
    try:
        import clip
        print("✓ CLIP imported successfully")
        return True
    except ImportError as e:
        print(f"✗ CLIP import failed: {e}")
        print("\nInstall with: pip install git+https://github.com/openai/CLIP.git")
        return False

def test_clip_models():
    """Test available CLIP models"""
    try:
        import clip
        models = clip.available_models()
        print(f"✓ Available models: {models}")
        return True
    except Exception as e:
        print(f"✗ Failed to get models: {e}")
        return False

def test_clip_load():
    """Test loading CLIP model"""
    try:
        import clip
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        print("Loading ViT-B/32 model...")
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("✓ Model loaded successfully")
        
        # Test tokenization
        text = clip.tokenize(["a dog", "a cat"])
        print(f"✓ Tokenization works: {text.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("OpenAI CLIP Installation Test")
    print("=" * 60)
    
    print("\n1. Testing CLIP import...")
    if not test_clip_import():
        return
    
    print("\n2. Testing available models...")
    if not test_clip_models():
        return
    
    print("\n3. Testing model loading...")
    if not test_clip_load():
        return
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! CLIP is ready to use.")
    print("=" * 60)

if __name__ == "__main__":
    main()
