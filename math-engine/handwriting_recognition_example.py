"""
Handwriting Recognition Example

This example demonstrates how to use the handwriting recognition system
for mathematical notation input in the AI Math Tutor.
"""

import os
import io
from PIL import Image, ImageDraw, ImageFont
from handwriting_recognition import create_handwriting_processor, OCRProvider


def create_sample_math_image(text: str, width: int = 400, height: int = 100) -> bytes:
    """
    Create a sample mathematical expression image for demonstration
    
    Args:
        text: Mathematical text to render
        width: Image width
        height: Image height
        
    Returns:
        Image bytes
    """
    # Create a white background image
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a decent font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position (centered)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw the text in black
    draw.text((x, y), text, fill='black', font=font)
    
    # Convert to bytes
    output_buffer = io.BytesIO()
    image.save(output_buffer, format='PNG')
    return output_buffer.getvalue()


def demonstrate_handwriting_recognition():
    """Demonstrate handwriting recognition with example scenarios"""
    
    print("=== AI Math Tutor Handwriting Recognition Demo ===\n")
    
    # Note: In a real implementation, you would need valid API credentials
    # For this demo, we'll simulate the functionality
    
    try:
        # Create handwriting recognition processor
        # Using Mathpix as it's specialized for mathematical notation
        processor = create_handwriting_processor(
            OCRProvider.MATHPIX,
            app_id=os.getenv("MATHPIX_APP_ID", "demo-app-id"),
            app_key=os.getenv("MATHPIX_APP_KEY", "demo-app-key")
        )
        
        print("Handwriting recognition processor created successfully!")
        print("Note: This demo simulates handwriting recognition with rendered text images.\n")
        
        # Test scenarios with different mathematical expressions
        test_scenarios = [
            "x^2 + 2x + 1 = 0",
            "∫ sin(x) dx",
            "∂f/∂x = 2x",
            "√(a² + b²)",
            "lim x→0 (sin x)/x = 1",
            "Σ(i=1 to n) i = n(n+1)/2",
            "α + β = γ",
            "2 × 3 = 6",  # Test OCR corrections
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"--- Scenario {i}: '{scenario}' ---")
            
            # Create a sample image with the mathematical expression
            sample_image = create_sample_math_image(scenario)
            
            # Simulate the handwriting recognition processing
            # In real usage, this would process actual handwritten images
            result = processor.symbol_recognizer.recognize_symbols(scenario)
            
            print(f"Original Expression: {scenario}")
            print(f"Recognized Symbols: {len(result)} symbols")
            
            if result:
                print("Symbol Details:")
                for symbol in result[:5]:  # Show first 5 symbols
                    print(f"  - '{symbol.symbol}' ({symbol.symbol_type}) -> {symbol.latex_representation}")
                if len(result) > 5:
                    print(f"  ... and {len(result) - 5} more symbols")
            
            # Demonstrate validation workflow
            print("Processing Status: Simulated (would require actual OCR API)")
            print()
        
        print("=== Handwriting Recognition Features ===")
        print("✓ Image preprocessing (contrast enhancement, noise reduction)")
        print("✓ OCR integration (Mathpix, Google Vision)")
        print("✓ Mathematical symbol recognition and classification")
        print("✓ LaTeX to Unicode conversion")
        print("✓ OCR error correction (l→1, O→0, etc.)")
        print("✓ Confidence scoring and validation workflow")
        print("✓ Intelligent correction suggestions")
        print("✓ Support for complex mathematical notation")
        
    except Exception as e:
        print(f"Error in demonstration: {e}")
        print("Note: This demo requires proper API configuration for full functionality.")


def demonstrate_symbol_recognition():
    """Demonstrate mathematical symbol recognition capabilities"""
    
    print("\n=== Mathematical Symbol Recognition Examples ===\n")
    
    from handwriting_recognition import MathematicalSymbolRecognizer
    recognizer = MathematicalSymbolRecognizer()
    
    examples = [
        # Basic expressions
        ("2 + 3 = 5", "Basic arithmetic"),
        ("x^2 - 4 = 0", "Quadratic equation"),
        ("sin(π/2) = 1", "Trigonometric function"),
        
        # LaTeX expressions
        ("\\int_{0}^{\\pi} \\sin(x) dx", "LaTeX integral"),
        ("\\sum_{i=1}^{n} i^2", "LaTeX summation"),
        ("\\frac{d}{dx} x^3 = 3x^2", "LaTeX derivative"),
        
        # Unicode symbols
        ("α + β = γ", "Greek letters"),
        ("∫₀^π sin(x) dx", "Unicode integral"),
        ("√(x² + y²)", "Square root"),
        
        # Complex expressions
        ("lim_{x→0} (sin x)/x = 1", "Limit expression"),
        ("∂²f/∂x² + ∂²f/∂y² = 0", "Partial derivatives"),
        ("det(A) = |a b; c d| = ad - bc", "Matrix determinant"),
    ]
    
    for expression, description in examples:
        print(f"{description}:")
        print(f"  Input: '{expression}'")
        
        # Process the expression
        symbols = recognizer.recognize_symbols(expression)
        
        print(f"  Symbols: {len(symbols)} recognized")
        
        # Show symbol breakdown
        symbol_types = {}
        for symbol in symbols:
            symbol_types[symbol.symbol_type] = symbol_types.get(symbol.symbol_type, 0) + 1
        
        if symbol_types:
            type_summary = ", ".join([f"{count} {type_}" for type_, count in symbol_types.items()])
            print(f"  Types: {type_summary}")
        
        # Show LaTeX conversion
        latex_symbols = [s.latex_representation for s in symbols if s.latex_representation != s.symbol]
        if latex_symbols:
            print(f"  LaTeX: {', '.join(latex_symbols[:3])}{'...' if len(latex_symbols) > 3 else ''}")
        
        print()


def demonstrate_image_preprocessing():
    """Demonstrate image preprocessing capabilities"""
    
    print("\n=== Image Preprocessing Demonstration ===\n")
    
    from handwriting_recognition import ImagePreprocessor
    preprocessor = ImagePreprocessor()
    
    # Create test images with different characteristics
    test_cases = [
        ("Small image (50x50)", create_sample_math_image("x+1", 50, 50)),
        ("Normal image (200x100)", create_sample_math_image("∫ f(x) dx", 200, 100)),
        ("Large image (800x400)", create_sample_math_image("lim x→∞", 800, 400)),
    ]
    
    for description, image_data in test_cases:
        print(f"{description}:")
        
        # Get original image info
        original_image = Image.open(io.BytesIO(image_data))
        print(f"  Original: {original_image.size[0]}×{original_image.size[1]} pixels, mode: {original_image.mode}")
        
        # Preprocess the image
        processed_data = preprocessor.preprocess_image(image_data)
        processed_image = Image.open(io.BytesIO(processed_data))
        
        print(f"  Processed: {processed_image.size[0]}×{processed_image.size[1]} pixels, mode: {processed_image.mode}")
        
        # Check if upscaling was applied
        if processed_image.size != original_image.size:
            print("  ✓ Image was resized for better OCR recognition")
        
        if processed_image.mode != original_image.mode:
            print("  ✓ Image was converted to grayscale")
        
        print()
    
    print("Preprocessing Features:")
    print("✓ Automatic grayscale conversion")
    print("✓ Contrast enhancement")
    print("✓ Noise reduction (median filter)")
    print("✓ Image binarization (black/white)")
    print("✓ Automatic upscaling for small images")
    print("✓ Error handling and fallback")


if __name__ == "__main__":
    demonstrate_handwriting_recognition()
    demonstrate_symbol_recognition()
    demonstrate_image_preprocessing()