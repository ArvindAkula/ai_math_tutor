"""
Voice Input Processing Example

This example demonstrates how to use the voice input processing system
for mathematical problem input in the AI Math Tutor.
"""

import os
from voice_input import create_voice_input_processor, SpeechToTextProvider


def demonstrate_voice_input():
    """Demonstrate voice input processing with example scenarios"""
    
    print("=== AI Math Tutor Voice Input Processing Demo ===\n")
    
    # Note: In a real implementation, you would need a valid OpenAI API key
    # For this demo, we'll simulate the functionality
    api_key = os.getenv("OPENAI_API_KEY", "demo-key")
    
    try:
        # Create voice input processor
        processor = create_voice_input_processor(
            SpeechToTextProvider.OPENAI_WHISPER, 
            api_key
        )
        
        print("Voice input processor created successfully!")
        print("Note: This demo simulates voice input with text examples.\n")
        
        # Simulate various voice input scenarios
        test_scenarios = [
            "two plus three equals five",
            "find the derivative of x squared",
            "solve x squared minus four equals zero",
            "what is the integral of sine x",
            "calculate the square root of sixteen",
            "um two times uh three",  # With filler words
            "matrix multiplication",   # Incomplete expression
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"--- Scenario {i}: '{scenario}' ---")
            
            # Simulate the voice input processing
            # In real usage, this would be audio data from microphone
            result = processor.notation_converter.convert_to_mathematical_notation(scenario)
            
            print(f"Raw Speech: {result.raw_text}")
            print(f"Mathematical Expression: {result.mathematical_expression}")
            print(f"Domain: {result.recognized_domain}")
            print(f"Confidence: {result.confidence_score:.2f}")
            print(f"Validation Required: {result.validation_required}")
            
            if result.suggestions:
                print("Suggestions:")
                for suggestion in result.suggestions:
                    print(f"  - {suggestion}")
            
            # Demonstrate validation workflow
            if result.validation_required:
                validation = processor.validate_and_confirm(result)
                print(f"Validation Status: {validation['status']}")
                if validation['status'] == 'awaiting_confirmation':
                    print(f"Confirmation Message: {validation['message']}")
            
            print()
        
        print("=== Voice Input Processing Features ===")
        print("✓ Speech-to-text conversion (OpenAI Whisper integration)")
        print("✓ Mathematical notation conversion from spoken language")
        print("✓ Domain identification (algebra, calculus, etc.)")
        print("✓ Confidence scoring and validation workflow")
        print("✓ Intelligent suggestions for improvement")
        print("✓ Support for complex mathematical expressions")
        print("✓ Error handling and fallback strategies")
        
    except Exception as e:
        print(f"Error in demonstration: {e}")
        print("Note: This demo requires proper API configuration for full functionality.")


def demonstrate_mathematical_phrase_conversion():
    """Demonstrate mathematical phrase conversion capabilities"""
    
    print("\n=== Mathematical Phrase Conversion Examples ===\n")
    
    from voice_input import MathematicalNotationConverter
    converter = MathematicalNotationConverter()
    
    examples = [
        # Basic arithmetic
        ("five plus three", "Basic arithmetic"),
        ("ten minus four", "Subtraction"),
        ("six times seven", "Multiplication"),
        ("twenty divided by four", "Division"),
        
        # Algebraic expressions
        ("x squared plus two x minus one", "Quadratic expression"),
        ("y to the power of three", "Power notation"),
        ("square root of twenty five", "Square root"),
        
        # Calculus
        ("derivative of sine x", "Calculus derivative"),
        ("integral of x squared", "Integration"),
        ("limit as x approaches zero", "Limits"),
        
        # Functions
        ("cosine of pi over two", "Trigonometric function"),
        ("natural log of e", "Logarithmic function"),
        
        # Complex expressions
        ("solve x squared plus three x minus four equals zero", "Quadratic equation"),
        ("find the eigenvalues of the matrix", "Linear algebra"),
    ]
    
    for spoken, description in examples:
        result = converter.convert_to_mathematical_notation(spoken)
        print(f"{description}:")
        print(f"  Spoken: '{spoken}'")
        print(f"  Mathematical: '{result.mathematical_expression}'")
        print(f"  Domain: {result.recognized_domain}")
        print()


if __name__ == "__main__":
    demonstrate_voice_input()
    demonstrate_mathematical_phrase_conversion()