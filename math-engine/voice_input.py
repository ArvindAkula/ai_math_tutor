"""
Voice Input Processing Module for AI Math Tutor

This module handles speech-to-text conversion and mathematical notation
conversion from spoken language for natural mathematical problem input.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# For speech-to-text integration (placeholder for actual API)
import requests
from abc import ABC, abstractmethod


class SpeechToTextProvider(Enum):
    """Supported speech-to-text providers"""
    OPENAI_WHISPER = "openai_whisper"
    GOOGLE_SPEECH = "google_speech"
    AZURE_SPEECH = "azure_speech"


@dataclass
class VoiceInputResult:
    """Result of voice input processing"""
    raw_text: str
    mathematical_expression: str
    confidence_score: float
    recognized_domain: str
    validation_required: bool
    suggestions: List[str]


@dataclass
class MathematicalPhrase:
    """Mapping between spoken phrases and mathematical notation"""
    spoken_phrase: str
    mathematical_notation: str
    domain: str
    confidence_weight: float


class SpeechToTextAPI(ABC):
    """Abstract base class for speech-to-text API integrations"""
    
    @abstractmethod
    def transcribe_audio(self, audio_data: bytes, language: str = "en-US") -> Tuple[str, float]:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Raw audio bytes
            language: Language code for transcription
            
        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        pass


class OpenAIWhisperAPI(SpeechToTextAPI):
    """OpenAI Whisper API integration for speech-to-text"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/audio/transcriptions"
    
    def transcribe_audio(self, audio_data: bytes, language: str = "en-US") -> Tuple[str, float]:
        """Transcribe audio using OpenAI Whisper API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            files = {
                "file": ("audio.wav", audio_data, "audio/wav"),
                "model": (None, "whisper-1"),
                "language": (None, language[:2])  # Whisper uses 2-letter codes
            }
            
            response = requests.post(self.base_url, headers=headers, files=files)
            response.raise_for_status()
            
            result = response.json()
            # Whisper doesn't provide confidence scores, so we estimate based on response
            confidence = 0.9 if len(result.get("text", "")) > 0 else 0.0
            
            return result.get("text", ""), confidence
            
        except Exception as e:
            logging.error(f"OpenAI Whisper API error: {e}")
            return "", 0.0


class MathematicalNotationConverter:
    """Converts spoken mathematical language to mathematical notation"""
    
    def __init__(self):
        self.mathematical_phrases = self._load_mathematical_phrases()
        self.domain_keywords = self._load_domain_keywords()
    
    def _load_mathematical_phrases(self) -> List[MathematicalPhrase]:
        """Load mathematical phrase mappings"""
        phrases = [
            # Basic operations
            MathematicalPhrase("plus", "+", "arithmetic", 1.0),
            MathematicalPhrase("add", "+", "arithmetic", 0.9),
            MathematicalPhrase("minus", "-", "arithmetic", 1.0),
            MathematicalPhrase("subtract", "-", "arithmetic", 0.9),
            MathematicalPhrase("times", "*", "arithmetic", 1.0),
            MathematicalPhrase("multiply", "*", "arithmetic", 0.9),
            MathematicalPhrase("divided by", "/", "arithmetic", 1.0),
            MathematicalPhrase("over", "/", "arithmetic", 0.8),
            
            # Powers and roots
            MathematicalPhrase("squared", "^2", "algebra", 1.0),
            MathematicalPhrase("cubed", "^3", "algebra", 1.0),
            MathematicalPhrase("to the power of", "^", "algebra", 1.0),
            MathematicalPhrase("square root of", "sqrt(", "algebra", 1.0),
            MathematicalPhrase("cube root of", "cbrt(", "algebra", 1.0),
            
            # Equations
            MathematicalPhrase("equals", "=", "algebra", 1.0),
            MathematicalPhrase("is equal to", "=", "algebra", 0.9),
            MathematicalPhrase("greater than", ">", "algebra", 1.0),
            MathematicalPhrase("less than", "<", "algebra", 1.0),
            MathematicalPhrase("greater than or equal to", ">=", "algebra", 1.0),
            MathematicalPhrase("less than or equal to", "<=", "algebra", 1.0),
            
            # Calculus
            MathematicalPhrase("derivative of", "d/dx", "calculus", 1.0),
            MathematicalPhrase("integral of", "∫", "calculus", 1.0),
            MathematicalPhrase("limit of", "lim", "calculus", 1.0),
            MathematicalPhrase("approaches", "→", "calculus", 1.0),
            
            # Functions
            MathematicalPhrase("sine of", "sin(", "trigonometry", 1.0),
            MathematicalPhrase("cosine of", "cos(", "trigonometry", 1.0),
            MathematicalPhrase("tangent of", "tan(", "trigonometry", 1.0),
            MathematicalPhrase("natural log of", "ln(", "algebra", 1.0),
            MathematicalPhrase("log of", "log(", "algebra", 1.0),
            MathematicalPhrase("exponential", "exp(", "algebra", 1.0),
            
            # Linear algebra
            MathematicalPhrase("matrix", "matrix", "linear_algebra", 1.0),
            MathematicalPhrase("vector", "vector", "linear_algebra", 1.0),
            MathematicalPhrase("dot product", "·", "linear_algebra", 1.0),
            MathematicalPhrase("cross product", "×", "linear_algebra", 1.0),
            
            # Parentheses and grouping
            MathematicalPhrase("open parenthesis", "(", "general", 1.0),
            MathematicalPhrase("close parenthesis", ")", "general", 1.0),
            MathematicalPhrase("open bracket", "[", "general", 1.0),
            MathematicalPhrase("close bracket", "]", "general", 1.0),
        ]
        return phrases
    
    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """Load keywords that help identify mathematical domains"""
        return {
            "algebra": ["equation", "solve", "variable", "polynomial", "factor"],
            "calculus": ["derivative", "integral", "limit", "continuous", "function"],
            "linear_algebra": ["matrix", "vector", "eigenvalue", "determinant", "system"],
            "trigonometry": ["sine", "cosine", "tangent", "angle", "triangle"],
            "statistics": ["mean", "median", "standard deviation", "probability"],
            "geometry": ["area", "perimeter", "volume", "angle", "triangle", "circle"]
        }
    
    def convert_to_mathematical_notation(self, spoken_text: str) -> VoiceInputResult:
        """
        Convert spoken mathematical language to mathematical notation
        
        Args:
            spoken_text: Raw transcribed text from speech-to-text
            
        Returns:
            VoiceInputResult with converted mathematical expression
        """
        # Clean and normalize the input text
        normalized_text = self._normalize_text(spoken_text)
        
        # Convert numbers from words to digits
        text_with_numbers = self._convert_number_words(normalized_text)
        
        # Apply mathematical phrase conversions
        mathematical_expression = self._apply_phrase_conversions(text_with_numbers)
        
        # Identify mathematical domain
        domain = self._identify_domain(normalized_text)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(spoken_text, mathematical_expression)
        
        # Determine if validation is required
        validation_required = confidence < 0.8 or self._has_ambiguous_terms(normalized_text)
        
        # Generate suggestions for improvement (use original text, not normalized)
        suggestions = self._generate_suggestions(spoken_text, mathematical_expression)
        
        return VoiceInputResult(
            raw_text=spoken_text,
            mathematical_expression=mathematical_expression,
            confidence_score=confidence,
            recognized_domain=domain,
            validation_required=validation_required,
            suggestions=suggestions
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing"""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove filler words
        filler_words = ["um", "uh", "like", "you know", "so", "well"]
        for filler in filler_words:
            text = re.sub(rf'\b{filler}\b', '', text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _convert_number_words(self, text: str) -> str:
        """Convert number words to digits"""
        number_words = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
            "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
            "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
            "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
            "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000"
        }
        
        for word, digit in number_words.items():
            text = re.sub(rf'\b{word}\b', digit, text)
        
        return text
    
    def _apply_phrase_conversions(self, text: str) -> str:
        """Apply mathematical phrase to notation conversions"""
        result = text
        
        # Sort phrases by length (longest first) to avoid partial matches
        sorted_phrases = sorted(self.mathematical_phrases, 
                              key=lambda x: len(x.spoken_phrase), reverse=True)
        
        for phrase in sorted_phrases:
            pattern = rf'\b{re.escape(phrase.spoken_phrase)}\b'
            result = re.sub(pattern, phrase.mathematical_notation, result)
        
        # Handle special cases for function parentheses
        result = self._fix_function_parentheses(result)
        
        # Clean up spacing around operators
        result = self._clean_operator_spacing(result)
        
        return result
    
    def _fix_function_parentheses(self, text: str) -> str:
        """Fix function parentheses that may have been separated"""
        functions = ["sin", "cos", "tan", "ln", "log", "exp", "sqrt", "cbrt"]
        
        for func in functions:
            # Pattern: function followed by opening parenthesis
            pattern = rf'{func}\s*\('
            replacement = f'{func}('
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _clean_operator_spacing(self, text: str) -> str:
        """Clean up spacing around mathematical operators"""
        operators = ["+", "-", "*", "/", "=", ">", "<", ">=", "<="]
        
        for op in operators:
            # Add spaces around operators, but be careful with ^ operator
            if op != "^":
                pattern = rf'\s*{re.escape(op)}\s*'
                replacement = f' {op} '
                text = re.sub(pattern, replacement, text)
        
        # Handle ^ operator specially (no spaces)
        text = re.sub(r'\s*\^\s*', '^', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _identify_domain(self, text: str) -> str:
        """Identify the mathematical domain based on keywords"""
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    score += 1
            domain_scores[domain] = score
        
        # Return domain with highest score, or "general" if no clear domain
        if domain_scores and max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        
        return "general"
    
    def _calculate_confidence(self, original_text: str, converted_text: str) -> float:
        """Calculate confidence score for the conversion"""
        # Base confidence starts at 0.5
        confidence = 0.5
        
        # Increase confidence if mathematical notation was found
        math_symbols = ["+", "-", "*", "/", "=", "^", "sqrt", "sin", "cos", "tan"]
        found_symbols = sum(1 for symbol in math_symbols if symbol in converted_text)
        confidence += min(found_symbols * 0.1, 0.3)
        
        # Decrease confidence if original text is very short or very long
        text_length = len(original_text.split())
        if text_length < 3:
            confidence -= 0.2
        elif text_length > 20:
            confidence -= 0.1
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    def _has_ambiguous_terms(self, text: str) -> bool:
        """Check if text contains ambiguous terms that need validation"""
        ambiguous_terms = ["over", "by", "of", "to", "from"]
        return any(term in text for term in ambiguous_terms)
    
    def _generate_suggestions(self, original_text: str, converted_text: str) -> List[str]:
        """Generate suggestions for improving voice input"""
        suggestions = []
        
        if len(original_text.split()) < 3:
            suggestions.append("Try speaking more complete mathematical expressions")
        
        # Check original text for filler words before normalization
        if "um" in original_text.lower() or "uh" in original_text.lower():
            suggestions.append("Speak clearly without filler words for better recognition")
        
        if converted_text.strip() == original_text.strip():
            suggestions.append("No mathematical notation detected. Try using mathematical terms like 'plus', 'equals', or 'squared'")
        
        return suggestions


class VoiceInputProcessor:
    """Main class for processing voice input for mathematical problems"""
    
    def __init__(self, speech_api: SpeechToTextAPI):
        self.speech_api = speech_api
        self.notation_converter = MathematicalNotationConverter()
        self.logger = logging.getLogger(__name__)
    
    def process_voice_input(self, audio_data: bytes, language: str = "en-US") -> VoiceInputResult:
        """
        Process voice input from audio data to mathematical notation
        
        Args:
            audio_data: Raw audio bytes
            language: Language code for speech recognition
            
        Returns:
            VoiceInputResult with processed mathematical expression
        """
        try:
            # Step 1: Convert speech to text
            transcribed_text, speech_confidence = self.speech_api.transcribe_audio(
                audio_data, language
            )
            
            if not transcribed_text:
                return VoiceInputResult(
                    raw_text="",
                    mathematical_expression="",
                    confidence_score=0.0,
                    recognized_domain="unknown",
                    validation_required=True,
                    suggestions=["Audio could not be transcribed. Please try speaking more clearly."]
                )
            
            self.logger.info(f"Transcribed text: {transcribed_text}")
            
            # Step 2: Convert to mathematical notation
            result = self.notation_converter.convert_to_mathematical_notation(transcribed_text)
            
            # Step 3: Adjust confidence based on speech recognition confidence
            result.confidence_score = (result.confidence_score + speech_confidence) / 2
            
            self.logger.info(f"Converted to mathematical notation: {result.mathematical_expression}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing voice input: {e}")
            return VoiceInputResult(
                raw_text="",
                mathematical_expression="",
                confidence_score=0.0,
                recognized_domain="unknown",
                validation_required=True,
                suggestions=["An error occurred processing your voice input. Please try again."]
            )
    
    def validate_and_confirm(self, result: VoiceInputResult, user_confirmation: bool = None) -> Dict:
        """
        Validate voice input result and handle user confirmation
        
        Args:
            result: VoiceInputResult to validate
            user_confirmation: Optional user confirmation (True/False/None)
            
        Returns:
            Dictionary with validation status and next steps
        """
        validation_response = {
            "needs_confirmation": result.validation_required,
            "confidence_score": result.confidence_score,
            "suggested_expression": result.mathematical_expression,
            "original_speech": result.raw_text,
            "domain": result.recognized_domain,
            "suggestions": result.suggestions
        }
        
        if result.validation_required and user_confirmation is None:
            validation_response["status"] = "awaiting_confirmation"
            validation_response["message"] = f"I heard: '{result.raw_text}' and converted it to: '{result.mathematical_expression}'. Is this correct?"
        
        elif result.validation_required and user_confirmation is False:
            validation_response["status"] = "rejected"
            validation_response["message"] = "Please try speaking your mathematical problem again, or use text input."
        
        elif not result.validation_required or user_confirmation is True:
            validation_response["status"] = "confirmed"
            validation_response["message"] = "Voice input processed successfully."
            validation_response["final_expression"] = result.mathematical_expression
        
        return validation_response


# Factory function for creating voice input processor
def create_voice_input_processor(provider: SpeechToTextProvider, api_key: str = None) -> VoiceInputProcessor:
    """
    Factory function to create a VoiceInputProcessor with the specified provider
    
    Args:
        provider: Speech-to-text provider to use
        api_key: API key for the provider (if required)
        
    Returns:
        Configured VoiceInputProcessor instance
    """
    if provider == SpeechToTextProvider.OPENAI_WHISPER:
        if not api_key:
            raise ValueError("API key required for OpenAI Whisper")
        speech_api = OpenAIWhisperAPI(api_key)
    else:
        raise ValueError(f"Unsupported speech-to-text provider: {provider}")
    
    return VoiceInputProcessor(speech_api)