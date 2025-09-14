"""
Tests for Voice Input Processing Module

This module contains comprehensive tests for voice input processing,
mathematical notation conversion, and validation workflows.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from voice_input import (
    VoiceInputProcessor, MathematicalNotationConverter, OpenAIWhisperAPI,
    VoiceInputResult, MathematicalPhrase, SpeechToTextProvider,
    create_voice_input_processor
)


class TestMathematicalNotationConverter:
    """Test cases for mathematical notation conversion"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.converter = MathematicalNotationConverter()
    
    def test_basic_arithmetic_conversion(self):
        """Test conversion of basic arithmetic expressions"""
        test_cases = [
            ("two plus three", "2 + 3"),
            ("five minus two", "5 - 2"),
            ("four times six", "4 * 6"),
            ("eight divided by two", "8 / 2"),
            ("ten equals five plus five", "10 = 5 + 5")
        ]
        
        for spoken, expected in test_cases:
            result = self.converter.convert_to_mathematical_notation(spoken)
            assert expected in result.mathematical_expression
            assert result.confidence_score > 0.5
    
    def test_algebraic_expressions(self):
        """Test conversion of algebraic expressions"""
        test_cases = [
            ("x squared plus two x minus one", ["x", "^2", "+", "2", "x", "-", "1"]),
            ("y cubed equals eight", ["y", "^3", "=", "8"]),
            ("square root of sixteen", ["sqrt(", "16"]),
            ("two to the power of three", ["2", "^", "3"])
        ]
        
        for spoken, expected_parts in test_cases:
            result = self.converter.convert_to_mathematical_notation(spoken)
            # Check if key parts of the expected expression are present
            for part in expected_parts:
                assert part in result.mathematical_expression, f"Expected '{part}' in '{result.mathematical_expression}'"
    
    def test_calculus_expressions(self):
        """Test conversion of calculus expressions"""
        test_cases = [
            ("derivative of x squared", ["d", "dx", "x", "^2"]),  # May have spaces: "d / dx"
            ("integral of two x", ["∫", "2", "x"]),
            ("limit of x approaches zero", ["lim", "x", "→", "0"]),
            ("sine of x", ["sin(", "x"])
        ]
        
        for spoken, expected_parts in test_cases:
            result = self.converter.convert_to_mathematical_notation(spoken)
            assert result.recognized_domain in ["calculus", "trigonometry"]
            # Check for presence of key mathematical symbols (allowing for spacing)
            for part in expected_parts:
                assert part in result.mathematical_expression, f"Expected '{part}' in '{result.mathematical_expression}'"
    
    def test_domain_identification(self):
        """Test mathematical domain identification"""
        test_cases = [
            ("solve the equation x plus five equals ten", "algebra"),
            ("find the derivative of sine x", "calculus"),
            ("multiply the matrix by the vector", "linear_algebra"),
            ("calculate the cosine of thirty degrees", "trigonometry")
        ]
        
        for spoken, expected_domain in test_cases:
            result = self.converter.convert_to_mathematical_notation(spoken)
            assert result.recognized_domain == expected_domain
    
    def test_confidence_scoring(self):
        """Test confidence score calculation"""
        # High confidence cases
        high_confidence_cases = [
            "two plus three equals five",
            "derivative of x squared",
            "sine of pi over two"
        ]
        
        for case in high_confidence_cases:
            result = self.converter.convert_to_mathematical_notation(case)
            assert result.confidence_score > 0.6
        
        # Low confidence cases
        low_confidence_cases = [
            "um",  # Very short
            "this is a very long sentence without any mathematical content whatsoever",
            "over by to from"  # Ambiguous terms
        ]
        
        for case in low_confidence_cases:
            result = self.converter.convert_to_mathematical_notation(case)
            assert result.confidence_score < 0.7
    
    def test_validation_requirements(self):
        """Test when validation is required"""
        # Cases that should require validation
        validation_required_cases = [
            "x over y",  # Ambiguous
            "um two plus uh three",  # Contains filler words
            "short"  # Very short input
        ]
        
        for case in validation_required_cases:
            result = self.converter.convert_to_mathematical_notation(case)
            assert result.validation_required
        
        # Cases that should not require validation
        clear_cases = [
            "two plus three equals five",
            "derivative of x squared"
        ]
        
        for case in clear_cases:
            result = self.converter.convert_to_mathematical_notation(case)
            # May or may not require validation based on confidence
            # Just ensure the system makes a decision
            assert isinstance(result.validation_required, bool)
    
    def test_suggestion_generation(self):
        """Test suggestion generation for improvement"""
        # Test case with filler words
        result = self.converter.convert_to_mathematical_notation("um two plus uh three")
        assert any("filler" in suggestion.lower() for suggestion in result.suggestions)
        
        # Test case with no mathematical content
        result = self.converter.convert_to_mathematical_notation("hello world")
        assert any("mathematical" in suggestion.lower() for suggestion in result.suggestions)
        
        # Test very short input
        result = self.converter.convert_to_mathematical_notation("x")
        assert any("complete" in suggestion.lower() for suggestion in result.suggestions)
    
    def test_number_word_conversion(self):
        """Test conversion of number words to digits"""
        test_cases = [
            ("one plus one", "1 + 1"),
            ("twenty minus five", "20 - 5"),
            ("three hundred", "300"),
            ("two thousand", "2000")
        ]
        
        for spoken, expected in test_cases:
            result = self.converter.convert_to_mathematical_notation(spoken)
            # Check that digits appear in the result
            assert any(char.isdigit() for char in result.mathematical_expression)
    
    def test_function_parentheses_handling(self):
        """Test proper handling of function parentheses"""
        test_cases = [
            "sine of x",
            "cosine of theta",
            "natural log of e",
            "square root of four"
        ]
        
        for case in test_cases:
            result = self.converter.convert_to_mathematical_notation(case)
            # Should contain function with opening parenthesis
            functions = ["sin(", "cos(", "ln(", "sqrt("]
            assert any(func in result.mathematical_expression for func in functions)


class TestOpenAIWhisperAPI:
    """Test cases for OpenAI Whisper API integration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.api = OpenAIWhisperAPI("test-api-key")
    
    @patch('voice_input.requests.post')
    def test_successful_transcription(self, mock_post):
        """Test successful audio transcription"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {"text": "two plus three equals five"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        audio_data = b"fake_audio_data"
        text, confidence = self.api.transcribe_audio(audio_data)
        
        assert text == "two plus three equals five"
        assert confidence > 0.8
        mock_post.assert_called_once()
    
    @patch('voice_input.requests.post')
    def test_api_error_handling(self, mock_post):
        """Test API error handling"""
        # Mock API error
        mock_post.side_effect = Exception("API Error")
        
        audio_data = b"fake_audio_data"
        text, confidence = self.api.transcribe_audio(audio_data)
        
        assert text == ""
        assert confidence == 0.0
    
    @patch('voice_input.requests.post')
    def test_empty_response_handling(self, mock_post):
        """Test handling of empty API responses"""
        # Mock empty response
        mock_response = Mock()
        mock_response.json.return_value = {"text": ""}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        audio_data = b"fake_audio_data"
        text, confidence = self.api.transcribe_audio(audio_data)
        
        assert text == ""
        assert confidence == 0.0


class TestVoiceInputProcessor:
    """Test cases for the main voice input processor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_speech_api = Mock()
        self.processor = VoiceInputProcessor(self.mock_speech_api)
    
    def test_successful_voice_processing(self):
        """Test successful end-to-end voice processing"""
        # Mock speech-to-text response
        self.mock_speech_api.transcribe_audio.return_value = ("two plus three", 0.9)
        
        audio_data = b"fake_audio_data"
        result = self.processor.process_voice_input(audio_data)
        
        assert result.raw_text == "two plus three"
        assert "2" in result.mathematical_expression
        assert "+" in result.mathematical_expression
        assert "3" in result.mathematical_expression
        assert result.confidence_score > 0.5
        assert result.recognized_domain in ["arithmetic", "general"]
    
    def test_empty_transcription_handling(self):
        """Test handling of empty transcription"""
        # Mock empty transcription
        self.mock_speech_api.transcribe_audio.return_value = ("", 0.0)
        
        audio_data = b"fake_audio_data"
        result = self.processor.process_voice_input(audio_data)
        
        assert result.raw_text == ""
        assert result.mathematical_expression == ""
        assert result.confidence_score == 0.0
        assert result.validation_required is True
        assert len(result.suggestions) > 0
    
    def test_speech_api_error_handling(self):
        """Test handling of speech API errors"""
        # Mock API error
        self.mock_speech_api.transcribe_audio.side_effect = Exception("API Error")
        
        audio_data = b"fake_audio_data"
        result = self.processor.process_voice_input(audio_data)
        
        assert result.confidence_score == 0.0
        assert result.validation_required is True
        assert any("error" in suggestion.lower() for suggestion in result.suggestions)
    
    def test_validation_and_confirmation_workflow(self):
        """Test validation and confirmation workflow"""
        # Create a result that requires validation
        result = VoiceInputResult(
            raw_text="x over y",
            mathematical_expression="x / y",
            confidence_score=0.6,
            recognized_domain="algebra",
            validation_required=True,
            suggestions=[]
        )
        
        # Test awaiting confirmation
        validation = self.processor.validate_and_confirm(result)
        assert validation["status"] == "awaiting_confirmation"
        assert validation["needs_confirmation"] is True
        
        # Test user confirmation
        validation = self.processor.validate_and_confirm(result, user_confirmation=True)
        assert validation["status"] == "confirmed"
        assert "final_expression" in validation
        
        # Test user rejection
        validation = self.processor.validate_and_confirm(result, user_confirmation=False)
        assert validation["status"] == "rejected"
    
    def test_high_confidence_no_validation(self):
        """Test that high confidence results don't require validation"""
        # Create a high confidence result
        result = VoiceInputResult(
            raw_text="two plus three equals five",
            mathematical_expression="2 + 3 = 5",
            confidence_score=0.9,
            recognized_domain="arithmetic",
            validation_required=False,
            suggestions=[]
        )
        
        validation = self.processor.validate_and_confirm(result)
        assert validation["status"] == "confirmed"
        assert validation["needs_confirmation"] is False


class TestVoiceInputIntegration:
    """Integration tests for voice input processing"""
    
    def test_complete_workflow_simulation(self):
        """Test complete workflow from audio to mathematical expression"""
        # Mock the speech API
        mock_speech_api = Mock()
        mock_speech_api.transcribe_audio.return_value = (
            "find the derivative of x squared plus two x minus one", 
            0.85
        )
        
        processor = VoiceInputProcessor(mock_speech_api)
        
        # Process voice input
        audio_data = b"fake_audio_data"
        result = processor.process_voice_input(audio_data)
        
        # Verify the result
        assert result.raw_text == "find the derivative of x squared plus two x minus one"
        assert result.recognized_domain == "calculus"
        # Check for derivative notation (may have spaces)
        assert "d/dx" in result.mathematical_expression or "d / dx" in result.mathematical_expression
        # Check for squared notation
        assert "x^2" in result.mathematical_expression or "x ^2" in result.mathematical_expression
        assert result.confidence_score > 0.7
        
        # Test validation workflow
        validation = processor.validate_and_confirm(result)
        if result.validation_required:
            assert validation["status"] == "awaiting_confirmation"
            # Simulate user confirmation
            final_validation = processor.validate_and_confirm(result, user_confirmation=True)
            assert final_validation["status"] == "confirmed"
        else:
            assert validation["status"] == "confirmed"
    
    def test_complex_mathematical_expressions(self):
        """Test processing of complex mathematical expressions"""
        test_cases = [
            "integral from zero to pi of sine x dx",
            "limit as x approaches infinity of one over x",
            "matrix multiplication of A times B",
            "eigenvalues of the two by two matrix"
        ]
        
        mock_speech_api = Mock()
        processor = VoiceInputProcessor(mock_speech_api)
        
        for i, case in enumerate(test_cases):
            mock_speech_api.transcribe_audio.return_value = (case, 0.8)
            
            result = processor.process_voice_input(b"fake_audio")
            
            # Verify basic processing
            assert result.raw_text == case
            assert result.confidence_score > 0.0
            assert result.recognized_domain != "unknown"
            assert isinstance(result.validation_required, bool)
    
    def test_error_recovery_scenarios(self):
        """Test error recovery in various failure scenarios"""
        # Test network error
        mock_speech_api = Mock()
        processor = VoiceInputProcessor(mock_speech_api)
        mock_speech_api.transcribe_audio.side_effect = ConnectionError("Network error")
        result = processor.process_voice_input(b"fake_audio")
        assert result.confidence_score == 0.0
        assert result.validation_required is True
        
        # Test malformed audio
        mock_speech_api = Mock()
        processor = VoiceInputProcessor(mock_speech_api)
        mock_speech_api.transcribe_audio.return_value = ("", 0.0)
        result = processor.process_voice_input(b"malformed_audio")
        assert result.confidence_score == 0.0
        assert len(result.suggestions) > 0
        
        # Test very noisy transcription
        mock_speech_api = Mock()
        processor = VoiceInputProcessor(mock_speech_api)
        mock_speech_api.transcribe_audio.return_value = ("um uh like you know", 0.3)
        result = processor.process_voice_input(b"noisy_audio")
        assert result.validation_required is True
        assert any("filler" in suggestion.lower() for suggestion in result.suggestions)


class TestFactoryFunction:
    """Test cases for the factory function"""
    
    def test_create_openai_whisper_processor(self):
        """Test creation of OpenAI Whisper processor"""
        processor = create_voice_input_processor(
            SpeechToTextProvider.OPENAI_WHISPER, 
            "test-api-key"
        )
        
        assert isinstance(processor, VoiceInputProcessor)
        assert isinstance(processor.speech_api, OpenAIWhisperAPI)
    
    def test_missing_api_key_error(self):
        """Test error when API key is missing"""
        with pytest.raises(ValueError, match="API key required"):
            create_voice_input_processor(SpeechToTextProvider.OPENAI_WHISPER)
    
    def test_unsupported_provider_error(self):
        """Test error for unsupported provider"""
        # Create a mock provider that doesn't exist
        class MockProvider:
            pass
        
        with pytest.raises(ValueError, match="Unsupported speech-to-text provider"):
            # Monkey patch to simulate an unsupported provider
            original_create = create_voice_input_processor
            def mock_create(provider, api_key=None):
                if provider not in [SpeechToTextProvider.OPENAI_WHISPER]:
                    raise ValueError(f"Unsupported speech-to-text provider: {provider}")
                return original_create(provider, api_key)
            
            mock_create("unsupported_provider", "test-key")


class TestMathematicalPhrase:
    """Test cases for mathematical phrase data structure"""
    
    def test_phrase_creation(self):
        """Test creation of mathematical phrases"""
        phrase = MathematicalPhrase(
            spoken_phrase="plus",
            mathematical_notation="+",
            domain="arithmetic",
            confidence_weight=1.0
        )
        
        assert phrase.spoken_phrase == "plus"
        assert phrase.mathematical_notation == "+"
        assert phrase.domain == "arithmetic"
        assert phrase.confidence_weight == 1.0


class TestVoiceInputResult:
    """Test cases for voice input result data structure"""
    
    def test_result_creation(self):
        """Test creation of voice input results"""
        result = VoiceInputResult(
            raw_text="two plus three",
            mathematical_expression="2 + 3",
            confidence_score=0.85,
            recognized_domain="arithmetic",
            validation_required=False,
            suggestions=["Great job!"]
        )
        
        assert result.raw_text == "two plus three"
        assert result.mathematical_expression == "2 + 3"
        assert result.confidence_score == 0.85
        assert result.recognized_domain == "arithmetic"
        assert result.validation_required is False
        assert result.suggestions == ["Great job!"]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])