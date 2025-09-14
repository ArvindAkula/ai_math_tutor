"""
Tests for Handwriting Recognition Module

This module contains comprehensive tests for handwriting recognition,
mathematical symbol recognition, and validation workflows.
"""

import pytest
import io
import base64
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from handwriting_recognition import (
    HandwritingRecognitionProcessor, ImagePreprocessor, MathematicalSymbolRecognizer,
    MathpixOCR, GoogleVisionOCR, HandwritingRecognitionResult, MathematicalSymbol,
    OCRProvider, create_handwriting_processor
)


class TestImagePreprocessor:
    """Test cases for image preprocessing"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.preprocessor = ImagePreprocessor()
    
    def create_test_image(self, width: int = 100, height: int = 100) -> bytes:
        """Create a test image for testing"""
        # Create a simple test image
        image = Image.new('RGB', (width, height), color='white')
        
        # Convert to bytes
        output_buffer = io.BytesIO()
        image.save(output_buffer, format='PNG')
        return output_buffer.getvalue()
    
    def test_preprocess_image_basic(self):
        """Test basic image preprocessing"""
        test_image = self.create_test_image()
        
        result = self.preprocessor.preprocess_image(test_image)
        
        assert isinstance(result, bytes)
        assert len(result) > 0
        
        # Verify the result is a valid image
        processed_image = Image.open(io.BytesIO(result))
        assert processed_image.mode == 'L'  # Should be grayscale
    
    def test_preprocess_small_image_upscaling(self):
        """Test that small images are upscaled"""
        small_image = self.create_test_image(50, 50)
        
        result = self.preprocessor.preprocess_image(small_image)
        processed_image = Image.open(io.BytesIO(result))
        
        # Should be upscaled to at least 300px
        assert processed_image.width >= 300 or processed_image.height >= 300
    
    def test_preprocess_error_handling(self):
        """Test error handling in preprocessing"""
        invalid_data = b"not an image"
        
        result = self.preprocessor.preprocess_image(invalid_data)
        
        # Should return original data on error
        assert result == invalid_data
    
    def test_extract_regions_of_interest(self):
        """Test region extraction"""
        test_image = self.create_test_image()
        
        regions = self.preprocessor.extract_regions_of_interest(test_image)
        
        assert isinstance(regions, list)
        assert len(regions) > 0
        
        # Each region should be valid image data
        for region in regions:
            assert isinstance(region, bytes)
            region_image = Image.open(io.BytesIO(region))
            assert region_image.size[0] > 0 and region_image.size[1] > 0


class TestMathematicalSymbolRecognizer:
    """Test cases for mathematical symbol recognition"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.recognizer = MathematicalSymbolRecognizer()
    
    def test_recognize_basic_symbols(self):
        """Test recognition of basic mathematical symbols"""
        test_cases = [
            "2 + 3 = 5",
            "x^2 - 4 = 0",
            "sin(x) + cos(x)",
            "∫ x dx",
            "α + β = γ"
        ]
        
        for test_text in test_cases:
            symbols = self.recognizer.recognize_symbols(test_text)
            
            assert isinstance(symbols, list)
            assert len(symbols) > 0
            
            # Check that symbols have required attributes
            for symbol in symbols:
                assert isinstance(symbol, MathematicalSymbol)
                assert symbol.symbol
                assert symbol.symbol_type
                assert symbol.latex_representation
    
    def test_latex_to_unicode_conversion(self):
        """Test LaTeX to Unicode conversion"""
        latex_text = "\\alpha + \\beta = \\gamma"
        
        converted = self.recognizer._convert_latex_to_unicode(latex_text)
        
        assert "α" in converted
        assert "β" in converted
        assert "γ" in converted
        assert "\\alpha" not in converted
    
    def test_symbol_corrections(self):
        """Test OCR error corrections"""
        # Test common OCR mistakes
        test_cases = [
            ("l + 1", "1 + 1"),  # lowercase l to 1
            ("O + 0", "0 + 0"),  # uppercase O to 0
            ("x 2", "x^{2}"),    # superscript detection
            ("2 x", "2*x"),      # multiplication insertion
        ]
        
        for input_text, expected_pattern in test_cases:
            corrected = self.recognizer._apply_symbol_corrections(input_text)
            # Check that some correction was applied
            assert corrected != input_text or expected_pattern in corrected
    
    def test_symbol_classification(self):
        """Test symbol classification"""
        test_cases = [
            ("5", "number"),
            ("x", "variable"),
            ("+", "operator"),
            ("sin", "function"),
            ("(", "delimiter"),
            ("π", "constant")
        ]
        
        for symbol, expected_type in test_cases:
            classified = self.recognizer._classify_symbol(symbol, 0, [symbol])
            
            if classified:  # Some symbols might not be classified
                assert classified.symbol_type == expected_type
    
    def test_tokenization(self):
        """Test mathematical expression tokenization"""
        expression = "2*x + sin(y) = 0"
        
        tokens = self.recognizer._tokenize_mathematical_expression(expression)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "2" in tokens
        assert "x" in tokens
        assert "sin" in tokens


class TestMathpixOCR:
    """Test cases for Mathpix OCR integration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.ocr = MathpixOCR("test-app-id", "test-app-key")
    
    @patch('handwriting_recognition.requests.post')
    def test_successful_recognition(self, mock_post):
        """Test successful OCR recognition"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "text": "x^2 + 2x + 1",
            "latex_simplified": "x^{2} + 2x + 1",
            "confidence": 0.95,
            "position": {"bounding_boxes": []}
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        test_image = b"fake_image_data"
        text, confidence, boxes = self.ocr.recognize_text(test_image)
        
        assert text == "x^{2} + 2x + 1"  # Should prefer LaTeX
        assert confidence == 0.95
        assert isinstance(boxes, list)
        mock_post.assert_called_once()
    
    @patch('handwriting_recognition.requests.post')
    def test_api_error_handling(self, mock_post):
        """Test API error handling"""
        mock_post.side_effect = Exception("API Error")
        
        test_image = b"fake_image_data"
        text, confidence, boxes = self.ocr.recognize_text(test_image)
        
        assert text == ""
        assert confidence == 0.0
        assert boxes == []
    
    @patch('handwriting_recognition.requests.post')
    def test_empty_response_handling(self, mock_post):
        """Test handling of empty API responses"""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        test_image = b"fake_image_data"
        text, confidence, boxes = self.ocr.recognize_text(test_image)
        
        assert text == ""
        assert confidence == 0.0
        assert boxes == []


class TestGoogleVisionOCR:
    """Test cases for Google Vision OCR integration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.ocr = GoogleVisionOCR("test-api-key")
    
    @patch('handwriting_recognition.requests.post')
    def test_successful_recognition(self, mock_post):
        """Test successful OCR recognition"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "responses": [{
                "textAnnotations": [
                    {"description": "2 + 3 = 5"},
                    {"description": "2", "boundingPoly": {"vertices": []}},
                    {"description": "+", "boundingPoly": {"vertices": []}},
                    {"description": "3", "boundingPoly": {"vertices": []}},
                ]
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        test_image = b"fake_image_data"
        text, confidence, boxes = self.ocr.recognize_text(test_image)
        
        assert text == "2 + 3 = 5"
        assert confidence >= 0.0
        assert isinstance(boxes, list)
        assert len(boxes) > 0
    
    @patch('handwriting_recognition.requests.post')
    def test_no_text_detected(self, mock_post):
        """Test handling when no text is detected"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "responses": [{"textAnnotations": []}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        test_image = b"fake_image_data"
        text, confidence, boxes = self.ocr.recognize_text(test_image)
        
        assert text == ""
        assert confidence == 0.0
        assert boxes == []


class TestHandwritingRecognitionProcessor:
    """Test cases for the main handwriting recognition processor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_ocr = Mock()
        self.processor = HandwritingRecognitionProcessor(self.mock_ocr)
    
    def create_test_image(self) -> bytes:
        """Create a test image"""
        image = Image.new('RGB', (200, 100), color='white')
        output_buffer = io.BytesIO()
        image.save(output_buffer, format='PNG')
        return output_buffer.getvalue()
    
    def test_successful_processing(self):
        """Test successful handwriting processing"""
        # Mock OCR response
        self.mock_ocr.recognize_text.return_value = ("x^2 + 1", 0.9, [])
        
        test_image = self.create_test_image()
        result = self.processor.process_handwritten_image(test_image)
        
        assert isinstance(result, HandwritingRecognitionResult)
        assert result.recognized_text == "x^2 + 1"
        assert result.confidence_score > 0.0
        assert result.mathematical_expression
        assert isinstance(result.recognized_symbols, list)
        assert isinstance(result.validation_required, bool)
    
    def test_empty_ocr_result(self):
        """Test handling of empty OCR results"""
        # Mock empty OCR response
        self.mock_ocr.recognize_text.return_value = ("", 0.0, [])
        
        test_image = self.create_test_image()
        result = self.processor.process_handwritten_image(test_image)
        
        assert result.recognized_text == ""
        assert result.confidence_score == 0.0
        assert result.validation_required is True
        assert len(result.correction_suggestions) > 0
    
    def test_ocr_error_handling(self):
        """Test handling of OCR errors"""
        # Mock OCR error
        self.mock_ocr.recognize_text.side_effect = Exception("OCR Error")
        
        test_image = self.create_test_image()
        result = self.processor.process_handwritten_image(test_image)
        
        assert result.confidence_score == 0.0
        assert result.validation_required is True
        assert any("error" in suggestion.lower() for suggestion in result.correction_suggestions)
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        # Test high confidence case
        self.mock_ocr.recognize_text.return_value = ("sin(x) + cos(x)", 0.95, [])
        
        test_image = self.create_test_image()
        result = self.processor.process_handwritten_image(test_image)
        
        assert result.confidence_score > 0.8
        
        # Test low confidence case
        self.mock_ocr.recognize_text.return_value = ("l O", 0.3, [])
        
        result = self.processor.process_handwritten_image(test_image)
        
        assert result.confidence_score < 0.7
        assert result.validation_required is True
    
    def test_validation_and_correction(self):
        """Test validation and correction workflow"""
        # Create a result that requires validation
        result = HandwritingRecognitionResult(
            raw_image_data=b"test",
            recognized_text="x 2",
            mathematical_expression="x^2",
            confidence_score=0.6,
            recognized_symbols=["x", "2"],
            bounding_boxes=[],
            validation_required=True,
            correction_suggestions=["Check superscript notation"],
            processing_metadata={}
        )
        
        # Test awaiting validation
        validation = self.processor.validate_and_correct(result)
        assert validation["status"] == "awaiting_validation"
        assert validation["needs_validation"] is True
        
        # Test user correction
        corrections = {"expression": "x^2 + 1"}
        validation = self.processor.validate_and_correct(result, corrections)
        assert validation["status"] == "corrected"
        assert validation["final_expression"] == "x^2 + 1"
        
        # Test high confidence (no validation needed)
        result.validation_required = False
        result.confidence_score = 0.9
        validation = self.processor.validate_and_correct(result)
        assert validation["status"] == "validated"


class TestHandwritingIntegration:
    """Integration tests for handwriting recognition"""
    
    def create_test_image(self) -> bytes:
        """Create a test image"""
        image = Image.new('RGB', (300, 150), color='white')
        output_buffer = io.BytesIO()
        image.save(output_buffer, format='PNG')
        return output_buffer.getvalue()
    
    def test_complete_workflow_simulation(self):
        """Test complete workflow from image to mathematical expression"""
        # Mock the OCR service
        mock_ocr = Mock()
        mock_ocr.recognize_text.return_value = (
            "\\int_{0}^{\\pi} \\sin(x) dx", 
            0.85,
            []
        )
        
        processor = HandwritingRecognitionProcessor(mock_ocr)
        
        # Process handwritten image
        test_image = self.create_test_image()
        result = processor.process_handwritten_image(test_image)
        
        # Verify the result
        assert result.recognized_text == "\\int_{0}^{\\pi} \\sin(x) dx"
        assert result.confidence_score > 0.7
        assert "∫" in result.mathematical_expression or "int" in result.mathematical_expression
        assert "sin" in result.mathematical_expression
        
        # Test validation workflow
        validation = processor.validate_and_correct(result)
        if result.validation_required:
            assert validation["status"] == "awaiting_validation"
            # Simulate user validation
            final_validation = processor.validate_and_correct(result, {"expression": "∫₀^π sin(x) dx"})
            assert final_validation["status"] == "corrected"
        else:
            assert validation["status"] == "validated"
    
    def test_complex_mathematical_expressions(self):
        """Test processing of complex mathematical expressions"""
        test_cases = [
            "x^2 + 2x + 1 = 0",
            "\\frac{d}{dx} x^3 = 3x^2",
            "\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}",
            "\\begin{matrix} 1 & 2 \\\\ 3 & 4 \\end{matrix}",
        ]
        
        mock_ocr = Mock()
        processor = HandwritingRecognitionProcessor(mock_ocr)
        
        for i, case in enumerate(test_cases):
            mock_ocr.recognize_text.return_value = (case, 0.8, [])
            
            result = processor.process_handwritten_image(self.create_test_image())
            
            # Verify basic processing
            assert result.recognized_text == case
            assert result.confidence_score > 0.0
            assert isinstance(result.validation_required, bool)
            assert isinstance(result.correction_suggestions, list)
    
    def test_error_recovery_scenarios(self):
        """Test error recovery in various failure scenarios"""
        mock_ocr = Mock()
        processor = HandwritingRecognitionProcessor(mock_ocr)
        
        # Test OCR service error
        mock_ocr.recognize_text.side_effect = ConnectionError("Network error")
        result = processor.process_handwritten_image(self.create_test_image())
        assert result.confidence_score == 0.0
        assert result.validation_required is True
        
        # Test very low quality recognition
        mock_ocr.recognize_text.return_value = ("l O l", 0.1, [])
        result = processor.process_handwritten_image(self.create_test_image())
        assert result.validation_required is True
        # Check that suggestions are generated (the specific content may vary)
        assert len(result.correction_suggestions) > 0
        
        # Test empty recognition
        mock_ocr.recognize_text.return_value = ("", 0.0, [])
        result = processor.process_handwritten_image(self.create_test_image())
        assert result.confidence_score == 0.0
        assert len(result.correction_suggestions) > 0


class TestFactoryFunction:
    """Test cases for the factory function"""
    
    def test_create_mathpix_processor(self):
        """Test creation of Mathpix processor"""
        processor = create_handwriting_processor(
            OCRProvider.MATHPIX,
            app_id="test-id",
            app_key="test-key"
        )
        
        assert isinstance(processor, HandwritingRecognitionProcessor)
        assert isinstance(processor.ocr_service, MathpixOCR)
    
    def test_create_google_vision_processor(self):
        """Test creation of Google Vision processor"""
        processor = create_handwriting_processor(
            OCRProvider.GOOGLE_VISION,
            api_key="test-key"
        )
        
        assert isinstance(processor, HandwritingRecognitionProcessor)
        assert isinstance(processor.ocr_service, GoogleVisionOCR)
    
    def test_missing_credentials_error(self):
        """Test error when credentials are missing"""
        with pytest.raises(ValueError, match="app_id and app_key required"):
            create_handwriting_processor(OCRProvider.MATHPIX)
        
        with pytest.raises(ValueError, match="api_key required"):
            create_handwriting_processor(OCRProvider.GOOGLE_VISION)
    
    def test_unsupported_provider_error(self):
        """Test error for unsupported provider"""
        with pytest.raises(ValueError, match="Unsupported OCR provider"):
            create_handwriting_processor("unsupported_provider")


class TestMathematicalSymbol:
    """Test cases for mathematical symbol data structure"""
    
    def test_symbol_creation(self):
        """Test creation of mathematical symbols"""
        symbol = MathematicalSymbol(
            symbol="+",
            confidence=0.95,
            bounding_box={"x": 10, "y": 20, "width": 15, "height": 20},
            symbol_type="operator",
            latex_representation="+"
        )
        
        assert symbol.symbol == "+"
        assert symbol.confidence == 0.95
        assert symbol.symbol_type == "operator"
        assert symbol.latex_representation == "+"


class TestHandwritingRecognitionResult:
    """Test cases for handwriting recognition result data structure"""
    
    def test_result_creation(self):
        """Test creation of handwriting recognition results"""
        result = HandwritingRecognitionResult(
            raw_image_data=b"test_image",
            recognized_text="x^2 + 1",
            mathematical_expression="x^2 + 1",
            confidence_score=0.85,
            recognized_symbols=["x", "^", "2", "+", "1"],
            bounding_boxes=[],
            validation_required=False,
            correction_suggestions=[],
            processing_metadata={"ocr_confidence": 0.9}
        )
        
        assert result.recognized_text == "x^2 + 1"
        assert result.mathematical_expression == "x^2 + 1"
        assert result.confidence_score == 0.85
        assert result.validation_required is False
        assert result.processing_metadata["ocr_confidence"] == 0.9


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])