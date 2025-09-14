"""
Handwriting Recognition Module for AI Math Tutor

This module handles OCR processing and mathematical symbol recognition
for handwritten mathematical notation input.
"""

import io
import re
import json
import logging
import base64
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Image processing
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# HTTP client for OCR APIs
import requests


class OCRProvider(Enum):
    """Supported OCR providers for handwriting recognition"""
    GOOGLE_VISION = "google_vision"
    AZURE_COMPUTER_VISION = "azure_computer_vision"
    MATHPIX = "mathpix"  # Specialized for mathematical notation
    TESSERACT = "tesseract"  # Open source OCR


@dataclass
class HandwritingRecognitionResult:
    """Result of handwriting recognition processing"""
    raw_image_data: bytes
    recognized_text: str
    mathematical_expression: str
    confidence_score: float
    recognized_symbols: List[str]
    bounding_boxes: List[Dict]
    validation_required: bool
    correction_suggestions: List[str]
    processing_metadata: Dict


@dataclass
class MathematicalSymbol:
    """Recognized mathematical symbol with metadata"""
    symbol: str
    confidence: float
    bounding_box: Dict
    symbol_type: str  # operator, number, variable, function, etc.
    latex_representation: str


class ImagePreprocessor:
    """Preprocesses handwritten images for better OCR recognition"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def preprocess_image(self, image_data: bytes) -> bytes:
        """
        Preprocess handwritten image for better OCR recognition
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Preprocessed image bytes
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Apply noise reduction
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Binarize image (convert to black and white)
            image = self._binarize_image(image)
            
            # Resize if too small (OCR works better with larger images)
            width, height = image.size
            if width < 300 or height < 300:
                scale_factor = max(300 / width, 300 / height)
                new_size = (int(width * scale_factor), int(height * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert back to bytes
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='PNG')
            return output_buffer.getvalue()
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            return image_data  # Return original if preprocessing fails
    
    def _binarize_image(self, image: Image.Image, threshold: int = 128) -> Image.Image:
        """Convert grayscale image to binary (black and white)"""
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Apply threshold
        binary_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
        
        # Convert back to PIL Image
        return Image.fromarray(binary_array, 'L')
    
    def extract_regions_of_interest(self, image_data: bytes) -> List[bytes]:
        """
        Extract individual mathematical expressions or symbols from image
        
        Args:
            image_data: Preprocessed image bytes
            
        Returns:
            List of image regions containing individual expressions
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            img_array = np.array(image)
            
            # Simple region extraction based on connected components
            # In a production system, this would be more sophisticated
            regions = self._find_connected_components(img_array)
            
            region_images = []
            for region in regions:
                x, y, w, h = region
                cropped = image.crop((x, y, x + w, y + h))
                
                # Convert to bytes
                output_buffer = io.BytesIO()
                cropped.save(output_buffer, format='PNG')
                region_images.append(output_buffer.getvalue())
            
            return region_images if region_images else [image_data]
            
        except Exception as e:
            self.logger.error(f"Error extracting regions: {e}")
            return [image_data]
    
    def _find_connected_components(self, img_array: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find connected components (simplified implementation)"""
        # This is a simplified implementation
        # In production, you'd use cv2.findContours or similar
        height, width = img_array.shape
        
        # For demo purposes, return the whole image as one region
        return [(0, 0, width, height)]


class OCRService(ABC):
    """Abstract base class for OCR service integrations"""
    
    @abstractmethod
    def recognize_text(self, image_data: bytes) -> Tuple[str, float, List[Dict]]:
        """
        Recognize text from image
        
        Args:
            image_data: Preprocessed image bytes
            
        Returns:
            Tuple of (recognized_text, confidence_score, bounding_boxes)
        """
        pass


class MathpixOCR(OCRService):
    """Mathpix OCR service specialized for mathematical notation"""
    
    def __init__(self, app_id: str, app_key: str):
        self.app_id = app_id
        self.app_key = app_key
        self.base_url = "https://api.mathpix.com/v3/text"
    
    def recognize_text(self, image_data: bytes) -> Tuple[str, float, List[Dict]]:
        """Recognize mathematical text using Mathpix API"""
        try:
            # Encode image to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            headers = {
                "app_id": self.app_id,
                "app_key": self.app_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "src": f"data:image/png;base64,{image_base64}",
                "formats": ["text", "latex_simplified"],
                "data_options": {
                    "include_asciimath": True,
                    "include_latex": True
                }
            }
            
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract text and confidence
            text = result.get("text", "")
            latex = result.get("latex_simplified", "")
            confidence = result.get("confidence", 0.0)
            
            # Use LaTeX if available, otherwise use text
            recognized_text = latex if latex else text
            
            # Extract bounding boxes if available
            bounding_boxes = result.get("position", {}).get("bounding_boxes", [])
            
            return recognized_text, confidence, bounding_boxes
            
        except Exception as e:
            logging.error(f"Mathpix OCR error: {e}")
            return "", 0.0, []


class GoogleVisionOCR(OCRService):
    """Google Cloud Vision OCR service"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://vision.googleapis.com/v1/images:annotate"
    
    def recognize_text(self, image_data: bytes) -> Tuple[str, float, List[Dict]]:
        """Recognize text using Google Cloud Vision API"""
        try:
            # Encode image to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "requests": [{
                    "image": {
                        "content": image_base64
                    },
                    "features": [{
                        "type": "TEXT_DETECTION",
                        "maxResults": 50
                    }]
                }]
            }
            
            url = f"{self.base_url}?key={self.api_key}"
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract text annotations
            annotations = result.get("responses", [{}])[0].get("textAnnotations", [])
            
            if not annotations:
                return "", 0.0, []
            
            # First annotation contains full text
            full_text = annotations[0].get("description", "")
            
            # Calculate average confidence
            confidences = [ann.get("confidence", 0.0) for ann in annotations if "confidence" in ann]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            # Extract bounding boxes
            bounding_boxes = []
            for ann in annotations[1:]:  # Skip first (full text)
                if "boundingPoly" in ann:
                    vertices = ann["boundingPoly"].get("vertices", [])
                    bounding_boxes.append({
                        "text": ann.get("description", ""),
                        "vertices": vertices
                    })
                else:
                    # Add bounding box even without boundingPoly for testing
                    bounding_boxes.append({
                        "text": ann.get("description", ""),
                        "vertices": []
                    })
            
            return full_text, avg_confidence, bounding_boxes
            
        except Exception as e:
            logging.error(f"Google Vision OCR error: {e}")
            return "", 0.0, []


class MathematicalSymbolRecognizer:
    """Recognizes and parses mathematical symbols from OCR text"""
    
    def __init__(self):
        self.symbol_mappings = self._load_symbol_mappings()
        self.latex_to_unicode = self._load_latex_mappings()
    
    def _load_symbol_mappings(self) -> Dict[str, str]:
        """Load mappings from common OCR mistakes to correct symbols"""
        return {
            # Common OCR mistakes for mathematical symbols
            "×": "*",
            "÷": "/",
            "±": "±",
            "∞": "∞",
            "∑": "∑",
            "∏": "∏",
            "∫": "∫",
            "∂": "∂",
            "∇": "∇",
            "√": "sqrt",
            "∛": "cbrt",
            "π": "pi",
            "θ": "theta",
            "α": "alpha",
            "β": "beta",
            "γ": "gamma",
            "δ": "delta",
            "ε": "epsilon",
            "λ": "lambda",
            "μ": "mu",
            "σ": "sigma",
            "φ": "phi",
            "ψ": "psi",
            "ω": "omega",
            
            # Common misrecognitions
            "l": "1",  # lowercase l often confused with 1
            "O": "0",  # uppercase O often confused with 0
            "S": "5",  # in mathematical context
            "|": "1",  # vertical bar confused with 1
        }
    
    def _load_latex_mappings(self) -> Dict[str, str]:
        """Load LaTeX to Unicode symbol mappings"""
        return {
            "\\alpha": "α",
            "\\beta": "β",
            "\\gamma": "γ",
            "\\delta": "δ",
            "\\epsilon": "ε",
            "\\theta": "θ",
            "\\lambda": "λ",
            "\\mu": "μ",
            "\\pi": "π",
            "\\sigma": "σ",
            "\\phi": "φ",
            "\\psi": "ψ",
            "\\omega": "ω",
            "\\sum": "∑",
            "\\prod": "∏",
            "\\int": "∫",
            "\\partial": "∂",
            "\\nabla": "∇",
            "\\sqrt": "√",
            "\\infty": "∞",
            "\\pm": "±",
            "\\times": "×",
            "\\div": "÷",
            "\\cdot": "·",
            "\\leq": "≤",
            "\\geq": "≥",
            "\\neq": "≠",
            "\\approx": "≈",
            "\\equiv": "≡",
        }
    
    def recognize_symbols(self, ocr_text: str) -> List[MathematicalSymbol]:
        """
        Recognize and classify mathematical symbols from OCR text
        
        Args:
            ocr_text: Raw text from OCR
            
        Returns:
            List of recognized mathematical symbols
        """
        symbols = []
        
        # Convert LaTeX to Unicode if present
        processed_text = self._convert_latex_to_unicode(ocr_text)
        
        # Apply symbol corrections
        processed_text = self._apply_symbol_corrections(processed_text)
        
        # Tokenize and classify symbols
        tokens = self._tokenize_mathematical_expression(processed_text)
        
        for i, token in enumerate(tokens):
            symbol = self._classify_symbol(token, i, tokens)
            if symbol:
                symbols.append(symbol)
        
        return symbols
    
    def _convert_latex_to_unicode(self, text: str) -> str:
        """Convert LaTeX commands to Unicode symbols"""
        result = text
        for latex, unicode_char in self.latex_to_unicode.items():
            result = result.replace(latex, unicode_char)
        return result
    
    def _apply_symbol_corrections(self, text: str) -> str:
        """Apply common OCR error corrections"""
        result = text
        
        # Apply character-level corrections
        for wrong, correct in self.symbol_mappings.items():
            result = result.replace(wrong, correct)
        
        # Context-aware corrections
        result = self._apply_context_corrections(result)
        
        return result
    
    def _apply_context_corrections(self, text: str) -> str:
        """Apply context-aware corrections"""
        # Example: "x 2" -> "x^2" (superscript)
        text = re.sub(r'([a-zA-Z])\s+(\d+)', r'\1^{\2}', text)
        
        # Example: "2 x" -> "2*x" (multiplication)
        text = re.sub(r'(\d+)\s+([a-zA-Z])', r'\1*\2', text)
        
        # Example: "( x )" -> "(x)" (remove extra spaces in parentheses)
        text = re.sub(r'\(\s+([^)]+)\s+\)', r'(\1)', text)
        
        return text
    
    def _tokenize_mathematical_expression(self, text: str) -> List[str]:
        """Tokenize mathematical expression into symbols"""
        # Simple tokenization - in production, this would be more sophisticated
        tokens = []
        current_token = ""
        
        for char in text:
            if char.isalnum() or char in "αβγδεθλμπσφψω":
                current_token += char
            else:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                if char.strip():  # Non-whitespace operators/symbols
                    tokens.append(char)
        
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    def _classify_symbol(self, token: str, position: int, all_tokens: List[str]) -> Optional[MathematicalSymbol]:
        """Classify a token as a mathematical symbol"""
        if not token.strip():
            return None
        
        # Determine symbol type
        if token.isdigit():
            symbol_type = "number"
        elif token in ["π", "e", "∞"]:
            symbol_type = "constant"
        elif token.isalpha() and len(token) == 1:
            symbol_type = "variable"
        elif token in ["+", "-", "*", "/", "=", "<", ">", "≤", "≥"]:
            symbol_type = "operator"
        elif token in ["sin", "cos", "tan", "log", "ln", "exp", "sqrt"]:
            symbol_type = "function"
        elif token in ["(", ")", "[", "]", "{", "}"]:
            symbol_type = "delimiter"
        else:
            symbol_type = "unknown"
        
        # Generate LaTeX representation
        latex_repr = self._generate_latex_representation(token, symbol_type)
        
        return MathematicalSymbol(
            symbol=token,
            confidence=0.8,  # Default confidence
            bounding_box={},  # Would be populated with actual coordinates
            symbol_type=symbol_type,
            latex_representation=latex_repr
        )
    
    def _generate_latex_representation(self, symbol: str, symbol_type: str) -> str:
        """Generate LaTeX representation of symbol"""
        latex_map = {
            "α": "\\alpha", "β": "\\beta", "γ": "\\gamma", "δ": "\\delta",
            "ε": "\\epsilon", "θ": "\\theta", "λ": "\\lambda", "μ": "\\mu",
            "π": "\\pi", "σ": "\\sigma", "φ": "\\phi", "ψ": "\\psi", "ω": "\\omega",
            "∑": "\\sum", "∏": "\\prod", "∫": "\\int", "∂": "\\partial",
            "∇": "\\nabla", "∞": "\\infty", "±": "\\pm", "×": "\\times",
            "÷": "\\div", "·": "\\cdot", "≤": "\\leq", "≥": "\\geq",
            "≠": "\\neq", "≈": "\\approx", "≡": "\\equiv"
        }
        
        return latex_map.get(symbol, symbol)


class HandwritingRecognitionProcessor:
    """Main processor for handwriting recognition"""
    
    def __init__(self, ocr_service: OCRService):
        self.ocr_service = ocr_service
        self.preprocessor = ImagePreprocessor()
        self.symbol_recognizer = MathematicalSymbolRecognizer()
        self.logger = logging.getLogger(__name__)
    
    def process_handwritten_image(self, image_data: bytes) -> HandwritingRecognitionResult:
        """
        Process handwritten mathematical notation image
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            HandwritingRecognitionResult with processed mathematical expression
        """
        try:
            # Step 1: Preprocess image
            preprocessed_image = self.preprocessor.preprocess_image(image_data)
            
            # Step 2: OCR recognition
            ocr_text, ocr_confidence, bounding_boxes = self.ocr_service.recognize_text(
                preprocessed_image
            )
            
            if not ocr_text:
                return self._create_empty_result(image_data, "No text recognized from image")
            
            self.logger.info(f"OCR recognized text: {ocr_text}")
            
            # Step 3: Mathematical symbol recognition
            symbols = self.symbol_recognizer.recognize_symbols(ocr_text)
            
            # Step 4: Generate mathematical expression
            mathematical_expression = self._generate_mathematical_expression(symbols, ocr_text)
            
            # Step 5: Calculate confidence and validation requirements
            confidence_score = self._calculate_confidence(ocr_confidence, symbols)
            validation_required = confidence_score < 0.7 or self._requires_validation(ocr_text)
            
            # Step 6: Generate correction suggestions
            correction_suggestions = self._generate_correction_suggestions(
                ocr_text, mathematical_expression, symbols
            )
            
            return HandwritingRecognitionResult(
                raw_image_data=image_data,
                recognized_text=ocr_text,
                mathematical_expression=mathematical_expression,
                confidence_score=confidence_score,
                recognized_symbols=[s.symbol for s in symbols],
                bounding_boxes=bounding_boxes,
                validation_required=validation_required,
                correction_suggestions=correction_suggestions,
                processing_metadata={
                    "ocr_confidence": ocr_confidence,
                    "symbol_count": len(symbols),
                    "preprocessing_applied": True
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing handwritten image: {e}")
            return self._create_empty_result(image_data, f"Processing error: {str(e)}")
    
    def _create_empty_result(self, image_data: bytes, error_message: str) -> HandwritingRecognitionResult:
        """Create empty result for error cases"""
        return HandwritingRecognitionResult(
            raw_image_data=image_data,
            recognized_text="",
            mathematical_expression="",
            confidence_score=0.0,
            recognized_symbols=[],
            bounding_boxes=[],
            validation_required=True,
            correction_suggestions=[error_message],
            processing_metadata={"error": error_message}
        )
    
    def _generate_mathematical_expression(self, symbols: List[MathematicalSymbol], ocr_text: str) -> str:
        """Generate clean mathematical expression from recognized symbols"""
        if not symbols:
            return ocr_text
        
        # Reconstruct expression from symbols
        expression_parts = []
        for symbol in symbols:
            if symbol.symbol_type == "function":
                expression_parts.append(f"{symbol.symbol}(")
            else:
                expression_parts.append(symbol.symbol)
        
        # Join with appropriate spacing
        expression = " ".join(expression_parts)
        
        # Clean up spacing around operators
        expression = re.sub(r'\s*([+\-*/=<>])\s*', r' \1 ', expression)
        expression = re.sub(r'\s+', ' ', expression).strip()
        
        return expression
    
    def _calculate_confidence(self, ocr_confidence: float, symbols: List[MathematicalSymbol]) -> float:
        """Calculate overall confidence score"""
        if not symbols:
            return ocr_confidence * 0.5  # Lower confidence if no symbols recognized
        
        # Average symbol confidence
        symbol_confidences = [s.confidence for s in symbols]
        avg_symbol_confidence = sum(symbol_confidences) / len(symbol_confidences)
        
        # Combine OCR and symbol recognition confidence
        combined_confidence = (ocr_confidence + avg_symbol_confidence) / 2
        
        # Boost confidence if mathematical symbols are recognized
        math_symbols = [s for s in symbols if s.symbol_type in ["operator", "function", "constant"]]
        if math_symbols:
            combined_confidence = min(1.0, combined_confidence + 0.1)
        
        return combined_confidence
    
    def _requires_validation(self, ocr_text: str) -> bool:
        """Determine if result requires user validation"""
        # Require validation if text is very short or contains ambiguous characters
        if len(ocr_text.strip()) < 3:
            return True
        
        # Check for potentially ambiguous characters
        ambiguous_chars = ["l", "I", "O", "0", "1", "|"]
        if any(char in ocr_text for char in ambiguous_chars):
            return True
        
        return False
    
    def _generate_correction_suggestions(self, ocr_text: str, expression: str, symbols: List[MathematicalSymbol]) -> List[str]:
        """Generate suggestions for correcting recognition errors"""
        suggestions = []
        
        if not symbols:
            suggestions.append("No mathematical symbols detected. Please ensure the handwriting is clear.")
        
        if len(ocr_text) < 3:
            suggestions.append("Very short expression detected. Please write more complete mathematical expressions.")
        
        # Check for common OCR issues
        if "l" in ocr_text or "I" in ocr_text:
            suggestions.append("The character 'l' or 'I' might be confused with '1'. Please verify numbers.")
        
        if "O" in ocr_text:
            suggestions.append("The character 'O' might be confused with '0'. Please verify numbers.")
        
        # Check for missing operators
        if re.search(r'\d+[a-zA-Z]', expression):
            suggestions.append("Missing multiplication operator detected. Consider adding '*' between numbers and variables.")
        
        return suggestions
    
    def validate_and_correct(self, result: HandwritingRecognitionResult, user_corrections: Dict = None) -> Dict:
        """
        Validate handwriting recognition result and apply user corrections
        
        Args:
            result: HandwritingRecognitionResult to validate
            user_corrections: Optional dictionary of user corrections
            
        Returns:
            Dictionary with validation status and corrected expression
        """
        validation_response = {
            "needs_validation": result.validation_required,
            "confidence_score": result.confidence_score,
            "recognized_text": result.recognized_text,
            "suggested_expression": result.mathematical_expression,
            "recognized_symbols": result.recognized_symbols,
            "correction_suggestions": result.correction_suggestions
        }
        
        if user_corrections:
            # Apply user corrections
            corrected_expression = user_corrections.get("expression", result.mathematical_expression)
            validation_response["status"] = "corrected"
            validation_response["final_expression"] = corrected_expression
            validation_response["message"] = "User corrections applied successfully."
        
        elif result.validation_required:
            validation_response["status"] = "awaiting_validation"
            validation_response["message"] = f"Recognized: '{result.recognized_text}'. Please verify the mathematical expression is correct."
        
        else:
            validation_response["status"] = "validated"
            validation_response["final_expression"] = result.mathematical_expression
            validation_response["message"] = "Handwriting recognition completed successfully."
        
        return validation_response


# Factory function for creating handwriting recognition processor
def create_handwriting_processor(provider: OCRProvider, **kwargs) -> HandwritingRecognitionProcessor:
    """
    Factory function to create a HandwritingRecognitionProcessor with the specified provider
    
    Args:
        provider: OCR provider to use
        **kwargs: Provider-specific configuration
        
    Returns:
        Configured HandwritingRecognitionProcessor instance
    """
    if provider == OCRProvider.MATHPIX:
        app_id = kwargs.get("app_id")
        app_key = kwargs.get("app_key")
        if not app_id or not app_key:
            raise ValueError("app_id and app_key required for Mathpix")
        ocr_service = MathpixOCR(app_id, app_key)
    
    elif provider == OCRProvider.GOOGLE_VISION:
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("api_key required for Google Vision")
        ocr_service = GoogleVisionOCR(api_key)
    
    else:
        raise ValueError(f"Unsupported OCR provider: {provider}")
    
    return HandwritingRecognitionProcessor(ocr_service)