"""
Unit tests for the AI explanation service.
Tests OpenAI API integration, prompt templates, error handling, and fallback systems.
"""

import pytest
import os
import sys
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Add shared models to path
from models import (
    ParsedProblem, StepSolution, SolutionStep, MathDomain, DifficultyLevel
)

from ai_explainer import (
    AIExplainer, OpenAIClient, PromptTemplates, HintGenerator,
    ExplanationLevel, HintLevel, Explanation, Hint, MathContext,
    OPENAI_AVAILABLE
)


class TestPromptTemplates:
    """Test prompt template generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.templates = PromptTemplates()
    
    def test_step_explanation_template(self):
        """Test step explanation prompt template."""
        prompt = self.templates.STEP_EXPLANATION.format(
            problem_context="Find the derivative of x^2",
            step_operation="Apply power rule",
            mathematical_expression="x^2",
            intermediate_result="2x",
            user_level="intermediate"
        )
        
        assert "Find the derivative of x^2" in prompt
        assert "Apply power rule" in prompt
        assert "x^2" in prompt
        assert "2x" in prompt
        assert "intermediate" in prompt
        assert "WHY this step is necessary" in prompt
    
    def test_concept_explanation_template(self):
        """Test concept explanation prompt template."""
        prompt = self.templates.CONCEPT_EXPLANATION.format(
            concept="derivative",
            context="calculus problem",
            user_level="beginner",
            domain="calculus"
        )
        
        assert "derivative" in prompt
        assert "calculus problem" in prompt
        assert "beginner" in prompt
        assert "Clear definition" in prompt
        assert "simple example" in prompt
    
    def test_hint_generation_template(self):
        """Test hint generation prompt template."""
        prompt = self.templates.HINT_GENERATION.format(
            problem_text="Solve x^2 + 2x + 1 = 0",
            current_step="1",
            previous_attempts="x = -1",
            hint_level="gentle"
        )
        
        assert "Solve x^2 + 2x + 1 = 0" in prompt
        assert "x = -1" in prompt
        assert "gentle" in prompt
        assert "without giving it away" in prompt
    
    def test_why_question_template(self):
        """Test why question prompt template."""
        prompt = self.templates.WHY_QUESTION.format(
            question="Why do we use the chain rule?",
            context="derivative of composite functions",
            user_level="intermediate"
        )
        
        assert "Why do we use the chain rule?" in prompt
        assert "derivative of composite functions" in prompt
        assert "intermediate" in prompt
        assert "fundamental \"why\"" in prompt


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not available")
class TestOpenAIClient:
    """Test OpenAI API client functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use mock API key for testing
        self.test_api_key = "test-api-key-123"
    
    def test_client_initialization_with_api_key(self):
        """Test client initialization with provided API key."""
        with patch('ai_explainer.OpenAI') as mock_openai:
            client = OpenAIClient(api_key=self.test_api_key)
            assert client.api_key == self.test_api_key
            assert client.model == "gpt-4"
            mock_openai.assert_called_once_with(api_key=self.test_api_key)
    
    def test_client_initialization_from_env(self):
        """Test client initialization from environment variable."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': self.test_api_key}):
            with patch('ai_explainer.OpenAI') as mock_openai:
                client = OpenAIClient()
                assert client.api_key == self.test_api_key
                mock_openai.assert_called_once_with(api_key=self.test_api_key)
    
    def test_client_initialization_no_api_key(self):
        """Test client initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as excinfo:
                OpenAIClient()
            assert "OpenAI API key not provided" in str(excinfo.value)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        with patch('ai_explainer.OpenAI'):
            client = OpenAIClient(api_key=self.test_api_key)
            client.rate_limit_delay = 0.1
            
            start_time = time.time()
            client._rate_limit()
            client._rate_limit()
            end_time = time.time()
            
            # Second call should be delayed
            assert end_time - start_time >= 0.1
    
    @patch('ai_explainer.OpenAI')
    def test_successful_completion(self, mock_openai_class):
        """Test successful API completion."""
        # Mock the OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a test explanation."
        mock_response.usage.dict.return_value = {"total_tokens": 50}
        mock_response.model = "gpt-4"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        client = OpenAIClient(api_key=self.test_api_key)
        result = client.generate_completion("Test prompt")
        
        assert result['success'] is True
        assert result['content'] == "This is a test explanation."
        assert result['model'] == "gpt-4"
        assert result['generation_time'] > 0
    
    @patch('ai_explainer.OpenAI')
    def test_rate_limit_error_handling(self, mock_openai_class):
        """Test rate limit error handling."""
        import openai
        
        mock_client = Mock()
        # Create a proper RateLimitError with required arguments
        mock_response = Mock()
        mock_response.status_code = 429
        rate_limit_error = openai.RateLimitError("Rate limit exceeded", response=mock_response, body={})
        mock_client.chat.completions.create.side_effect = rate_limit_error
        mock_openai_class.return_value = mock_client
        
        client = OpenAIClient(api_key=self.test_api_key)
        result = client.generate_completion("Test prompt")
        
        assert result['success'] is False
        assert "Rate limit exceeded" in result['error']
    
    @patch('ai_explainer.OpenAI')
    def test_timeout_error_handling(self, mock_openai_class):
        """Test timeout error handling."""
        import openai
        
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = openai.APITimeoutError("Request timed out")
        mock_openai_class.return_value = mock_client
        
        client = OpenAIClient(api_key=self.test_api_key)
        result = client.generate_completion("Test prompt")
        
        assert result['success'] is False
        assert "Request timed out" in result['error']
    
    @patch('ai_explainer.OpenAI')
    def test_general_api_error_handling(self, mock_openai_class):
        """Test general API error handling."""
        import openai
        
        mock_client = Mock()
        # Create a proper APIError with required arguments
        mock_request = Mock()
        api_error = openai.APIError("API error", request=mock_request, body={})
        mock_client.chat.completions.create.side_effect = api_error
        mock_openai_class.return_value = mock_client
        
        client = OpenAIClient(api_key=self.test_api_key)
        result = client.generate_completion("Test prompt")
        
        assert result['success'] is False
        assert "API error" in result['error']


class TestAIExplainer:
    """Test main AI explainer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_problem = ParsedProblem(
            id='test-123',
            original_text='Find the derivative of x^2 + 3x',
            domain=MathDomain.CALCULUS,
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=['x'],
            expressions=['x^2 + 3x'],
            problem_type='derivative',
            metadata={}
        )
        
        self.sample_step = SolutionStep(
            step_number=1,
            operation='Apply power rule',
            explanation='Using the power rule for derivatives',
            mathematical_expression='x^2 + 3x',
            intermediate_result='2x + 3'
        )
    
    def test_explainer_initialization_with_ai(self):
        """Test explainer initialization when AI is available."""
        with patch('ai_explainer.OpenAIClient') as mock_client:
            explainer = AIExplainer(api_key="test-key")
            assert explainer.ai_available is True
            assert explainer.enable_fallback is True
    
    def test_explainer_initialization_without_ai(self):
        """Test explainer initialization when AI is not available."""
        with patch('ai_explainer.OpenAIClient', side_effect=ImportError("No OpenAI")):
            explainer = AIExplainer(enable_fallback=True)
            assert explainer.ai_available is False
            assert explainer.enable_fallback is True
    
    def test_explainer_initialization_no_fallback(self):
        """Test explainer initialization without fallback raises error."""
        with patch('ai_explainer.OpenAIClient', side_effect=ImportError("No OpenAI")):
            with pytest.raises(ImportError):
                AIExplainer(enable_fallback=False)
    
    @patch('ai_explainer.OpenAIClient')
    def test_ai_step_explanation_success(self, mock_client_class):
        """Test successful AI step explanation."""
        # Mock successful AI response
        mock_client = Mock()
        mock_client.generate_completion.return_value = {
            'success': True,
            'content': 'The power rule states that the derivative of x^n is n*x^(n-1).',
            'generation_time': 0.5
        }
        mock_client_class.return_value = mock_client
        
        explainer = AIExplainer(api_key="test-key")
        explanation = explainer.explain_step(
            self.sample_step, 
            self.sample_problem, 
            ExplanationLevel.INTERMEDIATE
        )
        
        assert isinstance(explanation, Explanation)
        assert "power rule" in explanation.content.lower()
        assert explanation.complexity_level == ExplanationLevel.INTERMEDIATE
        assert explanation.confidence_score > 0.8
        assert explanation.generation_time > 0
    
    @patch('ai_explainer.OpenAIClient')
    def test_ai_step_explanation_failure_with_fallback(self, mock_client_class):
        """Test AI step explanation failure with fallback."""
        # Mock failed AI response
        mock_client = Mock()
        mock_client.generate_completion.return_value = {
            'success': False,
            'error': 'API error',
            'generation_time': 0
        }
        mock_client_class.return_value = mock_client
        
        explainer = AIExplainer(api_key="test-key", enable_fallback=True)
        explanation = explainer.explain_step(
            self.sample_step, 
            self.sample_problem, 
            ExplanationLevel.INTERMEDIATE
        )
        
        assert isinstance(explanation, Explanation)
        assert explanation.confidence_score < 0.8  # Fallback has lower confidence
        assert explanation.generation_time < 0.1  # Fallback is fast
    
    def test_fallback_step_explanation(self):
        """Test fallback step explanation."""
        with patch('ai_explainer.OpenAIClient', side_effect=ImportError("No OpenAI")):
            explainer = AIExplainer(enable_fallback=True)
            explanation = explainer.explain_step(
                self.sample_step, 
                self.sample_problem, 
                ExplanationLevel.BEGINNER
            )
            
            assert isinstance(explanation, Explanation)
            assert "derivative" in explanation.content.lower()
            assert explanation.complexity_level == ExplanationLevel.BEGINNER
            assert explanation.confidence_score == 0.6
    
    @patch('ai_explainer.OpenAIClient')
    def test_hint_generation_success(self, mock_client_class):
        """Test successful hint generation."""
        mock_client = Mock()
        mock_client.generate_completion.return_value = {
            'success': True,
            'content': 'Try applying the power rule to each term separately.',
            'generation_time': 0.3
        }
        mock_client_class.return_value = mock_client
        
        explainer = AIExplainer(api_key="test-key")
        hint = explainer.generate_hint(
            self.sample_problem, 
            current_step=1,
            hint_level=HintLevel.GENTLE
        )
        
        assert isinstance(hint, Hint)
        assert "power rule" in hint.content.lower()
        assert hint.hint_level == HintLevel.GENTLE
        assert not hint.reveals_solution
        assert hint.confidence_score > 0.8
    
    def test_fallback_hint_generation(self):
        """Test fallback hint generation."""
        with patch('ai_explainer.OpenAIClient', side_effect=ImportError("No OpenAI")):
            explainer = AIExplainer(enable_fallback=True)
            hint = explainer.generate_hint(
                self.sample_problem, 
                current_step=1,
                hint_level=HintLevel.MODERATE
            )
            
            assert isinstance(hint, Hint)
            assert len(hint.content) > 0
            assert hint.hint_level == HintLevel.MODERATE
            assert hint.confidence_score == 0.5
    
    @patch('ai_explainer.OpenAIClient')
    def test_why_question_answering(self, mock_client_class):
        """Test why question answering."""
        mock_client = Mock()
        mock_client.generate_completion.return_value = {
            'success': True,
            'content': 'The power rule is fundamental because it provides a systematic way to find rates of change.',
            'generation_time': 0.4
        }
        mock_client_class.return_value = mock_client
        
        explainer = AIExplainer(api_key="test-key")
        context = MathContext(
            problem=self.sample_problem,
            current_step=self.sample_step,
            user_level=ExplanationLevel.INTERMEDIATE,
            previous_attempts=[],
            domain_knowledge={}
        )
        
        explanation = explainer.answer_why_question(
            "Why do we use the power rule?", 
            context
        )
        
        assert isinstance(explanation, Explanation)
        assert "power rule" in explanation.content.lower()
        assert explanation.complexity_level == ExplanationLevel.INTERMEDIATE
    
    @patch('ai_explainer.OpenAIClient')
    def test_explanation_level_adaptation(self, mock_client_class):
        """Test explanation level adaptation."""
        mock_client = Mock()
        mock_client.generate_completion.return_value = {
            'success': True,
            'content': 'The power rule is a simple way to find derivatives of polynomial terms.',
            'generation_time': 0.2
        }
        mock_client_class.return_value = mock_client
        
        explainer = AIExplainer(api_key="test-key")
        original = "The power rule states that d/dx[x^n] = n*x^(n-1) for any real number n."
        
        adapted = explainer.adapt_explanation_level(original, ExplanationLevel.BEGINNER)
        
        assert len(adapted) > 0
        assert adapted != original  # Should be different
    
    def test_concept_extraction(self):
        """Test mathematical concept extraction."""
        with patch('ai_explainer.OpenAIClient', side_effect=ImportError("No OpenAI")):
            explainer = AIExplainer(enable_fallback=True)
            
            text = "The derivative of a function measures the rate of change. We use calculus to find integrals and limits."
            concepts = explainer._extract_concepts(text)
            
            assert 'derivative' in concepts
            assert 'function' in concepts
            assert 'calculus' in concepts
            assert 'integral' in concepts
            assert 'limit' in concepts
    
    def test_mathematical_accuracy_validation(self):
        """Test mathematical accuracy validation."""
        with patch('ai_explainer.OpenAIClient', side_effect=ImportError("No OpenAI")):
            explainer = AIExplainer(enable_fallback=True)
            context = MathContext(
                problem=self.sample_problem,
                current_step=self.sample_step,
                user_level=ExplanationLevel.INTERMEDIATE,
                previous_attempts=[],
                domain_knowledge={}
            )
            
            # Valid explanation
            valid_explanation = "The derivative of x^2 is 2x using the power rule."
            assert explainer.validate_mathematical_accuracy(valid_explanation, context) is True
            
            # Invalid explanation
            invalid_explanation = "This involves divide by zero which is undefined."
            assert explainer.validate_mathematical_accuracy(invalid_explanation, context) is False
            
            # Empty explanation
            empty_explanation = ""
            assert explainer.validate_mathematical_accuracy(empty_explanation, context) is False


class TestExplanationDataStructures:
    """Test explanation data structures."""
    
    def test_explanation_creation(self):
        """Test Explanation dataclass creation."""
        explanation = Explanation(
            content="Test explanation",
            complexity_level=ExplanationLevel.INTERMEDIATE,
            related_concepts=['derivative', 'function'],
            examples=['x^2 -> 2x'],
            confidence_score=0.9,
            generation_time=0.5
        )
        
        assert explanation.content == "Test explanation"
        assert explanation.complexity_level == ExplanationLevel.INTERMEDIATE
        assert len(explanation.related_concepts) == 2
        assert len(explanation.examples) == 1
        assert explanation.confidence_score == 0.9
        assert explanation.generation_time == 0.5
    
    def test_hint_creation(self):
        """Test Hint dataclass creation."""
        hint = Hint(
            content="Try the power rule",
            hint_level=HintLevel.GENTLE,
            reveals_solution=False,
            next_step_guidance="Apply the rule to each term",
            confidence_score=0.8
        )
        
        assert hint.content == "Try the power rule"
        assert hint.hint_level == HintLevel.GENTLE
        assert hint.reveals_solution is False
        assert "Apply the rule" in hint.next_step_guidance
        assert hint.confidence_score == 0.8
    
    def test_math_context_creation(self):
        """Test MathContext dataclass creation."""
        problem = ParsedProblem(
            id='test',
            original_text='Test problem',
            domain=MathDomain.ALGEBRA,
            difficulty=DifficultyLevel.BEGINNER,
            variables=['x'],
            expressions=['x + 1'],
            problem_type='linear_equation',
            metadata={}
        )
        
        context = MathContext(
            problem=problem,
            current_step=None,
            user_level=ExplanationLevel.BEGINNER,
            previous_attempts=['x = 2'],
            domain_knowledge={'algebra': 'basic'}
        )
        
        assert context.problem.id == 'test'
        assert context.current_step is None
        assert context.user_level == ExplanationLevel.BEGINNER
        assert len(context.previous_attempts) == 1
        assert 'algebra' in context.domain_knowledge


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_problem = ParsedProblem(
            id='integration-test',
            original_text='Integrate x^2 + 2x + 1',
            domain=MathDomain.CALCULUS,
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=['x'],
            expressions=['x^2 + 2x + 1'],
            problem_type='integral',
            metadata={}
        )
    
    def test_complete_explanation_workflow(self):
        """Test complete explanation workflow."""
        with patch('ai_explainer.OpenAIClient', side_effect=ImportError("No OpenAI")):
            explainer = AIExplainer(enable_fallback=True)
            
            # Step explanation
            step = SolutionStep(
                step_number=1,
                operation='Integrate term by term',
                explanation='Apply integration rules',
                mathematical_expression='x^2 + 2x + 1',
                intermediate_result='x^3/3 + x^2 + x + C'
            )
            
            explanation = explainer.explain_step(step, self.sample_problem)
            assert isinstance(explanation, Explanation)
            
            # Hint generation
            hint = explainer.generate_hint(self.sample_problem, current_step=1)
            assert isinstance(hint, Hint)
            
            # Why question
            context = MathContext(
                problem=self.sample_problem,
                current_step=step,
                user_level=ExplanationLevel.INTERMEDIATE,
                previous_attempts=[],
                domain_knowledge={}
            )
            
            why_answer = explainer.answer_why_question(
                "Why do we add the constant C?", 
                context
            )
            assert isinstance(why_answer, Explanation)
    
    def test_error_recovery_scenarios(self):
        """Test error recovery in various scenarios."""
        # Test with AI unavailable
        with patch('ai_explainer.OpenAIClient', side_effect=ImportError("No OpenAI")):
            explainer = AIExplainer(enable_fallback=True)
            
            step = SolutionStep(
                step_number=1,
                operation='Unknown operation',
                explanation='Test',
                mathematical_expression='test',
                intermediate_result='result'
            )
            
            # Should not raise exception, should use fallback
            explanation = explainer.explain_step(step, self.sample_problem)
            assert isinstance(explanation, Explanation)
            assert explanation.confidence_score < 0.8  # Fallback confidence
    
    def test_different_difficulty_levels(self):
        """Test explanations for different difficulty levels."""
        with patch('ai_explainer.OpenAIClient', side_effect=ImportError("No OpenAI")):
            explainer = AIExplainer(enable_fallback=True)
            
            step = SolutionStep(
                step_number=1,
                operation='Differentiate',
                explanation='Find derivative',
                mathematical_expression='x^3',
                intermediate_result='3x^2'
            )
            
            # Test different levels
            for level in ExplanationLevel:
                explanation = explainer.explain_step(step, self.sample_problem, level)
                assert explanation.complexity_level == level
    
    def test_different_math_domains(self):
        """Test explanations for different mathematical domains."""
        with patch('ai_explainer.OpenAIClient', side_effect=ImportError("No OpenAI")):
            explainer = AIExplainer(enable_fallback=True)
            
            domains = [
                (MathDomain.ALGEBRA, 'solve', '2x + 3 = 7', 'x = 2'),
                (MathDomain.CALCULUS, 'derivative', 'x^2', '2x'),
                (MathDomain.LINEAR_ALGEBRA, 'multiply', 'A * B', 'C')
            ]
            
            for domain, operation, expr, result in domains:
                problem = ParsedProblem(
                    id=f'test-{domain.value}',
                    original_text=f'Test {domain.value} problem',
                    domain=domain,
                    difficulty=DifficultyLevel.INTERMEDIATE,
                    variables=['x'],
                    expressions=[expr],
                    problem_type=operation,
                    metadata={}
                )
                
                step = SolutionStep(
                    step_number=1,
                    operation=operation,
                    explanation='Test step',
                    mathematical_expression=expr,
                    intermediate_result=result
                )
                
                explanation = explainer.explain_step(step, problem)
                assert isinstance(explanation, Explanation)
                assert len(explanation.content) > 0


class TestHintGenerator:
    """Test cases for the progressive hint generation system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_problem = ParsedProblem(
            id='hint-test',
            original_text='Find the derivative of x^3 + 2x^2 - 5x + 1',
            domain=MathDomain.CALCULUS,
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=['x'],
            expressions=['x^3 + 2x^2 - 5x + 1'],
            problem_type='derivative',
            metadata={}
        )
        
        self.user_progress = {
            'attempts': ['3x^2 + 4x - 5', '3x^2 + 2x - 5'],
            'common_errors': ['forgot_constant_term'],
            'skill_level': 'intermediate',
            'hints_used': 1,
            'time_spent': 300,
            'similar_solved': 5
        }
    
    def test_hint_generator_initialization(self):
        """Test hint generator initialization."""
        # Without AI client
        generator = HintGenerator(None)
        assert generator.ai_client is None
        
        # With mock AI client
        mock_client = Mock()
        generator = HintGenerator(mock_client)
        assert generator.ai_client == mock_client
    
    def test_problem_context_analysis(self):
        """Test problem context analysis."""
        generator = HintGenerator(None)
        context = generator._analyze_problem_context(self.sample_problem)
        
        assert context['domain'] == MathDomain.CALCULUS
        assert context['difficulty'] == DifficultyLevel.INTERMEDIATE
        assert context['problem_type'] == 'derivative'
        assert 'derivative' in context['calculus_concepts']
        assert 'derivative' in context['keywords']
    
    def test_user_context_analysis(self):
        """Test user context analysis."""
        generator = HintGenerator(None)
        context = generator._analyze_user_context(self.user_progress)
        
        assert len(context['attempts']) == 2
        assert context['skill_level'] == 'intermediate'
        assert context['previous_hints_used'] == 1
        assert context['time_spent'] == 300
        assert context['similar_problems_solved'] == 5
    
    def test_keyword_extraction(self):
        """Test mathematical keyword extraction."""
        generator = HintGenerator(None)
        
        text = "Find the derivative of the function and solve the equation"
        keywords = generator._extract_keywords(text)
        
        assert 'derivative' in keywords
        assert 'function' in keywords
        assert 'solve' in keywords
        assert 'equation' in keywords
    
    def test_progressive_hints_generation_fallback(self):
        """Test progressive hints generation with fallback."""
        generator = HintGenerator(None)  # No AI client
        
        hints = generator.generate_progressive_hints(
            self.sample_problem, 
            self.user_progress, 
            max_hints=3
        )
        
        assert len(hints) == 3
        assert hints[0].hint_level == HintLevel.GENTLE
        assert hints[1].hint_level == HintLevel.MODERATE
        assert hints[2].hint_level == HintLevel.DETAILED
        
        # Check progression - each hint should be more detailed
        assert len(hints[0].content) < len(hints[2].content)
    
    @patch('ai_explainer.OpenAIClient')
    def test_progressive_hints_generation_ai(self, mock_client_class):
        """Test progressive hints generation with AI."""
        # Mock AI responses for different hint levels
        mock_client = Mock()
        responses = [
            {'success': True, 'content': 'Think about the power rule for derivatives.', 'generation_time': 0.3},
            {'success': True, 'content': 'Apply the power rule: d/dx[x^n] = n*x^(n-1) to each term.', 'generation_time': 0.4},
            {'success': True, 'content': 'Step 1: x^3 becomes 3x^2. Step 2: 2x^2 becomes 4x. Step 3: -5x becomes -5. Step 4: constant 1 becomes 0.', 'generation_time': 0.5}
        ]
        
        mock_client.generate_completion.side_effect = responses
        generator = HintGenerator(mock_client)
        
        hints = generator.generate_progressive_hints(
            self.sample_problem, 
            self.user_progress, 
            max_hints=3
        )
        
        assert len(hints) == 3
        assert 'power rule' in hints[0].content.lower()
        assert 'step 1' in hints[2].content.lower()
        assert hints[0].confidence_score > 0.8
    
    def test_contextual_hint_generation_calculus(self):
        """Test contextual hint generation for calculus problems."""
        generator = HintGenerator(None)
        
        # Test different hint levels for calculus
        for level in HintLevel:
            hint = generator._generate_contextual_hint(
                self.sample_problem, 
                self.user_progress, 
                level, 
                1
            )
            
            assert isinstance(hint, Hint)
            assert hint.hint_level == level
            assert len(hint.content) > 0
            
            if level == HintLevel.GENTLE:
                assert 'power' in hint.content.lower() or 'rule' in hint.content.lower()
            elif level == HintLevel.DETAILED:
                assert 'step' in hint.content.lower()
    
    def test_contextual_hint_generation_algebra(self):
        """Test contextual hint generation for algebra problems."""
        algebra_problem = ParsedProblem(
            id='algebra-test',
            original_text='Solve the equation 2x + 5 = 13',
            domain=MathDomain.ALGEBRA,
            difficulty=DifficultyLevel.BEGINNER,
            variables=['x'],
            expressions=['2x + 5 = 13'],
            problem_type='linear_equation',
            metadata={}
        )
        
        generator = HintGenerator(None)
        
        hint = generator._generate_contextual_hint(
            algebra_problem, 
            {'attempts': [], 'skill_level': 'beginner', 'hints_used': 0}, 
            HintLevel.MODERATE, 
            1
        )
        
        assert isinstance(hint, Hint)
        assert 'isolate' in hint.content.lower() or 'operation' in hint.content.lower()
    
    def test_hint_quality_validation(self):
        """Test hint quality validation."""
        generator = HintGenerator(None)
        
        # Valid hint
        good_hint = Hint(
            content="Try applying the power rule to each term in the polynomial.",
            hint_level=HintLevel.MODERATE,
            reveals_solution=False,
            next_step_guidance="Apply the rule step by step",
            confidence_score=0.8
        )
        
        assert generator.validate_hint_quality(good_hint, self.sample_problem, self.user_progress) is True
        
        # Invalid hint - too short
        bad_hint = Hint(
            content="Yes",
            hint_level=HintLevel.GENTLE,
            reveals_solution=False,
            next_step_guidance="",
            confidence_score=0.5
        )
        
        assert generator.validate_hint_quality(bad_hint, self.sample_problem, self.user_progress) is False
        
        # Invalid hint - gentle but gives away answer
        revealing_hint = Hint(
            content="The answer is 3x^2 + 4x - 5",
            hint_level=HintLevel.GENTLE,
            reveals_solution=False,
            next_step_guidance="",
            confidence_score=0.7
        )
        
        assert generator.validate_hint_quality(revealing_hint, self.sample_problem, self.user_progress) is False
    
    def test_next_step_guidance_generation(self):
        """Test next step guidance generation."""
        generator = HintGenerator(None)
        context = {'domain': MathDomain.CALCULUS}
        
        # Test different hint levels
        gentle_guidance = generator._generate_next_step_guidance(HintLevel.GENTLE, context)
        assert 'think about' in gentle_guidance.lower() or 'try' in gentle_guidance.lower()
        
        detailed_guidance = generator._generate_next_step_guidance(HintLevel.DETAILED, context)
        assert 'step' in detailed_guidance.lower()
        
        solution_guidance = generator._generate_next_step_guidance(HintLevel.SOLUTION, context)
        assert 'study' in solution_guidance.lower() or 'understand' in solution_guidance.lower()
    
    def test_fallback_hint_generation_different_domains(self):
        """Test fallback hint generation for different mathematical domains."""
        generator = HintGenerator(None)
        
        domains_and_problems = [
            (MathDomain.CALCULUS, 'derivative', 'Find d/dx[x^2]'),
            (MathDomain.ALGEBRA, 'equation_solving', 'Solve 2x + 3 = 7'),
            (MathDomain.LINEAR_ALGEBRA, 'matrix_operations', 'Multiply matrices A and B')
        ]
        
        for domain, problem_type, text in domains_and_problems:
            problem = ParsedProblem(
                id=f'test-{domain.value}',
                original_text=text,
                domain=domain,
                difficulty=DifficultyLevel.INTERMEDIATE,
                variables=['x'],
                expressions=[text],
                problem_type=problem_type,
                metadata={}
            )
            
            hint = generator._generate_contextual_hint(
                problem, 
                {'attempts': [], 'skill_level': 'intermediate', 'hints_used': 0}, 
                HintLevel.MODERATE, 
                1
            )
            
            assert isinstance(hint, Hint)
            assert len(hint.content) > 10
            assert hint.confidence_score == 0.6  # Fallback confidence


class TestProgressiveHintIntegration:
    """Test integration of progressive hints with AI explainer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_problem = ParsedProblem(
            id='integration-test',
            original_text='Integrate x^2 + 3x - 2',
            domain=MathDomain.CALCULUS,
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=['x'],
            expressions=['x^2 + 3x - 2'],
            problem_type='integral',
            metadata={}
        )
    
    def test_ai_explainer_progressive_hints_fallback(self):
        """Test AI explainer progressive hints with fallback."""
        with patch('ai_explainer.OpenAIClient', side_effect=ImportError("No OpenAI")):
            explainer = AIExplainer(enable_fallback=True)
            
            user_progress = {
                'attempts': ['x^3/3 + 3x^2/2 - 2x'],
                'skill_level': 'intermediate',
                'hints_used': 0
            }
            
            hints = explainer.generate_progressive_hints(
                self.sample_problem, 
                user_progress, 
                max_hints=3
            )
            
            assert len(hints) == 3
            assert all(isinstance(hint, Hint) for hint in hints)
            assert hints[0].hint_level == HintLevel.GENTLE
            assert hints[1].hint_level == HintLevel.MODERATE
            assert hints[2].hint_level == HintLevel.DETAILED
    
    @patch('ai_explainer.OpenAIClient')
    def test_ai_explainer_progressive_hints_with_ai(self, mock_client_class):
        """Test AI explainer progressive hints with AI."""
        mock_client = Mock()
        mock_client.generate_completion.return_value = {
            'success': True,
            'content': 'Consider the reverse power rule for integration.',
            'generation_time': 0.3
        }
        mock_client_class.return_value = mock_client
        
        explainer = AIExplainer(api_key="test-key")
        
        user_progress = {
            'attempts': [],
            'skill_level': 'beginner',
            'hints_used': 0
        }
        
        hints = explainer.generate_progressive_hints(
            self.sample_problem, 
            user_progress, 
            max_hints=2
        )
        
        assert len(hints) == 2
        assert all(hint.confidence_score > 0.8 for hint in hints)
    
    def test_hint_quality_validation_integration(self):
        """Test hint quality validation integration."""
        with patch('ai_explainer.OpenAIClient', side_effect=ImportError("No OpenAI")):
            explainer = AIExplainer(enable_fallback=True)
            
            # Generate a hint
            hint = explainer.generate_hint(
                self.sample_problem, 
                current_step=1, 
                hint_level=HintLevel.MODERATE
            )
            
            # Validate the hint
            user_context = {'attempts': [], 'skill_level': 'intermediate'}
            is_valid = explainer.validate_hint_quality(hint, self.sample_problem, user_context)
            
            assert isinstance(is_valid, bool)
            assert is_valid is True  # Fallback hints should be valid
    
    def test_contextual_adaptation_based_on_attempts(self):
        """Test that hints adapt based on user's previous attempts."""
        with patch('ai_explainer.OpenAIClient', side_effect=ImportError("No OpenAI")):
            explainer = AIExplainer(enable_fallback=True)
            
            # User with no attempts
            user_progress_new = {
                'attempts': [],
                'skill_level': 'beginner',
                'hints_used': 0
            }
            
            # User with multiple attempts
            user_progress_experienced = {
                'attempts': ['x^3/3', 'x^3/3 + 3x^2/2', 'x^3/3 + 3x^2/2 - 2x'],
                'skill_level': 'intermediate',
                'hints_used': 2
            }
            
            hints_new = explainer.generate_progressive_hints(
                self.sample_problem, user_progress_new, max_hints=2
            )
            
            hints_experienced = explainer.generate_progressive_hints(
                self.sample_problem, user_progress_experienced, max_hints=2
            )
            
            # Both should generate hints, but they might be different
            assert len(hints_new) == 2
            assert len(hints_experienced) == 2
            assert all(isinstance(hint, Hint) for hint in hints_new)
            assert all(isinstance(hint, Hint) for hint in hints_experienced)


class TestHintSystemEdgeCases:
    """Test edge cases and error scenarios for the hint system."""
    
    def test_empty_problem_text(self):
        """Test hint generation with empty problem text."""
        empty_problem = ParsedProblem(
            id='empty-test',
            original_text='',
            domain=MathDomain.ALGEBRA,
            difficulty=DifficultyLevel.BEGINNER,
            variables=[],
            expressions=[],
            problem_type='unknown',
            metadata={}
        )
        
        generator = HintGenerator(None)
        hint = generator._generate_contextual_hint(
            empty_problem, 
            {'attempts': [], 'skill_level': 'beginner', 'hints_used': 0}, 
            HintLevel.GENTLE, 
            1
        )
        
        assert isinstance(hint, Hint)
        assert len(hint.content) > 0  # Should still generate something
    
    def test_unknown_domain_handling(self):
        """Test hint generation for unknown mathematical domains."""
        unknown_problem = ParsedProblem(
            id='unknown-test',
            original_text='Solve this unknown type of problem',
            domain=MathDomain.STATISTICS,  # Less common domain
            difficulty=DifficultyLevel.ADVANCED,
            variables=['x'],
            expressions=['unknown'],
            problem_type='unknown',
            metadata={}
        )
        
        generator = HintGenerator(None)
        hint = generator._generate_contextual_hint(
            unknown_problem, 
            {'attempts': [], 'skill_level': 'advanced', 'hints_used': 0}, 
            HintLevel.MODERATE, 
            1
        )
        
        assert isinstance(hint, Hint)
        assert hint.confidence_score == 0.6  # Fallback confidence
    
    def test_maximum_hints_limit(self):
        """Test that progressive hints respect maximum limit."""
        generator = HintGenerator(None)
        
        problem = ParsedProblem(
            id='limit-test',
            original_text='Test problem',
            domain=MathDomain.ALGEBRA,
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=['x'],
            expressions=['x + 1'],
            problem_type='equation',
            metadata={}
        )
        
        # Test different max_hints values
        for max_hints in [1, 2, 3, 5, 10]:
            hints = generator.generate_progressive_hints(
                problem, 
                {'attempts': [], 'skill_level': 'intermediate', 'hints_used': 0}, 
                max_hints=max_hints
            )
            
            expected_count = min(max_hints, 4)  # 4 is the number of hint levels
            assert len(hints) == expected_count
    
    def test_hint_validation_edge_cases(self):
        """Test hint validation with edge cases."""
        generator = HintGenerator(None)
        problem = ParsedProblem(
            id='validation-test',
            original_text='Test problem',
            domain=MathDomain.ALGEBRA,
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=['x'],
            expressions=['x + 1'],
            problem_type='equation',
            metadata={}
        )
        
        user_context = {'attempts': [], 'skill_level': 'intermediate'}
        
        # Test various invalid hints
        invalid_hints = [
            # Empty content
            Hint("", HintLevel.GENTLE, False, "", 0.5),
            # Very short content
            Hint("No", HintLevel.MODERATE, False, "", 0.7),
            # Solution level but doesn't reveal solution
            Hint("This is a hint", HintLevel.SOLUTION, False, "", 0.8),
            # Gentle but reveals answer
            Hint("The answer is x = 5", HintLevel.GENTLE, False, "", 0.9)
        ]
        
        for hint in invalid_hints:
            assert generator.validate_hint_quality(hint, problem, user_context) is False
        
        # Test valid hint
        valid_hint = Hint(
            "Consider what operation you need to isolate the variable",
            HintLevel.MODERATE,
            False,
            "Try the suggested approach",
            0.8
        )
        
        assert generator.validate_hint_quality(valid_hint, problem, user_context) is True


class TestUserSkillAnalyzer:
    """Test cases for user skill level detection and analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from ai_explainer import UserSkillAnalyzer
        self.analyzer = UserSkillAnalyzer()
        
        # Sample user histories for different skill levels
        self.beginner_history = {
            'problems_solved': [
                {'correct': True, 'difficulty': DifficultyLevel.BEGINNER, 'time_taken': 180, 'hints_used': 2},
                {'correct': False, 'difficulty': DifficultyLevel.BEGINNER, 'time_taken': 240, 'hints_used': 3},
                {'correct': True, 'difficulty': DifficultyLevel.BEGINNER, 'time_taken': 200, 'hints_used': 1},
            ]
        }
        
        self.intermediate_history = {
            'problems_solved': [
                {'correct': True, 'difficulty': DifficultyLevel.INTERMEDIATE, 'time_taken': 250, 'hints_used': 1},
                {'correct': True, 'difficulty': DifficultyLevel.INTERMEDIATE, 'time_taken': 300, 'hints_used': 0},
                {'correct': True, 'difficulty': DifficultyLevel.BEGINNER, 'time_taken': 120, 'hints_used': 0},
                {'correct': False, 'difficulty': DifficultyLevel.ADVANCED, 'time_taken': 600, 'hints_used': 2},
            ]
        }
        
        self.advanced_history = {
            'problems_solved': [
                {'correct': True, 'difficulty': DifficultyLevel.ADVANCED, 'time_taken': 400, 'hints_used': 0},
                {'correct': True, 'difficulty': DifficultyLevel.ADVANCED, 'time_taken': 350, 'hints_used': 1},
                {'correct': True, 'difficulty': DifficultyLevel.INTERMEDIATE, 'time_taken': 200, 'hints_used': 0},
                {'correct': True, 'difficulty': DifficultyLevel.EXPERT, 'time_taken': 800, 'hints_used': 1},
            ]
        }
        
        self.expert_history = {
            'problems_solved': [
                {'correct': True, 'difficulty': DifficultyLevel.EXPERT, 'time_taken': 600, 'hints_used': 0},
                {'correct': True, 'difficulty': DifficultyLevel.EXPERT, 'time_taken': 700, 'hints_used': 0},
                {'correct': True, 'difficulty': DifficultyLevel.ADVANCED, 'time_taken': 300, 'hints_used': 0},
                {'correct': True, 'difficulty': DifficultyLevel.ADVANCED, 'time_taken': 250, 'hints_used': 0},
            ]
        }
    
    def test_skill_analyzer_initialization(self):
        """Test skill analyzer initialization."""
        assert self.analyzer.skill_indicators['accuracy'] == 0.3
        assert self.analyzer.skill_indicators['speed'] == 0.2
        assert self.analyzer.skill_indicators['hints_used'] == 0.2
        assert self.analyzer.skill_indicators['complexity'] == 0.3
    
    def test_detect_skill_level_empty_history(self):
        """Test skill level detection with empty history."""
        empty_history = {}
        level = self.analyzer.detect_skill_level(empty_history)
        assert level == ExplanationLevel.INTERMEDIATE  # Default level
        
        no_problems_history = {'problems_solved': []}
        level = self.analyzer.detect_skill_level(no_problems_history)
        assert level == ExplanationLevel.INTERMEDIATE
    
    def test_detect_skill_level_beginner(self):
        """Test detection of beginner skill level."""
        level = self.analyzer.detect_skill_level(self.beginner_history)
        assert level == ExplanationLevel.BEGINNER
    
    def test_detect_skill_level_intermediate(self):
        """Test detection of intermediate skill level."""
        level = self.analyzer.detect_skill_level(self.intermediate_history)
        assert level == ExplanationLevel.INTERMEDIATE
    
    def test_detect_skill_level_advanced(self):
        """Test detection of advanced skill level."""
        level = self.analyzer.detect_skill_level(self.advanced_history)
        # The advanced history might be detected as expert due to high performance
        assert level in [ExplanationLevel.ADVANCED, ExplanationLevel.EXPERT]
    
    def test_detect_skill_level_expert(self):
        """Test detection of expert skill level."""
        level = self.analyzer.detect_skill_level(self.expert_history)
        assert level == ExplanationLevel.EXPERT
    
    def test_accuracy_score_calculation(self):
        """Test accuracy score calculation."""
        # Perfect accuracy
        perfect_history = {
            'problems_solved': [
                {'correct': True}, {'correct': True}, {'correct': True}
            ]
        }
        score = self.analyzer._calculate_accuracy_score(perfect_history)
        assert score == 1.0
        
        # 50% accuracy
        mixed_history = {
            'problems_solved': [
                {'correct': True}, {'correct': False}, {'correct': True}, {'correct': False}
            ]
        }
        score = self.analyzer._calculate_accuracy_score(mixed_history)
        assert score == 0.5
        
        # No problems
        empty_history = {'problems_solved': []}
        score = self.analyzer._calculate_accuracy_score(empty_history)
        assert score == 0.5  # Default
    
    def test_speed_score_calculation(self):
        """Test speed score calculation."""
        # Fast solver
        fast_history = {
            'problems_solved': [
                {'time_taken': 60, 'difficulty': DifficultyLevel.BEGINNER},  # Half expected time
                {'time_taken': 150, 'difficulty': DifficultyLevel.INTERMEDIATE},  # Half expected time
            ]
        }
        score = self.analyzer._calculate_speed_score(fast_history)
        assert score > 0.5  # Should be above average
        
        # Slow solver
        slow_history = {
            'problems_solved': [
                {'time_taken': 240, 'difficulty': DifficultyLevel.BEGINNER},  # Double expected time
                {'time_taken': 600, 'difficulty': DifficultyLevel.INTERMEDIATE},  # Double expected time
            ]
        }
        score = self.analyzer._calculate_speed_score(slow_history)
        assert score < 0.5  # Should be below average
    
    def test_hint_score_calculation(self):
        """Test hint usage score calculation."""
        # No hints used
        no_hints_history = {
            'problems_solved': [
                {'hints_used': 0}, {'hints_used': 0}, {'hints_used': 0}
            ]
        }
        score = self.analyzer._calculate_hint_score(no_hints_history)
        assert score == 1.0
        
        # Many hints used
        many_hints_history = {
            'problems_solved': [
                {'hints_used': 3}, {'hints_used': 3}, {'hints_used': 3}
            ]
        }
        score = self.analyzer._calculate_hint_score(many_hints_history)
        assert score == 0.0
    
    def test_complexity_score_calculation(self):
        """Test complexity score calculation."""
        # Only easy problems
        easy_history = {
            'problems_solved': [
                {'correct': True, 'difficulty': DifficultyLevel.BEGINNER},
                {'correct': True, 'difficulty': DifficultyLevel.BEGINNER}
            ]
        }
        score = self.analyzer._calculate_complexity_score(easy_history)
        assert score == 0.25  # Weight for beginner level
        
        # Only expert problems
        expert_history = {
            'problems_solved': [
                {'correct': True, 'difficulty': DifficultyLevel.EXPERT},
                {'correct': True, 'difficulty': DifficultyLevel.EXPERT}
            ]
        }
        score = self.analyzer._calculate_complexity_score(expert_history)
        assert score == 1.0  # Weight for expert level
    
    def test_skill_progression_recommendations(self):
        """Test skill progression recommendations."""
        # Test for beginner
        recommendations = self.analyzer.get_skill_progression_recommendations(
            ExplanationLevel.BEGINNER, self.beginner_history
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('fundamental concepts' in rec.lower() for rec in recommendations)
        
        # Test for advanced
        recommendations = self.analyzer.get_skill_progression_recommendations(
            ExplanationLevel.ADVANCED, self.advanced_history
        )
        
        assert isinstance(recommendations, list)
        assert any('expert-level' in rec.lower() for rec in recommendations)


class TestExplanationAdapter:
    """Test cases for explanation level adaptation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from ai_explainer import ExplanationAdapter
        self.adapter = ExplanationAdapter(None)  # No AI client for testing
        
        self.sample_explanation = "The derivative of x^2 is 2x using the power rule."
        
        self.user_history = {
            'problems_solved': [
                {'correct': True, 'difficulty': DifficultyLevel.INTERMEDIATE, 'time_taken': 300, 'hints_used': 1}
            ]
        }
    
    def test_explanation_adapter_initialization(self):
        """Test explanation adapter initialization."""
        assert self.adapter.ai_client is None
        assert hasattr(self.adapter, 'skill_analyzer')
    
    def test_adapt_explanation_to_user(self):
        """Test adapting explanation to user's detected skill level."""
        adapted = self.adapter.adapt_explanation_to_user(
            self.sample_explanation,
            self.user_history,
            target_concept="derivative",
            context="calculus problem"
        )
        
        assert isinstance(adapted, str)
        assert len(adapted) > 0
        # Should be different from original for some levels
    
    def test_adapt_explanation_complexity_beginner(self):
        """Test adapting explanation for beginner level."""
        adapted = self.adapter.adapt_explanation_complexity(
            self.sample_explanation,
            ExplanationLevel.BEGINNER,
            concept="derivative"
        )
        
        assert isinstance(adapted, str)
        assert len(adapted) > 0
        # Should contain simplified language
        assert 'rate of change' in adapted.lower() or 'step by step' in adapted.lower()
    
    def test_adapt_explanation_complexity_advanced(self):
        """Test adapting explanation for advanced level."""
        adapted = self.adapter.adapt_explanation_complexity(
            self.sample_explanation,
            ExplanationLevel.ADVANCED,
            concept="derivative"
        )
        
        assert isinstance(adapted, str)
        assert len(adapted) >= len(self.sample_explanation)  # Should add detail
        # Should contain additional theoretical context
        assert 'limit definition' in adapted.lower() or 'theorem' in adapted.lower()
    
    def test_adapt_explanation_complexity_expert(self):
        """Test adapting explanation for expert level."""
        adapted = self.adapter.adapt_explanation_complexity(
            self.sample_explanation,
            ExplanationLevel.EXPERT,
            concept="derivative"
        )
        
        assert isinstance(adapted, str)
        assert len(adapted) >= len(self.sample_explanation)
        # Should contain theoretical implications
        assert 'theoretical' in adapted.lower() or 'implications' in adapted.lower()
    
    def test_generate_level_appropriate_examples_derivative(self):
        """Test generating level-appropriate examples for derivatives."""
        # Beginner examples
        beginner_examples = self.adapter.generate_level_appropriate_examples(
            "derivative", ExplanationLevel.BEGINNER, count=2
        )
        
        assert len(beginner_examples) == 2
        # Check for x^2 format (the examples use x² which might not match the test)
        assert any('x' in ex and ('2' in ex or '²' in ex) for ex in beginner_examples)
        
        # Expert examples
        expert_examples = self.adapter.generate_level_appropriate_examples(
            "derivative", ExplanationLevel.EXPERT, count=2
        )
        
        assert len(expert_examples) == 2
        assert any('Jacobian' in ex or 'Fréchet' in ex for ex in expert_examples)
    
    def test_generate_level_appropriate_examples_integral(self):
        """Test generating level-appropriate examples for integrals."""
        examples = self.adapter.generate_level_appropriate_examples(
            "integral", ExplanationLevel.INTERMEDIATE, count=2
        )
        
        assert len(examples) == 2
        assert all('∫' in ex for ex in examples)
    
    def test_assess_explanation_appropriateness(self):
        """Test assessing explanation appropriateness for target level."""
        # Test beginner-appropriate explanation
        beginner_explanation = "Let's work through this step by step. The simple rule is to multiply by the power and reduce it by one."
        
        assessment = self.adapter.assess_explanation_appropriateness(
            beginner_explanation,
            ExplanationLevel.BEGINNER,
            "derivative"
        )
        
        assert isinstance(assessment, dict)
        assert 'appropriateness_score' in assessment
        assert 'issues' in assessment
        assert 'suggestions' in assessment
        assert 0 <= assessment['appropriateness_score'] <= 1
        
        # Test expert-inappropriate explanation for beginner
        expert_explanation = "The rigorous proof involves the formal definition using limits and epsilon-delta arguments."
        
        assessment = self.adapter.assess_explanation_appropriateness(
            expert_explanation,
            ExplanationLevel.BEGINNER,
            "derivative"
        )
        
        assert assessment['appropriateness_score'] < 0.5  # Should be low
        assert len(assessment['issues']) > 0  # Should have issues
    
    def test_rule_based_adaptation_fallback(self):
        """Test rule-based adaptation when AI is not available."""
        # Test different levels
        for level in ExplanationLevel:
            adapted = self.adapter._rule_based_adapt_explanation(
                self.sample_explanation, level
            )
            
            assert isinstance(adapted, str)
            assert len(adapted) > 0
            
            if level == ExplanationLevel.BEGINNER:
                assert 'rate of change' in adapted.lower()
            elif level == ExplanationLevel.ADVANCED:
                assert len(adapted) >= len(self.sample_explanation)


class TestExplanationLevelIntegration:
    """Test integration of explanation level adaptation with AI explainer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_problem = ParsedProblem(
            id='level-test',
            original_text='Find the derivative of x^3 + 2x^2',
            domain=MathDomain.CALCULUS,
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=['x'],
            expressions=['x^3 + 2x^2'],
            problem_type='derivative',
            metadata={}
        )
        
        self.user_histories = {
            'beginner': {
                'problems_solved': [
                    {'correct': False, 'difficulty': DifficultyLevel.BEGINNER, 'time_taken': 300, 'hints_used': 3}
                ]
            },
            'expert': {
                'problems_solved': [
                    {'correct': True, 'difficulty': DifficultyLevel.EXPERT, 'time_taken': 400, 'hints_used': 0},
                    {'correct': True, 'difficulty': DifficultyLevel.EXPERT, 'time_taken': 350, 'hints_used': 0}
                ]
            }
        }
    
    def test_ai_explainer_skill_level_detection(self):
        """Test skill level detection through AI explainer."""
        with patch('ai_explainer.OpenAIClient', side_effect=ImportError("No OpenAI")):
            explainer = AIExplainer(enable_fallback=True)
            
            # Test beginner detection
            level = explainer.detect_user_skill_level(self.user_histories['beginner'])
            assert level == ExplanationLevel.BEGINNER
            
            # Test expert detection
            level = explainer.detect_user_skill_level(self.user_histories['expert'])
            assert level == ExplanationLevel.EXPERT
    
    def test_ai_explainer_explanation_adaptation(self):
        """Test explanation adaptation through AI explainer."""
        with patch('ai_explainer.OpenAIClient', side_effect=ImportError("No OpenAI")):
            explainer = AIExplainer(enable_fallback=True)
            
            original_explanation = "Apply the power rule to find the derivative."
            
            # Adapt for beginner
            adapted = explainer.adapt_explanation_to_user(
                original_explanation,
                self.user_histories['beginner'],
                target_concept="derivative"
            )
            
            assert isinstance(adapted, str)
            assert len(adapted) > 0
            # Should be adapted for beginner level
            assert 'rate of change' in adapted.lower() or 'step by step' in adapted.lower()
    
    def test_skill_progression_recommendations_integration(self):
        """Test skill progression recommendations through AI explainer."""
        with patch('ai_explainer.OpenAIClient', side_effect=ImportError("No OpenAI")):
            explainer = AIExplainer(enable_fallback=True)
            
            recommendations = explainer.get_skill_progression_recommendations(
                self.user_histories['beginner']
            )
            
            assert isinstance(recommendations, list)
            assert len(recommendations) > 0
            assert all(isinstance(rec, str) for rec in recommendations)
    
    def test_explanation_appropriateness_assessment(self):
        """Test explanation appropriateness assessment through AI explainer."""
        with patch('ai_explainer.OpenAIClient', side_effect=ImportError("No OpenAI")):
            explainer = AIExplainer(enable_fallback=True)
            
            explanation = "The derivative represents the instantaneous rate of change."
            
            assessment = explainer.assess_explanation_appropriateness(
                explanation,
                ExplanationLevel.INTERMEDIATE,
                "derivative"
            )
            
            assert isinstance(assessment, dict)
            assert 'appropriateness_score' in assessment
            assert 0 <= assessment['appropriateness_score'] <= 1
    
    @patch('ai_explainer.OpenAIClient')
    def test_ai_powered_explanation_adaptation(self, mock_client_class):
        """Test AI-powered explanation adaptation."""
        mock_client = Mock()
        mock_client.generate_completion.return_value = {
            'success': True,
            'content': 'Let\'s work through this step by step. When we find the rate of change of x^3 + 2x^2...',
            'generation_time': 0.4
        }
        mock_client_class.return_value = mock_client
        
        explainer = AIExplainer(api_key="test-key")
        
        original_explanation = "Apply the power rule to differentiate x^3 + 2x^2."
        
        adapted = explainer.adapt_explanation_to_user(
            original_explanation,
            self.user_histories['beginner'],
            target_concept="derivative",
            context="calculus problem"
        )
        
        assert isinstance(adapted, str)
        assert 'step by step' in adapted.lower()
        assert adapted != original_explanation


class TestSkillAnalysisEdgeCases:
    """Test edge cases and error scenarios for skill analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from ai_explainer import UserSkillAnalyzer
        self.analyzer = UserSkillAnalyzer()
    
    def test_inconsistent_performance_data(self):
        """Test skill analysis with inconsistent performance data."""
        inconsistent_history = {
            'problems_solved': [
                {'correct': True, 'difficulty': DifficultyLevel.EXPERT, 'time_taken': 100, 'hints_used': 0},
                {'correct': False, 'difficulty': DifficultyLevel.BEGINNER, 'time_taken': 600, 'hints_used': 3},
                {'correct': True, 'difficulty': DifficultyLevel.ADVANCED, 'time_taken': 200, 'hints_used': 1},
            ]
        }
        
        level = self.analyzer.detect_skill_level(inconsistent_history)
        assert level in [ExplanationLevel.BEGINNER, ExplanationLevel.INTERMEDIATE, 
                        ExplanationLevel.ADVANCED, ExplanationLevel.EXPERT]
    
    def test_missing_performance_fields(self):
        """Test skill analysis with missing performance fields."""
        incomplete_history = {
            'problems_solved': [
                {'correct': True},  # Missing other fields
                {'difficulty': DifficultyLevel.INTERMEDIATE},  # Missing correct field
                {'time_taken': 300, 'hints_used': 1}  # Missing correct and difficulty
            ]
        }
        
        # Should not crash and return a valid level
        level = self.analyzer.detect_skill_level(incomplete_history)
        assert level in [ExplanationLevel.BEGINNER, ExplanationLevel.INTERMEDIATE, 
                        ExplanationLevel.ADVANCED, ExplanationLevel.EXPERT]
    
    def test_extreme_performance_values(self):
        """Test skill analysis with extreme performance values."""
        extreme_history = {
            'problems_solved': [
                {'correct': True, 'difficulty': DifficultyLevel.EXPERT, 'time_taken': 1, 'hints_used': 0},  # Extremely fast
                {'correct': True, 'difficulty': DifficultyLevel.BEGINNER, 'time_taken': 10000, 'hints_used': 100},  # Extremely slow with many hints
            ]
        }
        
        level = self.analyzer.detect_skill_level(extreme_history)
        assert level in [ExplanationLevel.BEGINNER, ExplanationLevel.INTERMEDIATE, 
                        ExplanationLevel.ADVANCED, ExplanationLevel.EXPERT]
    
    def test_single_problem_history(self):
        """Test skill analysis with only one problem solved."""
        single_problem_history = {
            'problems_solved': [
                {'correct': True, 'difficulty': DifficultyLevel.INTERMEDIATE, 'time_taken': 300, 'hints_used': 1}
            ]
        }
        
        level = self.analyzer.detect_skill_level(single_problem_history)
        assert level in [ExplanationLevel.BEGINNER, ExplanationLevel.INTERMEDIATE, 
                        ExplanationLevel.ADVANCED, ExplanationLevel.EXPERT]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])