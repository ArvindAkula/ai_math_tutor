"""
AI-powered explanation service for mathematical concepts.
Integrates with OpenAI API to generate natural language explanations, hints, and contextual help.
Enhanced with comprehensive error handling and fallback systems.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import sys

# Add shared models to path
from models import (
    ParsedProblem, StepSolution, SolutionStep, MathDomain, DifficultyLevel
)

# Import error handling
from error_handling import (
    error_handler, async_error_handler, ErrorCategory, ErrorSeverity,
    AIServiceError, CircuitBreaker, retry_manager, graceful_degradation
)

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not available. Install with: pip install openai")


class ExplanationLevel(Enum):
    """Difficulty levels for explanations."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class HintLevel(Enum):
    """Progressive hint levels."""
    GENTLE = "gentle"
    MODERATE = "moderate"
    DETAILED = "detailed"
    SOLUTION = "solution"


@dataclass
class Explanation:
    """Structured explanation response."""
    content: str
    complexity_level: ExplanationLevel
    related_concepts: List[str]
    examples: List[str]
    confidence_score: float
    generation_time: float


@dataclass
class Hint:
    """Structured hint response."""
    content: str
    hint_level: HintLevel
    reveals_solution: bool
    next_step_guidance: str
    confidence_score: float


@dataclass
class MathContext:
    """Mathematical context for explanations."""
    problem: ParsedProblem
    current_step: Optional[SolutionStep]
    user_level: ExplanationLevel
    previous_attempts: List[str]
    domain_knowledge: Dict[str, Any]


class PromptTemplates:
    """Template manager for OpenAI prompts."""
    
    STEP_EXPLANATION = """
You are an expert mathematics tutor. Explain the following mathematical step in a clear, educational way.

Problem Context: {problem_context}
Mathematical Step: {step_operation}
Expression: {mathematical_expression}
Result: {intermediate_result}
User Level: {user_level}

Provide a clear explanation that:
1. Explains WHY this step is necessary
2. Describes HOW the operation is performed
3. Connects to broader mathematical concepts
4. Uses appropriate language for {user_level} level

Keep the explanation concise but thorough. Use mathematical notation when helpful.
"""

    CONCEPT_EXPLANATION = """
You are an expert mathematics educator. Explain the mathematical concept clearly and pedagogically.

Concept: {concept}
Context: {context}
User Level: {user_level}
Domain: {domain}

Provide an explanation that includes:
1. Clear definition of the concept
2. Why it's important in mathematics
3. How it relates to other concepts
4. A simple example
5. Common misconceptions to avoid

Adapt the language and depth to {user_level} level.
"""

    HINT_GENERATION = """
You are a helpful mathematics tutor providing hints to guide student learning.

Problem: {problem_text}
Current Step: {current_step}
Student's Previous Attempts: {previous_attempts}
Hint Level: {hint_level}

Provide a {hint_level} hint that:
- Guides the student toward the solution without giving it away
- Builds on their current understanding
- Encourages mathematical thinking
- Is appropriate for the hint level requested

Do not solve the problem completely. Focus on the next logical step or insight needed.
"""

    WHY_QUESTION = """
You are an expert mathematics tutor answering a "why" question about mathematical concepts.

Question: {question}
Mathematical Context: {context}
User Level: {user_level}

Provide a clear, insightful answer that:
1. Addresses the fundamental "why" behind the concept
2. Explains the mathematical reasoning
3. Connects to intuitive understanding
4. Uses appropriate examples
5. Matches the user's level of understanding

Focus on building deep conceptual understanding rather than just procedural knowledge.
"""

    ERROR_ANALYSIS = """
You are a mathematics tutor helping a student understand their mistake.

Problem: {problem}
Student's Answer: {student_answer}
Correct Answer: {correct_answer}
Error Type: {error_type}

Provide helpful feedback that:
1. Identifies where the error occurred
2. Explains why it's incorrect
3. Shows the correct approach
4. Helps prevent similar mistakes
5. Encourages the student

Be supportive and educational, not critical.
"""


class OpenAIClient:
    """OpenAI API client with error handling and rate limiting."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _rate_limit(self):
        """Implement basic rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def generate_completion(self, 
                          prompt: str, 
                          max_tokens: int = 500,
                          temperature: float = 0.7,
                          timeout: int = 30) -> Dict[str, Any]:
        """
        Generate completion from OpenAI API with error handling.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens in response
            temperature: Creativity parameter (0-1)
            timeout: Request timeout in seconds
            
        Returns:
            Dict containing response and metadata
        """
        self._rate_limit()
        
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert mathematics tutor and educator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout
            )
            
            generation_time = time.time() - start_time
            
            return {
                'content': response.choices[0].message.content.strip(),
                'usage': response.usage.dict() if response.usage else {},
                'model': response.model,
                'generation_time': generation_time,
                'success': True
            }
            
        except openai.RateLimitError as e:
            self.logger.warning(f"Rate limit exceeded: {e}")
            return {
                'content': '',
                'error': 'Rate limit exceeded. Please try again later.',
                'success': False,
                'generation_time': 0
            }
            
        except openai.APITimeoutError as e:
            self.logger.error(f"API timeout: {e}")
            return {
                'content': '',
                'error': 'Request timed out. Please try again.',
                'success': False,
                'generation_time': 0
            }
            
        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {e}")
            return {
                'content': '',
                'error': f'API error: {str(e)}',
                'success': False,
                'generation_time': 0
            }
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return {
                'content': '',
                'error': f'Unexpected error: {str(e)}',
                'success': False,
                'generation_time': 0
            }


class HintGenerator:
    """Progressive hint generation system."""
    
    def __init__(self, ai_client: Optional[OpenAIClient] = None):
        """Initialize hint generator."""
        self.ai_client = ai_client
        self.templates = PromptTemplates()
        
    def generate_progressive_hints(self, 
                                 problem: ParsedProblem,
                                 user_progress: Dict[str, Any],
                                 max_hints: int = 4) -> List[Hint]:
        """
        Generate a sequence of progressive hints.
        
        Args:
            problem: Mathematical problem
            user_progress: User's current progress and attempts
            max_hints: Maximum number of hints to generate
            
        Returns:
            List of progressive hints
        """
        hints = []
        hint_levels = [HintLevel.GENTLE, HintLevel.MODERATE, HintLevel.DETAILED, HintLevel.SOLUTION]
        
        for i, level in enumerate(hint_levels[:max_hints]):
            hint = self._generate_contextual_hint(problem, user_progress, level, i + 1)
            hints.append(hint)
            
        return hints
    
    def _generate_contextual_hint(self, 
                                problem: ParsedProblem,
                                user_progress: Dict[str, Any],
                                hint_level: HintLevel,
                                hint_number: int) -> Hint:
        """Generate context-aware hint based on problem type and user progress."""
        context = self._analyze_problem_context(problem)
        user_context = self._analyze_user_context(user_progress)
        
        if self.ai_client:
            return self._ai_generate_contextual_hint(problem, context, user_context, hint_level, hint_number)
        else:
            return self._fallback_generate_contextual_hint(problem, context, user_context, hint_level, hint_number)
    
    def _analyze_problem_context(self, problem: ParsedProblem) -> Dict[str, Any]:
        """Analyze problem to determine context and difficulty factors."""
        context = {
            'domain': problem.domain,
            'difficulty': problem.difficulty,
            'problem_type': problem.problem_type,
            'variables': problem.variables,
            'expressions': problem.expressions,
            'keywords': self._extract_keywords(problem.original_text)
        }
        
        # Identify specific mathematical concepts
        text_lower = problem.original_text.lower()
        
        if problem.domain == MathDomain.CALCULUS:
            context['calculus_concepts'] = []
            if any(word in text_lower for word in ['derivative', 'differentiate', "d/dx"]):
                context['calculus_concepts'].append('derivative')
            if any(word in text_lower for word in ['integral', 'integrate', 'antiderivative']):
                context['calculus_concepts'].append('integral')
            if any(word in text_lower for word in ['limit', 'approaches', 'tends to']):
                context['calculus_concepts'].append('limit')
            if any(word in text_lower for word in ['maximum', 'minimum', 'optimize', 'critical']):
                context['calculus_concepts'].append('optimization')
        
        elif problem.domain == MathDomain.ALGEBRA:
            context['algebra_concepts'] = []
            if any(word in text_lower for word in ['solve', 'equation', '=']):
                context['algebra_concepts'].append('equation_solving')
            if any(word in text_lower for word in ['factor', 'factorize']):
                context['algebra_concepts'].append('factoring')
            if any(word in text_lower for word in ['simplify', 'expand']):
                context['algebra_concepts'].append('simplification')
        
        elif problem.domain == MathDomain.LINEAR_ALGEBRA:
            context['linalg_concepts'] = []
            if any(word in text_lower for word in ['matrix', 'matrices']):
                context['linalg_concepts'].append('matrix_operations')
            if any(word in text_lower for word in ['vector', 'vectors']):
                context['linalg_concepts'].append('vector_operations')
            if any(word in text_lower for word in ['eigenvalue', 'eigenvector']):
                context['linalg_concepts'].append('eigenanalysis')
        
        return context
    
    def _analyze_user_context(self, user_progress: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user's progress and learning patterns."""
        return {
            'attempts': user_progress.get('attempts', []),
            'common_errors': user_progress.get('common_errors', []),
            'skill_level': user_progress.get('skill_level', 'intermediate'),
            'previous_hints_used': user_progress.get('hints_used', 0),
            'time_spent': user_progress.get('time_spent', 0),
            'similar_problems_solved': user_progress.get('similar_solved', 0)
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract mathematical keywords from problem text."""
        keywords = []
        math_terms = [
            'derivative', 'integral', 'limit', 'function', 'equation', 'solve',
            'factor', 'simplify', 'expand', 'matrix', 'vector', 'eigenvalue',
            'maximum', 'minimum', 'optimize', 'graph', 'plot', 'domain', 'range'
        ]
        
        text_lower = text.lower()
        for term in math_terms:
            if term in text_lower:
                keywords.append(term)
        
        return keywords
    
    def _ai_generate_contextual_hint(self, 
                                   problem: ParsedProblem,
                                   problem_context: Dict[str, Any],
                                   user_context: Dict[str, Any],
                                   hint_level: HintLevel,
                                   hint_number: int) -> Hint:
        """Generate AI-powered contextual hint."""
        # Create enhanced prompt with context
        context_info = f"""
Problem Domain: {problem_context['domain'].value}
Problem Type: {problem_context.get('problem_type', 'unknown')}
User Skill Level: {user_context['skill_level']}
Previous Attempts: {len(user_context['attempts'])}
Hints Already Used: {user_context['previous_hints_used']}
"""
        
        if problem_context['domain'] == MathDomain.CALCULUS:
            concepts = problem_context.get('calculus_concepts', [])
            if concepts:
                context_info += f"Calculus Concepts: {', '.join(concepts)}\n"
        
        prompt = f"""
You are a mathematics tutor providing hint #{hint_number} at {hint_level.value} level.

{context_info}

Problem: {problem.original_text}

Previous user attempts: {', '.join(user_context['attempts'][-3:]) if user_context['attempts'] else 'None'}

Provide a {hint_level.value} hint that:
1. Builds on the user's current understanding
2. Addresses their specific skill level
3. Considers their previous attempts
4. Guides toward the solution without revealing it completely
5. Is appropriate for hint #{hint_number} in the sequence

For {hint_level.value} level:
- GENTLE: Give a conceptual nudge or ask a guiding question
- MODERATE: Suggest a specific approach or method
- DETAILED: Provide step-by-step guidance for the next part
- SOLUTION: Show the complete solution with explanation

Focus on helping the student learn, not just get the answer.
"""
        
        response = self.ai_client.generate_completion(prompt, max_tokens=300, temperature=0.5)
        
        if response['success']:
            return Hint(
                content=response['content'],
                hint_level=hint_level,
                reveals_solution=hint_level == HintLevel.SOLUTION,
                next_step_guidance=self._generate_next_step_guidance(hint_level, problem_context),
                confidence_score=0.85
            )
        else:
            return self._fallback_generate_contextual_hint(problem, problem_context, user_context, hint_level, hint_number)
    
    def _fallback_generate_contextual_hint(self, 
                                         problem: ParsedProblem,
                                         problem_context: Dict[str, Any],
                                         user_context: Dict[str, Any],
                                         hint_level: HintLevel,
                                         hint_number: int) -> Hint:
        """Generate rule-based contextual hint."""
        domain = problem_context['domain']
        concepts = []
        
        if domain == MathDomain.CALCULUS:
            concepts = problem_context.get('calculus_concepts', [])
        elif domain == MathDomain.ALGEBRA:
            concepts = problem_context.get('algebra_concepts', [])
        elif domain == MathDomain.LINEAR_ALGEBRA:
            concepts = problem_context.get('linalg_concepts', [])
        
        # Generate hint based on level and context
        if hint_level == HintLevel.GENTLE:
            content = self._generate_gentle_hint(domain, concepts, problem)
        elif hint_level == HintLevel.MODERATE:
            content = self._generate_moderate_hint(domain, concepts, problem)
        elif hint_level == HintLevel.DETAILED:
            content = self._generate_detailed_hint(domain, concepts, problem)
        else:  # SOLUTION
            content = self._generate_solution_hint(domain, concepts, problem)
        
        return Hint(
            content=content,
            hint_level=hint_level,
            reveals_solution=hint_level == HintLevel.SOLUTION,
            next_step_guidance=self._generate_next_step_guidance(hint_level, problem_context),
            confidence_score=0.6
        )
    
    def _generate_gentle_hint(self, domain: MathDomain, concepts: List[str], problem: ParsedProblem) -> str:
        """Generate gentle hint."""
        if domain == MathDomain.CALCULUS:
            if 'derivative' in concepts:
                return "Think about what rule applies when you have a power of x. What happens to the exponent?"
            elif 'integral' in concepts:
                return "Consider the reverse of differentiation. What function, when differentiated, gives you this?"
            elif 'limit' in concepts:
                return "What value does the function approach as x gets closer to the given point?"
        elif domain == MathDomain.ALGEBRA:
            if 'equation_solving' in concepts:
                return "What operation can you perform on both sides to isolate the variable?"
            elif 'factoring' in concepts:
                return "Look for common factors or patterns like difference of squares."
        
        return "Break down the problem into smaller, manageable steps. What's the first thing you need to identify?"
    
    def _generate_moderate_hint(self, domain: MathDomain, concepts: List[str], problem: ParsedProblem) -> str:
        """Generate moderate hint."""
        if domain == MathDomain.CALCULUS:
            if 'derivative' in concepts:
                return "Use the power rule: if f(x) = x^n, then f'(x) = n*x^(n-1). Apply this to each term."
            elif 'integral' in concepts:
                return "Try the reverse power rule: ∫x^n dx = x^(n+1)/(n+1) + C. Don't forget the constant!"
            elif 'optimization' in concepts:
                return "Find critical points by setting the derivative equal to zero, then test to see if they're maxima or minima."
        elif domain == MathDomain.ALGEBRA:
            if 'equation_solving' in concepts:
                return "Isolate the variable by performing inverse operations. Work systematically from the outside in."
        
        return "Identify the mathematical concept or rule that applies to this type of problem, then apply it step by step."
    
    def _generate_detailed_hint(self, domain: MathDomain, concepts: List[str], problem: ParsedProblem) -> str:
        """Generate detailed hint."""
        if domain == MathDomain.CALCULUS:
            if 'derivative' in concepts:
                return "Step 1: Identify each term in the expression. Step 2: Apply the power rule to each term separately. Step 3: Combine the results."
            elif 'integral' in concepts:
                return "Step 1: Separate the integral into individual terms. Step 2: Apply the power rule for integration to each term. Step 3: Add the constant of integration."
        elif domain == MathDomain.ALGEBRA:
            if 'equation_solving' in concepts:
                return "Step 1: Simplify both sides if needed. Step 2: Move all terms with the variable to one side. Step 3: Move constants to the other side. Step 4: Divide by the coefficient of the variable."
        
        return "Follow these steps: 1) Identify what you're solving for, 2) Choose the appropriate method, 3) Apply it systematically, 4) Check your answer."
    
    def _generate_solution_hint(self, domain: MathDomain, concepts: List[str], problem: ParsedProblem) -> str:
        """Generate solution-level hint."""
        return "Here's the complete approach: [This would contain the full solution with detailed explanation of each step]"
    
    def _generate_next_step_guidance(self, hint_level: HintLevel, problem_context: Dict[str, Any]) -> str:
        """Generate guidance for the next step after the hint."""
        if hint_level == HintLevel.GENTLE:
            return "Think about the hint and try to identify the first step. If you're still stuck, ask for a more detailed hint."
        elif hint_level == HintLevel.MODERATE:
            return "Apply the suggested method. Work through it step by step and see how far you can get."
        elif hint_level == HintLevel.DETAILED:
            return "Follow the outlined steps carefully. Make sure you understand each step before moving to the next."
        else:  # SOLUTION
            return "Study the complete solution and make sure you understand each step. Try a similar problem to test your understanding."
    
    def validate_hint_quality(self, hint: Hint, problem: ParsedProblem, user_context: Dict[str, Any]) -> bool:
        """Validate that the hint is appropriate and helpful."""
        # Check hint length
        if len(hint.content.strip()) < 10:
            return False
        
        # Check that solution-level hints actually reveal more
        if hint.hint_level == HintLevel.SOLUTION and not hint.reveals_solution:
            return False
        
        # Check that gentle hints don't give away too much
        if hint.hint_level == HintLevel.GENTLE and any(word in hint.content.lower() 
                                                      for word in ['answer is', 'solution is', 'equals']):
            return False
        
        return True


class AIExplainer:
    """Main AI explanation service."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 enable_fallback: bool = True):
        """
        Initialize AI explainer.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            enable_fallback: Whether to use rule-based fallbacks
        """
        self.enable_fallback = enable_fallback
        self.templates = PromptTemplates()
        
        try:
            self.client = OpenAIClient(api_key, model)
            self.ai_available = True
            self.hint_generator = HintGenerator(self.client)
        except (ImportError, ValueError) as e:
            self.ai_available = False
            self.hint_generator = HintGenerator(None)
            self.logger = logging.getLogger(__name__)
            self.logger.warning(f"AI service not available: {e}")
            
            if not enable_fallback:
                raise
    
    @error_handler(ErrorCategory.AI_SERVICE, ErrorSeverity.MEDIUM)
    def explain_step(self, 
                    step: SolutionStep, 
                    problem: ParsedProblem,
                    user_level: ExplanationLevel = ExplanationLevel.INTERMEDIATE) -> Explanation:
        """
        Generate explanation for a solution step.
        Enhanced with error handling and fallback systems.
        
        Args:
            step: Solution step to explain
            problem: Problem context
            user_level: User's skill level
            
        Returns:
            Explanation object
        """
        if self.ai_available and graceful_degradation.is_service_available('ai_service'):
            try:
                return self._ai_explain_step(step, problem, user_level)
            except Exception as e:
                graceful_degradation.mark_service_down('ai_service')
                if self.enable_fallback:
                    return self._fallback_explain_step(step, problem, user_level)
                raise AIServiceError(f"AI explanation failed: {str(e)}", "openai")
        elif self.enable_fallback:
            return self._fallback_explain_step(step, problem, user_level)
        else:
            raise AIServiceError("AI service not available and fallback disabled", "openai")
    
    def _ai_explain_step(self, 
                        step: SolutionStep, 
                        problem: ParsedProblem,
                        user_level: ExplanationLevel) -> Explanation:
        """Generate AI-powered step explanation."""
        prompt = self.templates.STEP_EXPLANATION.format(
            problem_context=problem.original_text,
            step_operation=step.operation,
            mathematical_expression=step.mathematical_expression,
            intermediate_result=step.intermediate_result,
            user_level=user_level.value
        )
        
        response = self.client.generate_completion(prompt, max_tokens=400, temperature=0.3)
        
        if response['success']:
            return Explanation(
                content=response['content'],
                complexity_level=user_level,
                related_concepts=self._extract_concepts(response['content']),
                examples=[],
                confidence_score=0.9,
                generation_time=response['generation_time']
            )
        else:
            # Fallback on AI failure
            if self.enable_fallback:
                return self._fallback_explain_step(step, problem, user_level)
            else:
                raise RuntimeError(f"AI explanation failed: {response.get('error', 'Unknown error')}")
    
    def _fallback_explain_step(self, 
                              step: SolutionStep, 
                              problem: ParsedProblem,
                              user_level: ExplanationLevel) -> Explanation:
        """Generate rule-based step explanation as fallback."""
        operation = step.operation.lower()
        problem_type = problem.problem_type.lower() if problem.problem_type else ""
        
        # Basic rule-based explanations based on operation and problem type
        if 'derivative' in operation or 'derivative' in problem_type or problem.domain == MathDomain.CALCULUS:
            if 'power' in operation:
                content = f"Taking the derivative of {step.mathematical_expression} using the power rule."
            else:
                content = f"Taking the derivative of {step.mathematical_expression} using differentiation rules."
        elif 'integral' in operation or 'integral' in problem_type:
            content = f"Integrating {step.mathematical_expression} using integration techniques."
        elif 'simplify' in operation:
            content = f"Simplifying the expression {step.mathematical_expression} by combining like terms."
        elif 'solve' in operation:
            content = f"Solving for the variable in {step.mathematical_expression}."
        elif problem.domain == MathDomain.CALCULUS:
            content = f"Applying calculus techniques to {step.mathematical_expression} to get {step.intermediate_result}."
        elif problem.domain == MathDomain.ALGEBRA:
            content = f"Using algebraic methods on {step.mathematical_expression} to get {step.intermediate_result}."
        else:
            content = f"Performing {step.operation} on {step.mathematical_expression} to get {step.intermediate_result}."
        
        return Explanation(
            content=content,
            complexity_level=user_level,
            related_concepts=[],
            examples=[],
            confidence_score=0.6,
            generation_time=0.001
        )
    
    def generate_hint(self, 
                     problem: ParsedProblem, 
                     current_step: int,
                     hint_level: HintLevel = HintLevel.GENTLE,
                     previous_attempts: List[str] = None) -> Hint:
        """
        Generate contextual hint for problem solving.
        
        Args:
            problem: Problem context
            current_step: Current step number
            hint_level: Level of hint detail
            previous_attempts: User's previous attempts
            
        Returns:
            Hint object
        """
        if previous_attempts is None:
            previous_attempts = []
        
        if self.ai_available:
            return self._ai_generate_hint(problem, current_step, hint_level, previous_attempts)
        elif self.enable_fallback:
            return self._fallback_generate_hint(problem, current_step, hint_level)
        else:
            raise RuntimeError("AI service not available and fallback disabled")
    
    def generate_progressive_hints(self, 
                                 problem: ParsedProblem,
                                 user_progress: Dict[str, Any],
                                 max_hints: int = 4) -> List[Hint]:
        """
        Generate a sequence of progressive hints using the hint generator.
        
        Args:
            problem: Mathematical problem
            user_progress: User's current progress and attempts
            max_hints: Maximum number of hints to generate
            
        Returns:
            List of progressive hints
        """
        return self.hint_generator.generate_progressive_hints(problem, user_progress, max_hints)
    
    def validate_hint_quality(self, hint: Hint, problem: ParsedProblem, user_context: Dict[str, Any]) -> bool:
        """
        Validate hint quality using the hint generator.
        
        Args:
            hint: Generated hint
            problem: Problem context
            user_context: User context
            
        Returns:
            True if hint is of good quality
        """
        return self.hint_generator.validate_hint_quality(hint, problem, user_context)
    
    def detect_user_skill_level(self, user_history: Dict[str, Any]) -> ExplanationLevel:
        """
        Detect user's skill level based on their problem-solving history.
        
        Args:
            user_history: User's performance history
            
        Returns:
            Detected skill level
        """
        if not hasattr(self, 'skill_analyzer'):
            from ai_explainer import UserSkillAnalyzer
            self.skill_analyzer = UserSkillAnalyzer()
        
        return self.skill_analyzer.detect_skill_level(user_history)
    
    def adapt_explanation_to_user(self, 
                                 explanation: str,
                                 user_history: Dict[str, Any],
                                 target_concept: str = "",
                                 context: str = "") -> str:
        """
        Adapt explanation to user's detected skill level.
        
        Args:
            explanation: Original explanation
            user_history: User's learning history
            target_concept: Specific concept being explained
            context: Additional context for adaptation
            
        Returns:
            Adapted explanation
        """
        if not hasattr(self, 'explanation_adapter'):
            from ai_explainer import ExplanationAdapter
            self.explanation_adapter = ExplanationAdapter(self.client if self.ai_available else None)
        
        return self.explanation_adapter.adapt_explanation_to_user(
            explanation, user_history, target_concept, context
        )
    
    def get_skill_progression_recommendations(self, 
                                           user_history: Dict[str, Any]) -> List[str]:
        """
        Get recommendations for user skill progression.
        
        Args:
            user_history: User's performance history
            
        Returns:
            List of recommendations
        """
        if not hasattr(self, 'skill_analyzer'):
            from ai_explainer import UserSkillAnalyzer
            self.skill_analyzer = UserSkillAnalyzer()
        
        current_level = self.skill_analyzer.detect_skill_level(user_history)
        return self.skill_analyzer.get_skill_progression_recommendations(current_level, user_history)
    
    def assess_explanation_appropriateness(self, 
                                         explanation: str,
                                         target_level: ExplanationLevel,
                                         concept: str) -> Dict[str, Any]:
        """
        Assess how well an explanation matches the target level.
        
        Args:
            explanation: Explanation to assess
            target_level: Target skill level
            concept: Mathematical concept
            
        Returns:
            Assessment results
        """
        if not hasattr(self, 'explanation_adapter'):
            from ai_explainer import ExplanationAdapter
            self.explanation_adapter = ExplanationAdapter(self.client if self.ai_available else None)
        
        return self.explanation_adapter.assess_explanation_appropriateness(
            explanation, target_level, concept
        )
    
    def _ai_generate_hint(self, 
                         problem: ParsedProblem, 
                         current_step: int,
                         hint_level: HintLevel,
                         previous_attempts: List[str]) -> Hint:
        """Generate AI-powered hint."""
        prompt = self.templates.HINT_GENERATION.format(
            problem_text=problem.original_text,
            current_step=current_step,
            previous_attempts=', '.join(previous_attempts) if previous_attempts else 'None',
            hint_level=hint_level.value
        )
        
        response = self.client.generate_completion(prompt, max_tokens=300, temperature=0.5)
        
        if response['success']:
            return Hint(
                content=response['content'],
                hint_level=hint_level,
                reveals_solution=hint_level == HintLevel.SOLUTION,
                next_step_guidance="Consider the hint and try the next step.",
                confidence_score=0.85
            )
        else:
            if self.enable_fallback:
                return self._fallback_generate_hint(problem, current_step, hint_level)
            else:
                raise RuntimeError(f"AI hint generation failed: {response.get('error', 'Unknown error')}")
    
    def _fallback_generate_hint(self, 
                               problem: ParsedProblem, 
                               current_step: int,
                               hint_level: HintLevel) -> Hint:
        """Generate rule-based hint as fallback."""
        domain = problem.domain
        
        if domain == MathDomain.CALCULUS:
            if 'derivative' in problem.original_text.lower():
                content = "Try applying the power rule or chain rule for derivatives."
            elif 'integral' in problem.original_text.lower():
                content = "Consider using substitution or integration by parts."
            else:
                content = "Break down the problem into smaller steps."
        elif domain == MathDomain.ALGEBRA:
            content = "Try isolating the variable by performing the same operation on both sides."
        else:
            content = "Consider what mathematical operation or rule applies to this type of problem."
        
        return Hint(
            content=content,
            hint_level=hint_level,
            reveals_solution=False,
            next_step_guidance="Apply the suggested approach and see if it helps.",
            confidence_score=0.5
        )
    
    def answer_why_question(self, 
                           question: str, 
                           context: MathContext) -> Explanation:
        """
        Answer "why" questions about mathematical concepts.
        
        Args:
            question: User's why question
            context: Mathematical context
            
        Returns:
            Explanation object
        """
        if self.ai_available:
            return self._ai_answer_why(question, context)
        elif self.enable_fallback:
            return self._fallback_answer_why(question, context)
        else:
            raise RuntimeError("AI service not available and fallback disabled")
    
    def _ai_answer_why(self, question: str, context: MathContext) -> Explanation:
        """Generate AI-powered answer to why question."""
        prompt = self.templates.WHY_QUESTION.format(
            question=question,
            context=context.problem.original_text,
            user_level=context.user_level.value
        )
        
        response = self.client.generate_completion(prompt, max_tokens=500, temperature=0.4)
        
        if response['success']:
            return Explanation(
                content=response['content'],
                complexity_level=context.user_level,
                related_concepts=self._extract_concepts(response['content']),
                examples=[],
                confidence_score=0.88,
                generation_time=response['generation_time']
            )
        else:
            if self.enable_fallback:
                return self._fallback_answer_why(question, context)
            else:
                raise RuntimeError(f"AI why-question failed: {response.get('error', 'Unknown error')}")
    
    def _fallback_answer_why(self, question: str, context: MathContext) -> Explanation:
        """Generate rule-based answer to why question."""
        content = "This is a fundamental concept in mathematics that helps us understand relationships between quantities and solve problems systematically."
        
        return Explanation(
            content=content,
            complexity_level=context.user_level,
            related_concepts=[],
            examples=[],
            confidence_score=0.4,
            generation_time=0.001
        )
    
    def adapt_explanation_level(self, 
                               explanation: str, 
                               target_level: ExplanationLevel) -> str:
        """
        Adapt explanation complexity to target level.
        
        Args:
            explanation: Original explanation
            target_level: Target complexity level
            
        Returns:
            Adapted explanation
        """
        if not self.ai_available:
            return explanation  # Return unchanged if AI not available
        
        prompt = f"""
Adapt the following mathematical explanation to {target_level.value} level:

Original explanation: {explanation}

Make it appropriate for {target_level.value} level by:
- Adjusting vocabulary and terminology
- Adding or removing technical details
- Changing examples if needed
- Maintaining mathematical accuracy

Adapted explanation:
"""
        
        response = self.client.generate_completion(prompt, max_tokens=400, temperature=0.3)
        
        if response['success']:
            return response['content']
        else:
            return explanation  # Return original on failure
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract mathematical concepts from explanation text."""
        # Simple keyword-based concept extraction
        concepts = []
        keywords = [
            'derivative', 'integral', 'limit', 'function', 'variable',
            'equation', 'matrix', 'vector', 'polynomial', 'trigonometry',
            'calculus', 'algebra', 'geometry', 'statistics', 'probability'
        ]
        
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                concepts.append(keyword)
        
        return list(set(concepts))  # Remove duplicates
    
    def validate_mathematical_accuracy(self, explanation: str, context: MathContext) -> bool:
        """
        Validate mathematical accuracy of generated explanation.
        
        Args:
            explanation: Generated explanation
            context: Mathematical context
            
        Returns:
            True if explanation appears mathematically accurate
        """
        # Basic validation - could be enhanced with more sophisticated checks
        if not explanation or len(explanation.strip()) < 10:
            return False
        
        # Check for common mathematical errors or inconsistencies
        problematic_phrases = [
            'divide by zero',
            'undefined result',
            'impossible operation',
            'mathematical error'
        ]
        
        explanation_lower = explanation.lower()
        for phrase in problematic_phrases:
            if phrase in explanation_lower:
                return False
        
        return True


class UserSkillAnalyzer:
    """Analyzes user skill level based on problem-solving history."""
    
    def __init__(self):
        """Initialize skill analyzer."""
        self.skill_indicators = {
            'accuracy': 0.3,      # Weight for answer accuracy
            'speed': 0.2,         # Weight for solving speed
            'hints_used': 0.2,    # Weight for hint usage (inverse)
            'complexity': 0.3     # Weight for problem complexity handled
        }
    
    def detect_skill_level(self, user_history: Dict[str, Any]) -> ExplanationLevel:
        """
        Detect user's skill level based on their problem-solving history.
        
        Args:
            user_history: Dictionary containing user's performance data
            
        Returns:
            Detected skill level
        """
        if not user_history or not user_history.get('problems_solved', []):
            return ExplanationLevel.INTERMEDIATE  # Default for new users
        
        # Calculate skill metrics
        accuracy_score = self._calculate_accuracy_score(user_history)
        speed_score = self._calculate_speed_score(user_history)
        hint_score = self._calculate_hint_score(user_history)
        complexity_score = self._calculate_complexity_score(user_history)
        
        # Weighted skill score
        total_score = (
            accuracy_score * self.skill_indicators['accuracy'] +
            speed_score * self.skill_indicators['speed'] +
            hint_score * self.skill_indicators['hints_used'] +
            complexity_score * self.skill_indicators['complexity']
        )
        
        # Map score to skill level
        if total_score >= 0.8:
            return ExplanationLevel.EXPERT
        elif total_score >= 0.65:
            return ExplanationLevel.ADVANCED
        elif total_score >= 0.4:
            return ExplanationLevel.INTERMEDIATE
        else:
            return ExplanationLevel.BEGINNER
    
    def _calculate_accuracy_score(self, user_history: Dict[str, Any]) -> float:
        """Calculate accuracy-based skill score."""
        problems = user_history.get('problems_solved', [])
        if not problems:
            return 0.5
        
        correct_count = sum(1 for p in problems if p.get('correct', False))
        return min(correct_count / len(problems), 1.0)
    
    def _calculate_speed_score(self, user_history: Dict[str, Any]) -> float:
        """Calculate speed-based skill score."""
        problems = user_history.get('problems_solved', [])
        if not problems:
            return 0.5
        
        # Calculate average time per problem relative to difficulty
        total_normalized_time = 0
        valid_problems = 0
        
        for problem in problems:
            time_taken = problem.get('time_taken', 0)
            difficulty = problem.get('difficulty', DifficultyLevel.INTERMEDIATE)
            
            if time_taken > 0:
                # Expected time based on difficulty (in seconds)
                expected_times = {
                    DifficultyLevel.BEGINNER: 120,
                    DifficultyLevel.INTERMEDIATE: 300,
                    DifficultyLevel.ADVANCED: 600,
                    DifficultyLevel.EXPERT: 900
                }
                
                expected_time = expected_times.get(difficulty, 300)
                normalized_time = min(time_taken / expected_time, 2.0)  # Cap at 2x expected
                total_normalized_time += normalized_time
                valid_problems += 1
        
        if valid_problems == 0:
            return 0.5
        
        avg_normalized_time = total_normalized_time / valid_problems
        # Convert to score (lower time = higher score)
        return max(0, 1.0 - (avg_normalized_time - 0.5))
    
    def _calculate_hint_score(self, user_history: Dict[str, Any]) -> float:
        """Calculate hint usage-based skill score."""
        problems = user_history.get('problems_solved', [])
        if not problems:
            return 0.5
        
        total_hints = sum(p.get('hints_used', 0) for p in problems)
        avg_hints = total_hints / len(problems)
        
        # Score decreases with more hints used (max 3 hints per problem assumed)
        return max(0, 1.0 - (avg_hints / 3.0))
    
    def _calculate_complexity_score(self, user_history: Dict[str, Any]) -> float:
        """Calculate complexity-based skill score."""
        problems = user_history.get('problems_solved', [])
        if not problems:
            return 0.5
        
        # Count problems by difficulty
        difficulty_counts = {level: 0 for level in DifficultyLevel}
        for problem in problems:
            difficulty = problem.get('difficulty', DifficultyLevel.INTERMEDIATE)
            if problem.get('correct', False):  # Only count correctly solved problems
                difficulty_counts[difficulty] += 1
        
        total_correct = sum(difficulty_counts.values())
        if total_correct == 0:
            return 0.2
        
        # Weight by difficulty level
        difficulty_weights = {
            DifficultyLevel.BEGINNER: 0.25,
            DifficultyLevel.INTERMEDIATE: 0.5,
            DifficultyLevel.ADVANCED: 0.75,
            DifficultyLevel.EXPERT: 1.0
        }
        
        weighted_score = sum(
            difficulty_counts[level] * difficulty_weights[level]
            for level in DifficultyLevel
        ) / total_correct
        
        return weighted_score
    
    def get_skill_progression_recommendations(self, 
                                           current_level: ExplanationLevel,
                                           user_history: Dict[str, Any]) -> List[str]:
        """
        Get recommendations for skill progression.
        
        Args:
            current_level: Current detected skill level
            user_history: User's performance history
            
        Returns:
            List of recommendations for improvement
        """
        recommendations = []
        
        # Analyze weaknesses
        accuracy = self._calculate_accuracy_score(user_history)
        speed = self._calculate_speed_score(user_history)
        hint_usage = self._calculate_hint_score(user_history)
        complexity = self._calculate_complexity_score(user_history)
        
        if accuracy < 0.6:
            recommendations.append("Focus on accuracy by double-checking your work and understanding each step.")
        
        if speed < 0.4:
            recommendations.append("Practice more problems to improve your solving speed and pattern recognition.")
        
        if hint_usage < 0.5:
            recommendations.append("Try to solve problems independently before asking for hints to build confidence.")
        
        if complexity < 0.5:
            recommendations.append("Gradually work on more challenging problems to expand your skill range.")
        
        # Level-specific recommendations
        if current_level == ExplanationLevel.BEGINNER:
            recommendations.append("Focus on mastering fundamental concepts before moving to advanced topics.")
        elif current_level == ExplanationLevel.INTERMEDIATE:
            recommendations.append("Challenge yourself with more complex problems to reach advanced level.")
        elif current_level == ExplanationLevel.ADVANCED:
            recommendations.append("Explore expert-level problems and help others to solidify your understanding.")
        
        return recommendations


class ExplanationAdapter:
    """Adapts explanations to different skill levels and contexts."""
    
    def __init__(self, ai_client: Optional[OpenAIClient] = None):
        """Initialize explanation adapter."""
        self.ai_client = ai_client
        self.skill_analyzer = UserSkillAnalyzer()
    
    def adapt_explanation_to_user(self, 
                                 explanation: str,
                                 user_history: Dict[str, Any],
                                 target_concept: str = "",
                                 context: str = "") -> str:
        """
        Adapt explanation to user's skill level and learning context.
        
        Args:
            explanation: Original explanation
            user_history: User's learning history
            target_concept: Specific concept being explained
            context: Additional context for adaptation
            
        Returns:
            Adapted explanation
        """
        # Detect user's skill level
        user_level = self.skill_analyzer.detect_skill_level(user_history)
        
        # Adapt explanation to detected level
        return self.adapt_explanation_complexity(explanation, user_level, target_concept, context)
    
    def adapt_explanation_complexity(self, 
                                   explanation: str,
                                   target_level: ExplanationLevel,
                                   concept: str = "",
                                   context: str = "") -> str:
        """
        Adapt explanation complexity to target level.
        
        Args:
            explanation: Original explanation
            target_level: Target complexity level
            concept: Mathematical concept being explained
            context: Additional context
            
        Returns:
            Adapted explanation
        """
        if self.ai_client:
            return self._ai_adapt_explanation(explanation, target_level, concept, context)
        else:
            return self._rule_based_adapt_explanation(explanation, target_level)
    
    def _ai_adapt_explanation(self, 
                            explanation: str,
                            target_level: ExplanationLevel,
                            concept: str,
                            context: str) -> str:
        """Use AI to adapt explanation complexity."""
        adaptation_prompts = {
            ExplanationLevel.BEGINNER: """
Adapt this mathematical explanation for a beginner level student:

Original explanation: {explanation}
Concept: {concept}
Context: {context}

Make it beginner-friendly by:
- Using simple, everyday language
- Avoiding complex mathematical terminology
- Including intuitive analogies or examples
- Breaking down complex ideas into smaller steps
- Explaining the "why" behind each concept

Adapted explanation:
""",
            ExplanationLevel.INTERMEDIATE: """
Adapt this mathematical explanation for an intermediate level student:

Original explanation: {explanation}
Concept: {concept}
Context: {context}

Make it intermediate-appropriate by:
- Using standard mathematical terminology with brief explanations
- Including relevant examples and applications
- Connecting to previously learned concepts
- Providing moderate detail without overwhelming
- Balancing conceptual understanding with procedural knowledge

Adapted explanation:
""",
            ExplanationLevel.ADVANCED: """
Adapt this mathematical explanation for an advanced level student:

Original explanation: {explanation}
Concept: {concept}
Context: {context}

Make it advanced-appropriate by:
- Using precise mathematical language and notation
- Including connections to broader mathematical concepts
- Discussing alternative approaches or generalizations
- Providing deeper insights and theoretical context
- Assuming familiarity with prerequisite concepts

Adapted explanation:
""",
            ExplanationLevel.EXPERT: """
Adapt this mathematical explanation for an expert level student:

Original explanation: {explanation}
Concept: {concept}
Context: {context}

Make it expert-appropriate by:
- Using rigorous mathematical language and formal notation
- Including advanced theoretical connections and implications
- Discussing edge cases, proofs, and generalizations
- Connecting to current research or applications
- Assuming deep mathematical maturity

Adapted explanation:
"""
        }
        
        prompt = adaptation_prompts[target_level].format(
            explanation=explanation,
            concept=concept,
            context=context
        )
        
        response = self.ai_client.generate_completion(prompt, max_tokens=500, temperature=0.3)
        
        if response['success']:
            return response['content']
        else:
            return self._rule_based_adapt_explanation(explanation, target_level)
    
    def _rule_based_adapt_explanation(self, 
                                    explanation: str,
                                    target_level: ExplanationLevel) -> str:
        """Use rule-based approach to adapt explanation complexity."""
        if target_level == ExplanationLevel.BEGINNER:
            # Simplify language and add basic explanations
            adapted = explanation.replace("derivative", "rate of change")
            adapted = adapted.replace("integral", "area under the curve")
            adapted = adapted.replace("function", "mathematical relationship")
            
            # Add encouraging language
            if not adapted.startswith("Let's"):
                adapted = "Let's work through this step by step. " + adapted
            
        elif target_level == ExplanationLevel.ADVANCED:
            # Add more technical detail
            if "power rule" in explanation.lower():
                adapted = explanation + " This follows from the limit definition of the derivative."
            elif "integration" in explanation.lower():
                adapted = explanation + " This is based on the Fundamental Theorem of Calculus."
            else:
                adapted = explanation
                
        elif target_level == ExplanationLevel.EXPERT:
            # Add theoretical context
            adapted = explanation + " Consider the broader theoretical implications and connections to other mathematical structures."
            
        else:  # INTERMEDIATE
            adapted = explanation
        
        return adapted
    
    def generate_level_appropriate_examples(self, 
                                          concept: str,
                                          target_level: ExplanationLevel,
                                          count: int = 2) -> List[str]:
        """
        Generate examples appropriate for the target skill level.
        
        Args:
            concept: Mathematical concept
            target_level: Target skill level
            count: Number of examples to generate
            
        Returns:
            List of level-appropriate examples
        """
        examples = []
        
        if concept.lower() == "derivative":
            if target_level == ExplanationLevel.BEGINNER:
                examples = [
                    "If f(x) = x², then f'(x) = 2x",
                    "If f(x) = 3x, then f'(x) = 3"
                ]
            elif target_level == ExplanationLevel.INTERMEDIATE:
                examples = [
                    "If f(x) = x³ + 2x² - 5x + 1, then f'(x) = 3x² + 4x - 5",
                    "If f(x) = sin(x), then f'(x) = cos(x)"
                ]
            elif target_level == ExplanationLevel.ADVANCED:
                examples = [
                    "If f(x) = e^(x²), then f'(x) = 2xe^(x²) using the chain rule",
                    "If f(x) = ln(x² + 1), then f'(x) = 2x/(x² + 1)"
                ]
            else:  # EXPERT
                examples = [
                    "For f: ℝⁿ → ℝᵐ, the Jacobian matrix represents the linear approximation",
                    "The Fréchet derivative generalizes the concept to infinite-dimensional spaces"
                ]
        
        elif concept.lower() == "integral":
            if target_level == ExplanationLevel.BEGINNER:
                examples = [
                    "∫x dx = x²/2 + C",
                    "∫1 dx = x + C"
                ]
            elif target_level == ExplanationLevel.INTERMEDIATE:
                examples = [
                    "∫(x² + 3x - 2) dx = x³/3 + 3x²/2 - 2x + C",
                    "∫sin(x) dx = -cos(x) + C"
                ]
            elif target_level == ExplanationLevel.ADVANCED:
                examples = [
                    "∫x·e^x dx = (x-1)e^x + C using integration by parts",
                    "∫₀^π sin²(x) dx = π/2 using trigonometric identities"
                ]
            else:  # EXPERT
                examples = [
                    "The Lebesgue integral extends integration to more general measure spaces",
                    "Stokes' theorem generalizes the fundamental theorem of calculus to manifolds"
                ]
        
        return examples[:count]
    
    def assess_explanation_appropriateness(self, 
                                         explanation: str,
                                         target_level: ExplanationLevel,
                                         concept: str) -> Dict[str, Any]:
        """
        Assess how well an explanation matches the target level.
        
        Args:
            explanation: Explanation to assess
            target_level: Target skill level
            concept: Mathematical concept
            
        Returns:
            Assessment results
        """
        assessment = {
            'appropriateness_score': 0.0,
            'issues': [],
            'suggestions': []
        }
        
        explanation_lower = explanation.lower()
        
        # Check vocabulary complexity
        advanced_terms = ['theorem', 'lemma', 'corollary', 'rigorous', 'formal', 'proof']
        beginner_terms = ['simple', 'easy', 'basic', 'step by step', 'let\'s']
        
        advanced_count = sum(1 for term in advanced_terms if term in explanation_lower)
        beginner_count = sum(1 for term in beginner_terms if term in explanation_lower)
        
        # Assess appropriateness based on level
        if target_level == ExplanationLevel.BEGINNER:
            if advanced_count > 2:
                assessment['issues'].append("Too many advanced terms for beginner level")
                assessment['suggestions'].append("Simplify vocabulary and add more basic explanations")
            if beginner_count > 0:
                assessment['appropriateness_score'] += 0.3
        
        elif target_level == ExplanationLevel.EXPERT:
            if beginner_count > 2:
                assessment['issues'].append("Too simplistic for expert level")
                assessment['suggestions'].append("Add more theoretical depth and formal notation")
            if advanced_count > 0:
                assessment['appropriateness_score'] += 0.3
        
        # Check length appropriateness
        word_count = len(explanation.split())
        expected_lengths = {
            ExplanationLevel.BEGINNER: (50, 150),
            ExplanationLevel.INTERMEDIATE: (80, 200),
            ExplanationLevel.ADVANCED: (100, 300),
            ExplanationLevel.EXPERT: (120, 400)
        }
        
        min_len, max_len = expected_lengths[target_level]
        if word_count < min_len:
            assessment['issues'].append("Explanation too brief for level")
            assessment['suggestions'].append("Add more detail and examples")
        elif word_count > max_len:
            assessment['issues'].append("Explanation too verbose for level")
            assessment['suggestions'].append("Condense to essential information")
        else:
            assessment['appropriateness_score'] += 0.4
        
        # Check for examples
        has_examples = any(phrase in explanation_lower 
                          for phrase in ['for example', 'such as', 'like', 'consider'])
        
        if target_level in [ExplanationLevel.BEGINNER, ExplanationLevel.INTERMEDIATE] and not has_examples:
            assessment['issues'].append("Missing examples for this level")
            assessment['suggestions'].append("Add concrete examples to illustrate concepts")
        elif has_examples:
            assessment['appropriateness_score'] += 0.3
        
        # Ensure score is between 0 and 1
        assessment['appropriateness_score'] = min(assessment['appropriateness_score'], 1.0)
        
        return assessment