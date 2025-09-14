"""
Quiz Generation System for AI Math Tutor
Generates adaptive quizzes based on topic and difficulty level.
"""

import random
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import sympy as sp
from sympy import symbols, expand, factor, diff, integrate, solve, simplify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
import sys
import os

# Add shared models to path
from models import (
    Question, Quiz, QuestionType, DifficultyLevel, MathDomain,
    ParsedProblem, MathTutorError
)


class ProblemBank:
    """Database of mathematical problems organized by topic and difficulty."""
    
    def __init__(self):
        """Initialize the problem bank with predefined problems."""
        self.problems = {
            MathDomain.ALGEBRA: {
                DifficultyLevel.BEGINNER: [
                    {
                        'template': 'Solve for x: {a}x + {b} = {c}',
                        'type': 'linear_equation',
                        'variables': ['a', 'b', 'c'],
                        'constraints': {'a': (1, 10), 'b': (-10, 10), 'c': (-20, 20)},
                        'answer_formula': '({c} - {b}) / {a}'
                    },
                    {
                        'template': 'Simplify: {a}x + {b}x',
                        'type': 'simplification',
                        'variables': ['a', 'b'],
                        'constraints': {'a': (1, 10), 'b': (1, 10)},
                        'answer_formula': '({a} + {b})x'
                    },
                    {
                        'template': 'Expand: ({a}x + {b})({c}x + {d})',
                        'type': 'expansion',
                        'variables': ['a', 'b', 'c', 'd'],
                        'constraints': {'a': (1, 5), 'b': (-5, 5), 'c': (1, 5), 'd': (-5, 5)},
                        'answer_formula': '{a}*{c}*x^2 + ({a}*{d} + {b}*{c})*x + {b}*{d}'
                    }
                ],
                DifficultyLevel.INTERMEDIATE: [
                    {
                        'template': 'Solve for x: {a}x² + {b}x + {c} = 0',
                        'type': 'quadratic_equation',
                        'variables': ['a', 'b', 'c'],
                        'constraints': {'a': (1, 5), 'b': (-10, 10), 'c': (-10, 10)},
                        'answer_formula': 'quadratic_formula'
                    },
                    {
                        'template': 'Factor: x² + {b}x + {c}',
                        'type': 'factoring',
                        'variables': ['b', 'c'],
                        'constraints': {'b': (-10, 10), 'c': (-25, 25)},
                        'answer_formula': 'factored_form'
                    }
                ],
                DifficultyLevel.ADVANCED: [
                    {
                        'template': 'Solve the system: {a}x + {b}y = {c}, {d}x + {e}y = {f}',
                        'type': 'system_of_equations',
                        'variables': ['a', 'b', 'c', 'd', 'e', 'f'],
                        'constraints': {'a': (1, 5), 'b': (1, 5), 'c': (-10, 10), 'd': (1, 5), 'e': (1, 5), 'f': (-10, 10)},
                        'answer_formula': 'system_solution'
                    }
                ]
            },
            MathDomain.CALCULUS: {
                DifficultyLevel.BEGINNER: [
                    {
                        'template': 'Find the derivative of {a}x² + {b}x + {c}',
                        'type': 'derivative',
                        'variables': ['a', 'b', 'c'],
                        'constraints': {'a': (1, 10), 'b': (-10, 10), 'c': (-10, 10)},
                        'answer_formula': '{2*a}*x + {b}'
                    },
                    {
                        'template': 'Find the derivative of x^{n}',
                        'type': 'power_rule',
                        'variables': ['n'],
                        'constraints': {'n': (2, 8)},
                        'answer_formula': '{n}*x^({n-1})'
                    }
                ],
                DifficultyLevel.INTERMEDIATE: [
                    {
                        'template': 'Find the derivative of ({a}x + {b})^{n}',
                        'type': 'chain_rule',
                        'variables': ['a', 'b', 'n'],
                        'constraints': {'a': (1, 5), 'b': (-5, 5), 'n': (2, 4)},
                        'answer_formula': '{n}*({a}x + {b})^({n-1})*{a}'
                    },
                    {
                        'template': 'Integrate: {a}x² + {b}x + {c}',
                        'type': 'integration',
                        'variables': ['a', 'b', 'c'],
                        'constraints': {'a': (1, 10), 'b': (-10, 10), 'c': (-10, 10)},
                        'answer_formula': '{a/3}*x³ + {b/2}*x² + {c}*x + C'
                    }
                ],
                DifficultyLevel.ADVANCED: [
                    {
                        'template': 'Find the limit as x approaches {a}: (x² - {a}²)/(x - {a})',
                        'type': 'limit',
                        'variables': ['a'],
                        'constraints': {'a': (1, 10)},
                        'answer_formula': '2*{a}'
                    }
                ]
            },
            MathDomain.LINEAR_ALGEBRA: {
                DifficultyLevel.BEGINNER: [
                    {
                        'template': 'Add the vectors [{a}, {b}] + [{c}, {d}]',
                        'type': 'vector_addition',
                        'variables': ['a', 'b', 'c', 'd'],
                        'constraints': {'a': (-10, 10), 'b': (-10, 10), 'c': (-10, 10), 'd': (-10, 10)},
                        'answer_formula': '[{a+c}, {b+d}]'
                    },
                    {
                        'template': 'Find the dot product of [{a}, {b}] · [{c}, {d}]',
                        'type': 'dot_product',
                        'variables': ['a', 'b', 'c', 'd'],
                        'constraints': {'a': (-5, 5), 'b': (-5, 5), 'c': (-5, 5), 'd': (-5, 5)},
                        'answer_formula': '{a*c + b*d}'
                    }
                ],
                DifficultyLevel.INTERMEDIATE: [
                    {
                        'template': 'Find the determinant of the 2x2 matrix [[{a}, {b}], [{c}, {d}]]',
                        'type': 'determinant_2x2',
                        'variables': ['a', 'b', 'c', 'd'],
                        'constraints': {'a': (-5, 5), 'b': (-5, 5), 'c': (-5, 5), 'd': (-5, 5)},
                        'answer_formula': '{a*d - b*c}'
                    }
                ]
            }
        }
        
        # Standard transformations for parsing
        self.transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    
    def get_problems_by_criteria(self, domain: MathDomain, difficulty: DifficultyLevel, 
                               count: int = 10) -> List[Dict]:
        """Get problems matching the specified criteria."""
        if domain not in self.problems:
            raise MathTutorError(f"Domain {domain} not supported")
        
        if difficulty not in self.problems[domain]:
            raise MathTutorError(f"Difficulty {difficulty} not available for {domain}")
        
        available_problems = self.problems[domain][difficulty]
        
        # If we need more problems than available templates, repeat with variations
        selected_problems = []
        for i in range(count):
            template = available_problems[i % len(available_problems)]
            selected_problems.append(template)
        
        return selected_problems
    
    def generate_problem_instance(self, template: Dict) -> Tuple[str, str, Dict]:
        """Generate a specific problem instance from a template."""
        # Generate random values for variables
        values = {}
        for var, (min_val, max_val) in template['constraints'].items():
            values[var] = random.randint(min_val, max_val)
        
        # Generate problem text
        problem_text = template['template'].format(**values)
        
        # Calculate correct answer
        correct_answer = self._calculate_answer(template, values)
        
        return problem_text, correct_answer, values
    
    def _calculate_answer(self, template: Dict, values: Dict) -> str:
        """Calculate the correct answer for a problem instance."""
        answer_formula = template['answer_formula']
        problem_type = template['type']
        
        try:
            if problem_type == 'linear_equation':
                # (c - b) / a
                result = (values['c'] - values['b']) / values['a']
                return str(result)
            
            elif problem_type == 'simplification':
                # (a + b)x
                coeff = values['a'] + values['b']
                return f"{coeff}x" if coeff != 1 else "x"
            
            elif problem_type == 'expansion':
                # ax*cx + ax*d + bx*c + b*d = acx² + (ad + bc)x + bd
                a, b, c, d = values['a'], values['b'], values['c'], values['d']
                x2_coeff = a * c
                x_coeff = a * d + b * c
                const = b * d
                
                terms = []
                if x2_coeff != 0:
                    if x2_coeff == 1:
                        terms.append("x²")
                    elif x2_coeff == -1:
                        terms.append("-x²")
                    else:
                        terms.append(f"{x2_coeff}x²")
                
                if x_coeff != 0:
                    if x_coeff == 1:
                        terms.append("x" if not terms else " + x")
                    elif x_coeff == -1:
                        terms.append("-x" if not terms else " - x")
                    else:
                        if terms and x_coeff > 0:
                            terms.append(f" + {x_coeff}x")
                        else:
                            terms.append(f"{x_coeff}x")
                
                if const != 0:
                    if terms and const > 0:
                        terms.append(f" + {const}")
                    else:
                        terms.append(str(const))
                
                return "".join(terms) if terms else "0"
            
            elif problem_type == 'quadratic_equation':
                # Use quadratic formula
                a, b, c = values['a'], values['b'], values['c']
                discriminant = b**2 - 4*a*c
                
                if discriminant < 0:
                    return "No real solutions"
                elif discriminant == 0:
                    root = -b / (2*a)
                    return str(root)
                else:
                    import math
                    root1 = (-b + math.sqrt(discriminant)) / (2*a)
                    root2 = (-b - math.sqrt(discriminant)) / (2*a)
                    return f"{root1}, {root2}"
            
            elif problem_type == 'factoring':
                # Factor x² + bx + c
                b, c = values['b'], values['c']
                # Find factors of c that add up to b
                for i in range(-abs(c), abs(c) + 1):
                    if i != 0 and c % i == 0:
                        j = c // i
                        if i + j == b:
                            return f"(x + {i})(x + {j})"
                return f"x² + {b}x + {c}"  # Cannot factor
            
            elif problem_type == 'derivative':
                # Derivative of ax² + bx + c is 2ax + b
                a, b = values['a'], values['b']
                terms = []
                if a != 0:
                    coeff = 2 * a
                    if coeff == 1:
                        terms.append("x")
                    elif coeff == -1:
                        terms.append("-x")
                    else:
                        terms.append(f"{coeff}x")
                
                if b != 0:
                    if terms and b > 0:
                        terms.append(f" + {b}")
                    else:
                        terms.append(str(b))
                
                return "".join(terms) if terms else "0"
            
            elif problem_type == 'power_rule':
                # Derivative of x^n is nx^(n-1)
                n = values['n']
                if n == 2:
                    return "2x"
                elif n == 1:
                    return "1"
                else:
                    return f"{n}x^{n-1}"
            
            elif problem_type == 'vector_addition':
                # [a, b] + [c, d] = [a+c, b+d]
                result_x = values['a'] + values['c']
                result_y = values['b'] + values['d']
                return f"[{result_x}, {result_y}]"
            
            elif problem_type == 'dot_product':
                # [a, b] · [c, d] = ac + bd
                result = values['a'] * values['c'] + values['b'] * values['d']
                return str(result)
            
            elif problem_type == 'determinant_2x2':
                # det([[a, b], [c, d]]) = ad - bc
                result = values['a'] * values['d'] - values['b'] * values['c']
                return str(result)
            
            else:
                return "Answer calculation not implemented"
                
        except Exception as e:
            return f"Error calculating answer: {str(e)}"


class QuizGenerator:
    """Generates adaptive quizzes based on user preferences and performance."""
    
    def __init__(self):
        """Initialize the quiz generator."""
        self.problem_bank = ProblemBank()
        self.question_types = {
            'numeric': QuestionType.NUMERIC,
            'algebraic': QuestionType.ALGEBRAIC,
            'multiple_choice': QuestionType.MULTIPLE_CHOICE
        }
    
    def generate_quiz(self, topic: str, difficulty: DifficultyLevel, 
                     num_questions: int = 10, question_type: str = 'mixed') -> Quiz:
        """
        Generate a quiz based on specified parameters.
        
        Args:
            topic: Mathematical topic (algebra, calculus, linear_algebra)
            difficulty: Difficulty level
            num_questions: Number of questions to generate
            question_type: Type of questions ('numeric', 'algebraic', 'multiple_choice', 'mixed')
            
        Returns:
            Quiz: Generated quiz with questions
        """
        try:
            # Convert topic string to MathDomain
            domain = self._string_to_domain(topic)
            
            # Get problem templates
            problem_templates = self.problem_bank.get_problems_by_criteria(
                domain, difficulty, num_questions
            )
            
            # Generate questions
            questions = []
            for i, template in enumerate(problem_templates):
                question = self._generate_question(template, i + 1, question_type)
                questions.append(question)
            
            # Create quiz
            quiz = Quiz(
                id=str(uuid.uuid4()),
                title=f"{topic.title()} Quiz - {difficulty.name.title()}",
                questions=questions,
                time_limit=num_questions * 120,  # 2 minutes per question
                topic=topic,
                difficulty=difficulty,
                created_at=datetime.now()
            )
            
            return quiz
            
        except Exception as e:
            raise MathTutorError(f"Failed to generate quiz: {str(e)}")
    
    def generate_similar_problems(self, original_problem: str, count: int = 5) -> List[str]:
        """
        Generate similar problems based on an original problem.
        
        Args:
            original_problem: The original problem text
            count: Number of similar problems to generate
            
        Returns:
            List of similar problem texts
        """
        # This is a simplified implementation
        # In a full system, this would analyze the problem structure and generate variations
        similar_problems = []
        
        # For now, generate variations by changing numbers in the problem
        import re
        
        for i in range(count):
            # Find numbers in the problem and replace with random variations
            def replace_number(match):
                original_num = int(match.group())
                # Generate a number within ±50% of the original
                variation = random.randint(
                    max(1, int(original_num * 0.5)),
                    int(original_num * 1.5)
                )
                return str(variation)
            
            varied_problem = re.sub(r'\b\d+\b', replace_number, original_problem)
            similar_problems.append(varied_problem)
        
        return similar_problems
    
    def _generate_question(self, template: Dict, question_num: int, 
                          question_type: str) -> Question:
        """Generate a single question from a template."""
        # Generate problem instance
        problem_text, correct_answer, values = self.problem_bank.generate_problem_instance(template)
        
        # Determine question type
        if question_type == 'mixed':
            # Randomly choose question type
            q_type = random.choice([QuestionType.NUMERIC, QuestionType.ALGEBRAIC])
        else:
            q_type = self.question_types.get(question_type, QuestionType.NUMERIC)
        
        # Generate options for multiple choice
        options = []
        if q_type == QuestionType.MULTIPLE_CHOICE:
            options = self._generate_multiple_choice_options(correct_answer, template['type'])
        
        # Generate hints
        hints = self._generate_hints(template, values)
        
        return Question(
            id=str(uuid.uuid4()),
            text=problem_text,
            question_type=q_type,
            options=options,
            correct_answer=correct_answer,
            hints=hints,
            difficulty=DifficultyLevel.INTERMEDIATE,  # Will be set by caller
            topic=template['type']
        )
    
    def _generate_multiple_choice_options(self, correct_answer: str, 
                                        problem_type: str) -> List[str]:
        """Generate plausible multiple choice options."""
        options = [correct_answer]
        
        try:
            if problem_type in ['linear_equation', 'quadratic_equation']:
                # Generate numerical distractors
                if ',' in correct_answer:  # Multiple solutions
                    # For now, just add the correct answer
                    pass
                else:
                    try:
                        correct_num = float(correct_answer)
                        # Generate nearby wrong answers
                        options.extend([
                            str(int(correct_num + random.randint(1, 5))),
                            str(int(correct_num - random.randint(1, 5))),
                            str(int(correct_num * 2)),
                            str(int(-correct_num))
                        ])
                    except ValueError:
                        # Non-numeric answer, generate algebraic distractors
                        options.extend([
                            correct_answer.replace('x', '2x'),
                            correct_answer.replace('+', '-'),
                            f"2({correct_answer})"
                        ])
            
            elif problem_type in ['expansion', 'factoring', 'simplification']:
                # Generate algebraic distractors
                options.extend([
                    correct_answer.replace('+', '-'),
                    correct_answer.replace('x²', '2x²'),
                    correct_answer.replace('x', '2x'),
                    f"({correct_answer})"
                ])
            
            # Ensure we have exactly 4 options and shuffle
            options = list(set(options))[:4]  # Remove duplicates and limit to 4
            while len(options) < 4:
                options.append(f"Option {len(options) + 1}")
            
            random.shuffle(options)
            
        except Exception:
            # Fallback to generic options
            options = [correct_answer, "Option A", "Option B", "Option C"]
            random.shuffle(options)
        
        return options
    
    def _generate_hints(self, template: Dict, values: Dict) -> List[str]:
        """Generate progressive hints for a problem."""
        problem_type = template['type']
        hints = []
        
        if problem_type == 'linear_equation':
            hints = [
                "Start by isolating the variable term on one side",
                "Move constants to the other side of the equation",
                "Divide both sides by the coefficient of the variable"
            ]
        elif problem_type == 'quadratic_equation':
            hints = [
                "This is a quadratic equation - consider using the quadratic formula",
                "First, make sure the equation is in standard form ax² + bx + c = 0",
                "The quadratic formula is x = (-b ± √(b² - 4ac)) / (2a)"
            ]
        elif problem_type == 'expansion':
            hints = [
                "Use the distributive property (FOIL method)",
                "Multiply each term in the first parentheses by each term in the second",
                "Combine like terms in your final answer"
            ]
        elif problem_type == 'derivative':
            hints = [
                "Use the power rule: d/dx[x^n] = nx^(n-1)",
                "The derivative of a constant is 0",
                "Apply the rule to each term separately"
            ]
        else:
            hints = [
                "Read the problem carefully and identify what you're solving for",
                "Break the problem down into smaller steps",
                "Check your work by substituting back into the original problem"
            ]
        
        return hints
    
    def _string_to_domain(self, topic: str) -> MathDomain:
        """Convert topic string to MathDomain enum."""
        topic_mapping = {
            'algebra': MathDomain.ALGEBRA,
            'calculus': MathDomain.CALCULUS,
            'linear_algebra': MathDomain.LINEAR_ALGEBRA,
            'statistics': MathDomain.STATISTICS,
            'ai_ml_math': MathDomain.AI_ML_MATH
        }
        
        if topic.lower() not in topic_mapping:
            raise MathTutorError(f"Unsupported topic: {topic}")
        
        return topic_mapping[topic.lower()]