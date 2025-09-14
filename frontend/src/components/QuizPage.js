import React, { useState, useEffect } from 'react';
import {
  Typography,
  Box,
  Card,
  CardContent,
  Button,
  RadioGroup,
  FormControlLabel,
  Radio,
  TextField,
  LinearProgress,
  Alert,
  Chip,
  Grid,
  Paper,
  Divider,
  useTheme,
  useMediaQuery
} from '@mui/material';
import {
  CheckCircle as CheckIcon,
  Cancel as CancelIcon,
  Timer as TimerIcon,
  Quiz as QuizIcon
} from '@mui/icons-material';
import MathRenderer from './MathRenderer';
import { useGuestLimitations } from '../contexts/GuestLimitationContext';
import { useAuth } from '../contexts/AuthContext';

function QuizPage() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [currentQuiz, setCurrentQuiz] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [userAnswer, setUserAnswer] = useState('');
  const [showFeedback, setShowFeedback] = useState(false);
  const [quizResults, setQuizResults] = useState(null);
  const [timeLeft, setTimeLeft] = useState(0);
  const [selectedTopic, setSelectedTopic] = useState('');

  // Guest limitation hooks
  const { trackUsage, showPrompt } = useGuestLimitations();
  const { isAuthenticated } = useAuth();

  const topics = [
    { id: 'algebra', name: 'Algebra', difficulty: 'Beginner' },
    { id: 'calculus', name: 'Calculus', difficulty: 'Intermediate' },
    { id: 'linear-algebra', name: 'Linear Algebra', difficulty: 'Advanced' },
    { id: 'statistics', name: 'Statistics', difficulty: 'Intermediate' }
  ];

  // Mock quiz data
  const mockQuizzes = {
    algebra: {
      title: 'Algebra Fundamentals',
      timeLimit: 300, // 5 minutes
      questions: [
        {
          id: 1,
          question: 'Solve for x: $2x + 3 = 7$',
          type: 'multiple-choice',
          options: ['x = 1', 'x = 2', 'x = 3', 'x = 4'],
          correctAnswer: 'x = 2',
          explanation: 'Subtract 3 from both sides: $2x = 4$, then divide by 2: $x = 2$'
        },
        {
          id: 2,
          question: 'What is the slope of the line $y = 3x - 2$?',
          type: 'text',
          correctAnswer: '3',
          explanation: 'In the slope-intercept form $y = mx + b$, the coefficient of $x$ is the slope, which is 3.'
        },
        {
          id: 3,
          question: 'Factor the expression: $x^2 - 5x + 6$',
          type: 'multiple-choice',
          options: ['$(x-2)(x-3)$', '$(x-1)(x-6)$', '$(x+2)(x+3)$', '$(x-6)(x-1)$'],
          correctAnswer: '$(x-2)(x-3)$',
          explanation: 'We need two numbers that multiply to 6 and add to -5. These are -2 and -3.'
        }
      ]
    },
    calculus: {
      title: 'Calculus Basics',
      timeLimit: 600, // 10 minutes
      questions: [
        {
          id: 1,
          question: 'Find the derivative of $f(x) = x^2 + 3x$',
          type: 'text',
          correctAnswer: '2x + 3',
          explanation: 'Using the power rule: $\\frac{d}{dx}(x^2) = 2x$ and $\\frac{d}{dx}(3x) = 3$'
        },
        {
          id: 2,
          question: 'What is $\\int 2x \\, dx$?',
          type: 'multiple-choice',
          options: ['$x^2 + C$', '$2x^2 + C$', '$x^2/2 + C$', '$2x + C$'],
          correctAnswer: '$x^2 + C$',
          explanation: 'Using the power rule for integration: $\\int x^n dx = \\frac{x^{n+1}}{n+1} + C$'
        }
      ]
    }
  };

  useEffect(() => {
    let timer;
    if (currentQuiz && timeLeft > 0 && !showFeedback) {
      timer = setTimeout(() => setTimeLeft(timeLeft - 1), 1000);
    } else if (timeLeft === 0 && currentQuiz && !showFeedback) {
      handleSubmitAnswer();
    }
    return () => clearTimeout(timer);
  }, [timeLeft, currentQuiz, showFeedback]);

  const startQuiz = (topicId) => {
    // Track usage for guest users and check limits
    if (!isAuthenticated) {
      trackUsage('QUIZ');
      
      // Check if limit is exceeded and show prompt
      const limitExceeded = showPrompt('QUIZ');
      if (limitExceeded) {
        // The prompt will be shown by the context, but we can still continue
        // This allows users to see the prompt but still use the feature
      }
    }

    const quiz = mockQuizzes[topicId];
    if (quiz) {
      setCurrentQuiz(quiz);
      setCurrentQuestion(0);
      setTimeLeft(quiz.timeLimit);
      setUserAnswer('');
      setShowFeedback(false);
      setQuizResults(null);
      setSelectedTopic(topicId);
    }
  };

  const handleSubmitAnswer = () => {
    const question = currentQuiz.questions[currentQuestion];
    const isCorrect = userAnswer.toLowerCase().trim() === question.correctAnswer.toLowerCase().trim();
    
    setShowFeedback(true);
    
    // Update results
    const newResults = quizResults || {
      correct: 0,
      total: currentQuiz.questions.length,
      answers: []
    };
    
    newResults.answers[currentQuestion] = {
      question: question.question,
      userAnswer,
      correctAnswer: question.correctAnswer,
      isCorrect,
      explanation: question.explanation
    };
    
    if (isCorrect) {
      newResults.correct++;
    }
    
    setQuizResults(newResults);
  };

  const nextQuestion = () => {
    if (currentQuestion < currentQuiz.questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
      setUserAnswer('');
      setShowFeedback(false);
    } else {
      // Quiz completed
      setCurrentQuiz(null);
    }
  };

  const resetQuiz = () => {
    setCurrentQuiz(null);
    setCurrentQuestion(0);
    setUserAnswer('');
    setShowFeedback(false);
    setQuizResults(null);
    setTimeLeft(0);
    setSelectedTopic('');
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Quiz selection screen
  if (!currentQuiz && !quizResults) {
    return (
      <Box>
        <Typography variant="h2" component="h1" gutterBottom>
          Interactive Quizzes
        </Typography>
        
        <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
          Test your knowledge with adaptive quizzes that provide immediate feedback and explanations.
        </Typography>

        <Grid container spacing={3}>
          {topics.map((topic) => (
            <Grid item xs={12} sm={6} md={4} key={topic.id}>
              <Card 
                sx={{ 
                  cursor: 'pointer',
                  transition: 'transform 0.2s',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: 4
                  }
                }}
                onClick={() => startQuiz(topic.id)}
              >
                <CardContent sx={{ textAlign: 'center' }}>
                  <QuizIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                  <Typography variant="h5" gutterBottom>
                    {topic.name}
                  </Typography>
                  <Chip 
                    label={topic.difficulty} 
                    color={topic.difficulty === 'Beginner' ? 'success' : topic.difficulty === 'Intermediate' ? 'warning' : 'error'}
                    sx={{ mb: 2 }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    Click to start quiz
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  // Quiz results screen
  if (quizResults && !currentQuiz) {
    const percentage = Math.round((quizResults.correct / quizResults.total) * 100);
    
    return (
      <Box>
        <Typography variant="h2" component="h1" gutterBottom>
          Quiz Results
        </Typography>

        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box sx={{ textAlign: 'center', mb: 3 }}>
              <Typography variant="h3" color={percentage >= 70 ? 'success.main' : 'error.main'}>
                {percentage}%
              </Typography>
              <Typography variant="h6" color="text.secondary">
                {quizResults.correct} out of {quizResults.total} correct
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={percentage} 
                sx={{ mt: 2, height: 8, borderRadius: 4 }}
                color={percentage >= 70 ? 'success' : 'error'}
              />
            </Box>

            <Divider sx={{ my: 3 }} />

            <Typography variant="h6" gutterBottom>
              Question Review
            </Typography>

            {quizResults.answers.map((answer, index) => (
              <Paper key={index} sx={{ p: 2, mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  {answer.isCorrect ? (
                    <CheckIcon color="success" sx={{ mr: 1 }} />
                  ) : (
                    <CancelIcon color="error" sx={{ mr: 1 }} />
                  )}
                  <Typography variant="subtitle1">
                    Question {index + 1}
                  </Typography>
                </Box>
                
                <Box sx={{ mb: 2 }}>
                  <MathRenderer math={answer.question} />
                </Box>
                
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Your answer: <strong>{answer.userAnswer || 'No answer'}</strong>
                </Typography>
                
                {!answer.isCorrect && (
                  <Typography variant="body2" color="success.main" gutterBottom>
                    Correct answer: <MathRenderer math={answer.correctAnswer} />
                  </Typography>
                )}
                
                <Alert severity="info" sx={{ mt: 1 }}>
                  <MathRenderer math={answer.explanation} />
                </Alert>
              </Paper>
            ))}

            <Box sx={{ textAlign: 'center', mt: 3 }}>
              <Button variant="contained" onClick={resetQuiz} size="large">
                Take Another Quiz
              </Button>
            </Box>
          </CardContent>
        </Card>
      </Box>
    );
  }

  // Active quiz screen
  const question = currentQuiz.questions[currentQuestion];
  const progress = ((currentQuestion + 1) / currentQuiz.questions.length) * 100;

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          {currentQuiz.title}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <TimerIcon sx={{ mr: 1 }} />
          <Typography variant="h6" color={timeLeft < 60 ? 'error.main' : 'text.primary'}>
            {formatTime(timeLeft)}
          </Typography>
        </Box>
      </Box>

      <LinearProgress 
        variant="determinate" 
        value={progress} 
        sx={{ mb: 3, height: 8, borderRadius: 4 }}
      />

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Question {currentQuestion + 1} of {currentQuiz.questions.length}
          </Typography>

          <Box sx={{ mb: 3 }}>
            <MathRenderer math={question.question} />
          </Box>

          {question.type === 'multiple-choice' ? (
            <RadioGroup
              value={userAnswer}
              onChange={(e) => setUserAnswer(e.target.value)}
            >
              {question.options.map((option, index) => (
                <FormControlLabel
                  key={index}
                  value={option}
                  control={<Radio />}
                  label={<MathRenderer math={option} />}
                  disabled={showFeedback}
                />
              ))}
            </RadioGroup>
          ) : (
            <TextField
              fullWidth
              value={userAnswer}
              onChange={(e) => setUserAnswer(e.target.value)}
              placeholder="Enter your answer"
              disabled={showFeedback}
              sx={{ mb: 2 }}
            />
          )}

          {showFeedback && (
            <Alert 
              severity={quizResults.answers[currentQuestion]?.isCorrect ? 'success' : 'error'}
              sx={{ mt: 2 }}
            >
              <Typography variant="subtitle2" gutterBottom>
                {quizResults.answers[currentQuestion]?.isCorrect ? 'Correct!' : 'Incorrect'}
              </Typography>
              <MathRenderer math={question.explanation} />
            </Alert>
          )}

          <Box sx={{ mt: 3, display: 'flex', justifyContent: 'space-between' }}>
            <Button variant="outlined" onClick={resetQuiz}>
              Exit Quiz
            </Button>
            
            {!showFeedback ? (
              <Button 
                variant="contained" 
                onClick={handleSubmitAnswer}
                disabled={!userAnswer.trim()}
              >
                Submit Answer
              </Button>
            ) : (
              <Button variant="contained" onClick={nextQuestion}>
                {currentQuestion < currentQuiz.questions.length - 1 ? 'Next Question' : 'View Results'}
              </Button>
            )}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}

export default QuizPage;