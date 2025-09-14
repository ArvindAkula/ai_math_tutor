import React from 'react';
import { 
  Typography, 
  Grid, 
  Card, 
  CardContent, 
  CardActions, 
  Button,
  Box,
  useMediaQuery,
  useTheme
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import CalculateIcon from '@mui/icons-material/Calculate';
import QuizIcon from '@mui/icons-material/Quiz';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import MathRenderer from './MathRenderer';
import InteractiveVisualization from './InteractiveVisualization';

function HomePage() {
  const navigate = useNavigate();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const features = [
    {
      title: 'Problem Solver',
      description: 'Get step-by-step solutions with detailed explanations for any math problem.',
      icon: <CalculateIcon sx={{ fontSize: 40 }} />,
      path: '/solve',
      color: 'primary'
    },
    {
      title: 'Interactive Quizzes',
      description: 'Practice with adaptive quizzes that adjust to your skill level.',
      icon: <QuizIcon sx={{ fontSize: 40 }} />,
      path: '/quiz',
      color: 'secondary'
    },
    {
      title: 'Progress Tracking',
      description: 'Monitor your learning journey and get personalized recommendations.',
      icon: <TrendingUpIcon sx={{ fontSize: 40 }} />,
      path: '/progress',
      color: 'success'
    }
  ];

  return (
    <Box>
      <Typography variant="h1" component="h1" gutterBottom align="center">
        Welcome to AI Math Tutor
      </Typography>
      
      <Typography variant="h5" component="h2" gutterBottom align="center" color="text.secondary" sx={{ mb: 4 }}>
        Your intelligent companion for mastering mathematics
      </Typography>

      <Grid container spacing={4}>
        {features.map((feature, index) => (
          <Grid item xs={12} md={4} key={index}>
            <Card 
              sx={{ 
                height: '100%', 
                display: 'flex', 
                flexDirection: 'column',
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: 4
                }
              }}
            >
              <CardContent sx={{ flexGrow: 1, textAlign: 'center' }}>
                <Box sx={{ color: `${feature.color}.main`, mb: 2 }}>
                  {feature.icon}
                </Box>
                <Typography variant="h5" component="h3" gutterBottom>
                  {feature.title}
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  {feature.description}
                </Typography>
              </CardContent>
              <CardActions sx={{ justifyContent: 'center', pb: 2 }}>
                <Button 
                  variant="contained" 
                  color={feature.color}
                  onClick={() => navigate(feature.path)}
                  size="large"
                >
                  Get Started
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Mathematical Examples Section */}
      <Box sx={{ mt: 6, p: 3, bgcolor: 'background.paper', borderRadius: 2 }}>
        <Typography variant="h4" component="h2" gutterBottom align="center">
          Solve Complex Mathematical Problems
        </Typography>
        <Typography variant="body1" align="center" color="text.secondary" sx={{ mb: 4 }}>
          From basic algebra to advanced calculus and AI/ML mathematics
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom color="primary">
                Algebra & Equations
              </Typography>
              <Box sx={{ mb: 2 }}>
                <MathRenderer math="2x + 3 = 7" block />
                <MathRenderer math="x^2 - 5x + 6 = 0" block />
                <MathRenderer math="\sqrt{x^2 + y^2} = r" block />
              </Box>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom color="secondary">
                Calculus & Analysis
              </Typography>
              <Box sx={{ mb: 2 }}>
                <MathRenderer math="\frac{d}{dx}(x^2 + 3x) = 2x + 3" block />
                <MathRenderer math="\int_0^1 x^2 \, dx = \frac{1}{3}" block />
                <MathRenderer math="\lim_{x \to 0} \frac{\sin x}{x} = 1" block />
              </Box>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom sx={{ color: 'success.main' }}>
                Linear Algebra
              </Typography>
              <Box sx={{ mb: 2 }}>
                <MathRenderer math="\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}" block />
                <MathRenderer math="Ax = \lambda x" block />
              </Box>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom sx={{ color: 'warning.main' }}>
                AI/ML Mathematics
              </Typography>
              <Box sx={{ mb: 2 }}>
                <MathRenderer math="\nabla f(x) = \frac{\partial f}{\partial x}" block />
                <MathRenderer math="J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2" block />
              </Box>
            </Card>
          </Grid>
        </Grid>
      </Box>

      {/* Interactive Visualization Preview */}
      {!isMobile && (
        <Box sx={{ mt: 6 }}>
          <Typography variant="h4" component="h2" gutterBottom align="center">
            Interactive Mathematical Visualizations
          </Typography>
          <Typography variant="body1" align="center" color="text.secondary" sx={{ mb: 4 }}>
            Explore mathematical concepts through interactive plots and graphs
          </Typography>
          <InteractiveVisualization 
            title="Sample Function: f(x) = sin(x) × cos(0.5x)"
            interactive={true}
          />
        </Box>
      )}

      <Box sx={{ mt: 6, p: 3, bgcolor: 'background.paper', borderRadius: 2 }}>
        <Typography variant="h4" component="h2" gutterBottom align="center">
          Features Coming Soon
        </Typography>
        <Grid container spacing={2} sx={{ mt: 2 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="h6">📝 Voice Input</Typography>
            <Typography variant="body2" color="text.secondary">
              Solve problems by speaking them aloud
            </Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="h6">✍️ Handwriting Recognition</Typography>
            <Typography variant="body2" color="text.secondary">
              Write equations naturally and get instant solutions
            </Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="h6">🤖 AI/ML Mathematics</Typography>
            <Typography variant="body2" color="text.secondary">
              Specialized tools for machine learning math
            </Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="h6">📊 Advanced Visualizations</Typography>
            <Typography variant="body2" color="text.secondary">
              Interactive 3D plots and animations
            </Typography>
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
}

export default HomePage;