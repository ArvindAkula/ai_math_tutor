import React, { useState, useEffect } from 'react';
import {
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Avatar,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
  useTheme,
  useMediaQuery
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  School as SchoolIcon,
  EmojiEvents as TrophyIcon,
  LocalFireDepartment as StreakIcon,
  Functions as FunctionsIcon,
  Calculate as CalculateIcon,
  Timeline as TimelineIcon,
  Star as StarIcon
} from '@mui/icons-material';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { useGuestLimitations } from '../contexts/GuestLimitationContext';
import { useAuth } from '../contexts/AuthContext';

function ProgressDashboard() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [userStats, setUserStats] = useState(null);

  // Guest limitation hooks
  const { trackUsage, showPrompt } = useGuestLimitations();
  const { isAuthenticated } = useAuth();

  useEffect(() => {
    // Track usage for guest users and check limits
    if (!isAuthenticated) {
      trackUsage('PROGRESS');
      
      // Check if limit is exceeded and show prompt
      const limitExceeded = showPrompt('PROGRESS');
      if (limitExceeded) {
        // The prompt will be shown by the context, but we can still continue
        // This allows users to see the prompt but still use the feature
      }
    }
  }, [isAuthenticated, trackUsage, showPrompt]);

  useEffect(() => {
    // Mock user progress data - in real implementation, this would come from API
    const mockStats = {
      user: {
        name: "Student",
        level: 15,
        totalXP: 2450,
        currentStreak: 7,
        longestStreak: 12,
        problemsSolved: 156,
        averageAccuracy: 78
      },
      skillLevels: {
        algebra: 85,
        calculus: 62,
        linearAlgebra: 45,
        statistics: 71,
        aiMlMath: 38
      },
      recentActivity: [
        { date: '2024-01-15', problemsSolved: 8, accuracy: 87 },
        { date: '2024-01-14', problemsSolved: 5, accuracy: 92 },
        { date: '2024-01-13', problemsSolved: 12, accuracy: 75 },
        { date: '2024-01-12', problemsSolved: 6, accuracy: 83 },
        { date: '2024-01-11', problemsSolved: 9, accuracy: 79 },
        { date: '2024-01-10', problemsSolved: 7, accuracy: 88 },
        { date: '2024-01-09', problemsSolved: 4, accuracy: 95 }
      ],
      achievements: [
        { id: 1, title: "First Steps", description: "Solved your first problem", earned: true, icon: "🎯" },
        { id: 2, title: "Streak Master", description: "Maintained a 7-day streak", earned: true, icon: "🔥" },
        { id: 3, title: "Algebra Expert", description: "Reached 80% proficiency in Algebra", earned: true, icon: "📐" },
        { id: 4, title: "Century Club", description: "Solved 100 problems", earned: true, icon: "💯" },
        { id: 5, title: "Calculus Novice", description: "Completed first calculus problem", earned: true, icon: "∫" },
        { id: 6, title: "Perfect Score", description: "Get 100% accuracy on a quiz", earned: false, icon: "⭐" }
      ],
      recommendations: [
        { topic: "Quadratic Equations", reason: "Based on recent struggles", priority: "high" },
        { topic: "Integration by Parts", reason: "Next logical step in calculus", priority: "medium" },
        { topic: "Matrix Operations", reason: "Foundation for AI/ML math", priority: "low" }
      ]
    };
    
    setUserStats(mockStats);
  }, []);

  if (!userStats) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <LinearProgress sx={{ width: '50%' }} />
      </Box>
    );
  }

  const skillData = Object.entries(userStats.skillLevels).map(([skill, level]) => ({
    name: skill.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase()),
    level: level,
    color: getSkillColor(skill)
  }));

  function getSkillColor(skill) {
    const colors = {
      algebra: '#1976d2',
      calculus: '#dc004e',
      linearAlgebra: '#2e7d32',
      statistics: '#ed6c02',
      aiMlMath: '#9c27b0'
    };
    return colors[skill] || '#757575';
  }

  const COLORS = ['#1976d2', '#dc004e', '#2e7d32', '#ed6c02', '#9c27b0'];

  return (
    <Box>
      <Typography variant="h2" component="h1" gutterBottom>
        Progress Dashboard
      </Typography>

      {/* User Overview Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Avatar sx={{ bgcolor: 'primary.main', mx: 'auto', mb: 1 }}>
                <SchoolIcon />
              </Avatar>
              <Typography variant="h4" component="div">
                {userStats.user.level}
              </Typography>
              <Typography color="text.secondary">
                Current Level
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Avatar sx={{ bgcolor: 'secondary.main', mx: 'auto', mb: 1 }}>
                <StreakIcon />
              </Avatar>
              <Typography variant="h4" component="div">
                {userStats.user.currentStreak}
              </Typography>
              <Typography color="text.secondary">
                Day Streak
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Avatar sx={{ bgcolor: 'success.main', mx: 'auto', mb: 1 }}>
                <CalculateIcon />
              </Avatar>
              <Typography variant="h4" component="div">
                {userStats.user.problemsSolved}
              </Typography>
              <Typography color="text.secondary">
                Problems Solved
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Avatar sx={{ bgcolor: 'warning.main', mx: 'auto', mb: 1 }}>
                <TrendingUpIcon />
              </Avatar>
              <Typography variant="h4" component="div">
                {userStats.user.averageAccuracy}%
              </Typography>
              <Typography color="text.secondary">
                Avg. Accuracy
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Skill Levels */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Skill Levels
              </Typography>
              {skillData.map((skill, index) => (
                <Box key={skill.name} sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">{skill.name}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      {skill.level}%
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={skill.level}
                    sx={{
                      height: 8,
                      borderRadius: 4,
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: skill.color,
                      },
                    }}
                  />
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Activity Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Recent Activity
              </Typography>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={userStats.recentActivity}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Line 
                    type="monotone" 
                    dataKey="problemsSolved" 
                    stroke="#1976d2" 
                    strokeWidth={2}
                    name="Problems Solved"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="accuracy" 
                    stroke="#dc004e" 
                    strokeWidth={2}
                    name="Accuracy %"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Achievements */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Achievements
              </Typography>
              <Grid container spacing={1}>
                {userStats.achievements.map((achievement) => (
                  <Grid item xs={6} sm={4} key={achievement.id}>
                    <Paper
                      sx={{
                        p: 2,
                        textAlign: 'center',
                        opacity: achievement.earned ? 1 : 0.5,
                        bgcolor: achievement.earned ? 'background.paper' : 'grey.100',
                        border: achievement.earned ? `2px solid ${theme.palette.primary.main}` : '2px solid transparent',
                      }}
                    >
                      <Typography variant="h4" component="div" sx={{ mb: 1 }}>
                        {achievement.icon}
                      </Typography>
                      <Typography variant="subtitle2" gutterBottom>
                        {achievement.title}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {achievement.description}
                      </Typography>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Learning Recommendations */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Recommended Topics
              </Typography>
              <List>
                {userStats.recommendations.map((rec, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <FunctionsIcon color={rec.priority === 'high' ? 'error' : rec.priority === 'medium' ? 'warning' : 'action'} />
                    </ListItemIcon>
                    <ListItemText
                      primary={rec.topic}
                      secondary={rec.reason}
                    />
                    <Chip
                      label={rec.priority}
                      size="small"
                      color={rec.priority === 'high' ? 'error' : rec.priority === 'medium' ? 'warning' : 'default'}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Skill Distribution Pie Chart */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Skill Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={skillData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, level }) => `${name}: ${level}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="level"
                  >
                    {skillData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default ProgressDashboard;