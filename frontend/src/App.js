import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { 
  Container, 
  AppBar, 
  Toolbar, 
  Typography, 
  Box, 
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  useMediaQuery,
  Divider,
  Button,
  Menu,
  MenuItem,
  Avatar,
  CircularProgress
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import HomeIcon from '@mui/icons-material/Home';
import CalculateIcon from '@mui/icons-material/Calculate';
import QuizIcon from '@mui/icons-material/Quiz';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import LoginIcon from '@mui/icons-material/Login';
import LogoutIcon from '@mui/icons-material/Logout';

// Import components
import HomePage from './components/HomePage';
import ProblemSolver from './components/ProblemSolver';
import QuizPage from './components/QuizPage';
import ProgressDashboard from './components/ProgressDashboard';
import Login from './components/Login';
import Register from './components/Register';

// Import auth context
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { GuestLimitationProvider } from './contexts/GuestLimitationContext';
import GuestPromptModal from './components/GuestPromptModal';

// Create Material-UI theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
  },
});

function NavigationDrawer({ open, onClose, onNavigate }) {
  const menuItems = [
    { text: 'Home', icon: <HomeIcon />, path: '/' },
    { text: 'Problem Solver', icon: <CalculateIcon />, path: '/solve' },
    { text: 'Quizzes', icon: <QuizIcon />, path: '/quiz' },
    { text: 'Progress', icon: <TrendingUpIcon />, path: '/progress' }
  ];

  return (
    <Drawer anchor="left" open={open} onClose={onClose}>
      <Box sx={{ width: 250 }} role="presentation">
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" component="div">
            🧮 AI Math Tutor
          </Typography>
        </Box>
        <Divider />
        <List>
          {menuItems.map((item) => (
            <ListItem key={item.text} disablePadding>
              <ListItemButton onClick={() => onNavigate(item.path)}>
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>
    </Drawer>
  );
}

function AppContent() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [authMode, setAuthMode] = useState('login'); // 'login' or 'register'
  const [userMenuAnchor, setUserMenuAnchor] = useState(null);
  const navigate = useNavigate();
  const location = useLocation();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const { user, loading, login, logout, register, isAuthenticated } = useAuth();

  const handleDrawerToggle = () => {
    setDrawerOpen(!drawerOpen);
  };

  const handleNavigate = (path) => {
    navigate(path);
    setDrawerOpen(false);
  };

  const handleUserMenuOpen = (event) => {
    setUserMenuAnchor(event.currentTarget);
  };

  const handleUserMenuClose = () => {
    setUserMenuAnchor(null);
  };

  const handleLogout = async () => {
    await logout();
    handleUserMenuClose();
    navigate('/');
  };

  const handleLogin = (userData, token) => {
    login(userData, token);
    navigate('/');
  };

  const handleRegister = (userData, token) => {
    register(userData, token);
    navigate('/');
  };

  const getPageTitle = () => {
    switch (location.pathname) {
      case '/': return 'Home';
      case '/solve': return 'Problem Solver';
      case '/quiz': return 'Quizzes';
      case '/progress': return 'Progress Dashboard';
      case '/login': return 'Sign In';
      case '/register': return 'Sign Up';
      default: return 'AI Math Tutor';
    }
  };

  if (loading) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '100vh'
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  // Show auth pages if not authenticated and on auth routes
  if (!isAuthenticated && (location.pathname === '/login' || location.pathname === '/register')) {
    return (
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              🧮 AI Math Tutor - {getPageTitle()}
            </Typography>
          </Toolbar>
        </AppBar>
        
        <Routes>
          <Route 
            path="/login" 
            element={
              <Login 
                onLogin={handleLogin}
                onSwitchToRegister={() => navigate('/register')}
              />
            } 
          />
          <Route 
            path="/register" 
            element={
              <Register 
                onRegister={handleRegister}
                onSwitchToLogin={() => navigate('/login')}
              />
            } 
          />
        </Routes>
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          {isMobile && (
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}
          <Typography 
            variant="h6" 
            component="div" 
            sx={{ 
              flexGrow: 1, 
              cursor: 'pointer',
              '&:hover': {
                opacity: 0.8
              }
            }}
            onClick={() => navigate('/')}
          >
            🧮 AI Math Tutor {isMobile && `- ${getPageTitle()}`}
          </Typography>
          
          {/* User Authentication Section */}
          {isAuthenticated ? (
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Button
                color="inherit"
                onClick={handleUserMenuOpen}
                startIcon={<AccountCircleIcon />}
              >
                {user?.username || 'User'}
              </Button>
              <Menu
                anchorEl={userMenuAnchor}
                open={Boolean(userMenuAnchor)}
                onClose={handleUserMenuClose}
              >
                <MenuItem onClick={() => { handleUserMenuClose(); navigate('/progress'); }}>
                  <TrendingUpIcon sx={{ mr: 1 }} />
                  My Progress
                </MenuItem>
                <MenuItem onClick={handleLogout}>
                  <LogoutIcon sx={{ mr: 1 }} />
                  Logout
                </MenuItem>
              </Menu>
            </Box>
          ) : (
            <Box>
              <Button
                color="inherit"
                onClick={() => navigate('/login')}
                startIcon={<LoginIcon />}
                sx={{ mr: 1 }}
              >
                Sign In
              </Button>
              <Button
                color="inherit"
                variant="outlined"
                onClick={() => navigate('/register')}
                sx={{ borderColor: 'white', color: 'white' }}
              >
                Sign Up
              </Button>
            </Box>
          )}
        </Toolbar>
      </AppBar>
      
      <NavigationDrawer 
        open={drawerOpen} 
        onClose={() => setDrawerOpen(false)}
        onNavigate={handleNavigate}
      />
      
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/solve" element={<ProblemSolver />} />
          <Route path="/quiz" element={<QuizPage />} />
          <Route path="/progress" element={<ProgressDashboard />} />
          <Route 
            path="/login" 
            element={
              <Login 
                onLogin={handleLogin}
                onSwitchToRegister={() => navigate('/register')}
              />
            } 
          />
          <Route 
            path="/register" 
            element={
              <Register 
                onRegister={handleRegister}
                onSwitchToLogin={() => navigate('/login')}
              />
            } 
          />
        </Routes>
      </Container>
    </Box>
  );
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <AuthProvider>
          <GuestLimitationProvider>
            <AppContent />
            <GuestPromptModal />
          </GuestLimitationProvider>
        </AuthProvider>
      </Router>
    </ThemeProvider>
  );
}

export default App;