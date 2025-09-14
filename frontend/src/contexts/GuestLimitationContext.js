import React, { createContext, useContext, useState, useEffect } from 'react';
import { useAuth } from './AuthContext';

const GuestLimitationContext = createContext({
  trackUsage: () => {},
  checkLimit: () => false,
  showPrompt: () => {},
  resetLimits: () => {},
  isLimitExceeded: () => false,
  getUsageCount: () => 0
});

// Feature limits configuration
const FEATURE_LIMITS = {
  PROBLEM_SOLVER: 2,
  QUIZ: 2,
  PROGRESS: 1,
  VISUALIZATION: 3
};

// Feature display names and benefits
const FEATURE_CONFIG = {
  PROBLEM_SOLVER: {
    name: 'Problem Solver',
    icon: '🧮',
    description: 'step-by-step math solutions',
    benefits: [
      'Save your problem-solving history',
      'Get personalized hints and explanations',
      'Track your learning progress',
      'Access unlimited problem solving'
    ]
  },
  QUIZ: {
    name: 'Quiz System',
    icon: '🎯',
    description: 'interactive math quizzes',
    benefits: [
      'Save your quiz results and progress',
      'Get adaptive difficulty adjustment',
      'View detailed performance analytics',
      'Take unlimited quizzes'
    ]
  },
  PROGRESS: {
    name: 'Progress Dashboard',
    icon: '📊',
    description: 'learning analytics',
    benefits: [
      'View comprehensive learning analytics',
      'Get personalized learning recommendations',
      'Earn achievements and badges',
      'Access your complete learning history'
    ]
  }
};

const STORAGE_KEY = 'guestUsageData';

export const GuestLimitationProvider = ({ children }) => {
  const { isAuthenticated } = useAuth();
  const [usageData, setUsageData] = useState({});
  const [currentPrompt, setCurrentPrompt] = useState(null);

  // Load usage data from localStorage on mount
  useEffect(() => {
    const savedData = localStorage.getItem(STORAGE_KEY);
    if (savedData) {
      try {
        setUsageData(JSON.parse(savedData));
      } catch (error) {
        console.error('Error loading guest usage data:', error);
        setUsageData({});
      }
    }
  }, []);

  // Save usage data to localStorage whenever it changes
  useEffect(() => {
    if (Object.keys(usageData).length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(usageData));
    }
  }, [usageData]);

  // Reset limits when user authenticates
  useEffect(() => {
    if (isAuthenticated) {
      resetLimits();
    }
  }, [isAuthenticated]);

  const trackUsage = (feature) => {
    // Don't track usage for authenticated users
    if (isAuthenticated) return;

    const now = Date.now();
    setUsageData(prev => ({
      ...prev,
      [feature]: {
        count: (prev[feature]?.count || 0) + 1,
        lastUsed: now,
        promptsShown: prev[feature]?.promptsShown || 0
      },
      sessionStart: prev.sessionStart || now,
      totalInteractions: (prev.totalInteractions || 0) + 1
    }));
  };

  const getUsageCount = (feature) => {
    return usageData[feature]?.count || 0;
  };

  const isLimitExceeded = (feature) => {
    if (isAuthenticated) return false;
    const count = getUsageCount(feature);
    const limit = FEATURE_LIMITS[feature] || 0;
    return count > limit;
  };

  const checkLimit = (feature) => {
    return isLimitExceeded(feature);
  };

  const showPrompt = (feature, onContinue, onSignUp, onSignIn) => {
    if (isAuthenticated) return false;

    const shouldShow = isLimitExceeded(feature);
    if (shouldShow) {
      setCurrentPrompt({
        feature,
        onContinue: () => {
          setCurrentPrompt(null);
          if (onContinue) onContinue();
        },
        onSignUp: () => {
          setCurrentPrompt(null);
          if (onSignUp) onSignUp();
        },
        onSignIn: () => {
          setCurrentPrompt(null);
          if (onSignIn) onSignIn();
        },
        onClose: () => {
          setCurrentPrompt(null);
        }
      });

      // Track that we showed a prompt
      setUsageData(prev => ({
        ...prev,
        [feature]: {
          ...prev[feature],
          promptsShown: (prev[feature]?.promptsShown || 0) + 1
        }
      }));
    }

    return shouldShow;
  };

  const resetLimits = () => {
    setUsageData({});
    localStorage.removeItem(STORAGE_KEY);
    setCurrentPrompt(null);
  };

  const getFeatureConfig = (feature) => {
    return FEATURE_CONFIG[feature] || {
      name: feature,
      icon: '🔧',
      description: 'feature',
      benefits: ['Unlimited access to this feature']
    };
  };

  const value = {
    trackUsage,
    checkLimit,
    showPrompt,
    resetLimits,
    isLimitExceeded,
    getUsageCount,
    getFeatureConfig,
    currentPrompt,
    usageData
  };

  return (
    <GuestLimitationContext.Provider value={value}>
      {children}
    </GuestLimitationContext.Provider>
  );
};

export const useGuestLimitations = () => {
  const context = useContext(GuestLimitationContext);
  if (!context) {
    throw new Error('useGuestLimitations must be used within a GuestLimitationProvider');
  }
  return context;
};

export default GuestLimitationContext;