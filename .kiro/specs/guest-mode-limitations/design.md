# Guest Mode Limitations - Design Document

## Overview

The guest mode limitations feature implements a progressive engagement strategy that allows new users to explore the platform while encouraging registration through smart prompts and feature limitations. The system tracks usage patterns and presents contextual registration prompts at optimal moments.

## Architecture

### Component Structure

```
GuestLimitationProvider (Context)
├── useGuestLimitations (Hook)
├── GuestPromptModal (Component)
└── withGuestLimitation (HOC)
```

### Data Flow

1. **Usage Tracking**: Each feature interaction increments counters in localStorage
2. **Limit Checking**: Before feature access, check if limits are exceeded
3. **Prompt Display**: Show contextual registration prompts when limits are hit
4. **Graceful Continuation**: Allow continued usage with periodic reminders

## Components and Interfaces

### 1. GuestLimitationProvider (Context)

```javascript
const GuestLimitationContext = createContext({
  trackUsage: (feature) => {},
  checkLimit: (feature) => boolean,
  showPrompt: (feature, onContinue) => {},
  resetLimits: () => {},
  isLimitExceeded: (feature) => boolean
});

const FEATURE_LIMITS = {
  PROBLEM_SOLVER: 2,
  QUIZ: 2, 
  PROGRESS: 1,
  VISUALIZATION: 3
};
```

### 2. useGuestLimitations Hook

```javascript
const useGuestLimitations = () => {
  const trackUsage = (feature) => {
    // Increment counter in localStorage
    // Check if authenticated user (skip tracking)
    // Update usage statistics
  };
  
  const checkLimit = (feature) => {
    // Return true if limit exceeded
    // Return false if user is authenticated
    // Return false if under limit
  };
  
  const showPrompt = (feature, onContinue) => {
    // Display modal with feature-specific messaging
    // Handle user choice (signup/signin/continue)
    // Track prompt interactions
  };
  
  return { trackUsage, checkLimit, showPrompt, resetLimits };
};
```

### 3. GuestPromptModal Component

```javascript
const GuestPromptModal = ({ 
  open, 
  feature, 
  onSignUp, 
  onSignIn, 
  onContinue, 
  onClose 
}) => {
  const getFeatureBenefits = (feature) => {
    switch(feature) {
      case 'PROBLEM_SOLVER':
        return [
          'Save your problem-solving history',
          'Get personalized hints and explanations',
          'Track your learning progress',
          'Access unlimited problem solving'
        ];
      case 'QUIZ':
        return [
          'Save your quiz results and progress',
          'Get adaptive difficulty adjustment',
          'Detailed performance analytics',
          'Unlimited quiz attempts'
        ];
      case 'PROGRESS':
        return [
          'Comprehensive learning analytics',
          'Personalized learning recommendations',
          'Achievement tracking and badges',
          'Long-term progress history'
        ];
    }
  };
  
  return (
    <Modal open={open} onClose={onClose}>
      <Card>
        <CardContent>
          <Typography variant="h5">
            Unlock Your Full Learning Potential! 🚀
          </Typography>
          <Typography variant="body1">
            You've been exploring our {getFeatureName(feature)} feature. 
            Create a free account to unlock these benefits:
          </Typography>
          <List>
            {getFeatureBenefits(feature).map(benefit => (
              <ListItem key={benefit}>
                <ListItemIcon><CheckIcon /></ListItemIcon>
                <ListItemText primary={benefit} />
              </ListItem>
            ))}
          </List>
          <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
            <Button variant="contained" onClick={onSignUp}>
              Sign Up Free
            </Button>
            <Button variant="outlined" onClick={onSignIn}>
              Sign In
            </Button>
            <Button variant="text" onClick={onContinue}>
              Continue as Guest
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Modal>
  );
};
```

### 4. withGuestLimitation HOC

```javascript
const withGuestLimitation = (WrappedComponent, feature) => {
  return (props) => {
    const { trackUsage, checkLimit, showPrompt } = useGuestLimitations();
    const { isAuthenticated } = useAuth();
    const [showLimitPrompt, setShowLimitPrompt] = useState(false);
    
    const handleFeatureAccess = () => {
      if (!isAuthenticated) {
        trackUsage(feature);
        if (checkLimit(feature)) {
          setShowLimitPrompt(true);
          return false; // Block access temporarily
        }
      }
      return true; // Allow access
    };
    
    const handleContinueAsGuest = () => {
      setShowLimitPrompt(false);
      // Allow continued usage but track for future prompts
    };
    
    return (
      <>
        <WrappedComponent 
          {...props} 
          onFeatureAccess={handleFeatureAccess}
        />
        <GuestPromptModal 
          open={showLimitPrompt}
          feature={feature}
          onContinue={handleContinueAsGuest}
          onClose={() => setShowLimitPrompt(false)}
        />
      </>
    );
  };
};
```

## Data Models

### Usage Tracking Storage

```javascript
// localStorage structure
const guestUsageData = {
  problemSolver: {
    count: 0,
    lastUsed: timestamp,
    promptsShown: 0
  },
  quiz: {
    count: 0,
    lastUsed: timestamp,
    promptsShown: 0
  },
  progress: {
    count: 0,
    lastUsed: timestamp,
    promptsShown: 0
  },
  sessionStart: timestamp,
  totalInteractions: 0
};
```

### Feature Configuration

```javascript
const FEATURE_CONFIG = {
  PROBLEM_SOLVER: {
    limit: 2,
    name: 'Problem Solver',
    icon: 'calculate',
    description: 'step-by-step math solutions'
  },
  QUIZ: {
    limit: 2,
    name: 'Quiz System',
    icon: 'quiz',
    description: 'interactive math quizzes'
  },
  PROGRESS: {
    limit: 1,
    name: 'Progress Dashboard',
    icon: 'trending_up',
    description: 'learning analytics'
  }
};
```

## Implementation Strategy

### Phase 1: Core Infrastructure
1. Create GuestLimitationProvider context
2. Implement useGuestLimitations hook
3. Add localStorage usage tracking
4. Create basic limit checking logic

### Phase 2: UI Components
1. Design and implement GuestPromptModal
2. Create feature-specific messaging
3. Add smooth animations and transitions
4. Implement responsive design

### Phase 3: Integration
1. Integrate with existing components (ProblemSolver, QuizPage, ProgressDashboard)
2. Add tracking to key user interactions
3. Test limit enforcement and prompt display
4. Ensure proper cleanup on authentication

### Phase 4: Enhancement
1. Add analytics tracking for prompt effectiveness
2. Implement A/B testing for different messaging
3. Add progressive disclosure of benefits
4. Optimize conversion rates

## User Experience Flow

### First-Time User Journey

1. **Initial Access**: User can freely use any feature
2. **Second Usage**: Normal usage, counter incremented
3. **Third Usage**: Prompt appears with registration benefits
4. **User Choice**:
   - **Sign Up**: Redirect to registration, reset all limits
   - **Sign In**: Redirect to login, reset all limits  
   - **Continue**: Allow usage, show periodic reminders

### Prompt Messaging Strategy

#### Problem Solver Prompt
```
🧮 Ready to Supercharge Your Math Learning?

You've solved a few problems! Create a free account to:
✓ Save your problem-solving history
✓ Get personalized hints and explanations  
✓ Track your learning progress
✓ Access unlimited problem solving

[Sign Up Free] [Sign In] [Continue as Guest]
```

#### Quiz Prompt
```
🎯 Take Your Math Skills to the Next Level!

You've tried our quizzes! Create a free account to:
✓ Save your quiz results and progress
✓ Get adaptive difficulty adjustment
✓ View detailed performance analytics
✓ Take unlimited quizzes

[Sign Up Free] [Sign In] [Continue as Guest]
```

#### Progress Dashboard Prompt
```
📊 Unlock Your Complete Learning Journey!

Ready to see your full progress? Create a free account to:
✓ View comprehensive learning analytics
✓ Get personalized learning recommendations
✓ Earn achievements and badges
✓ Access your complete learning history

[Sign Up Free] [Sign In] [Continue as Guest]
```

## Technical Considerations

### Performance
- Lightweight localStorage operations
- Minimal impact on component rendering
- Efficient limit checking algorithms
- Lazy loading of prompt components

### Privacy
- No personal data stored in localStorage
- Anonymous usage tracking only
- Clear data cleanup on registration
- Respect user privacy preferences

### Accessibility
- Screen reader compatible prompts
- Keyboard navigation support
- High contrast mode compatibility
- Clear focus management

### Testing Strategy
- Unit tests for usage tracking logic
- Integration tests for prompt display
- E2E tests for complete user flows
- A/B testing for conversion optimization