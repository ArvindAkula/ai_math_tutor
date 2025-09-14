import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Typography,
  Button,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Box,
  Card,
  CardContent,
  Divider,
  useTheme,
  useMediaQuery
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import StarIcon from '@mui/icons-material/Star';
import { useNavigate } from 'react-router-dom';
import { useGuestLimitations } from '../contexts/GuestLimitationContext';

const GuestPromptModal = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const navigate = useNavigate();
  const { currentPrompt, getFeatureConfig } = useGuestLimitations();
  const [open, setOpen] = useState(false);

  // Update open state when currentPrompt changes
  React.useEffect(() => {
    setOpen(!!currentPrompt);
  }, [currentPrompt]);

  if (!currentPrompt) return null;

  const { feature, onContinue, onSignUp, onSignIn } = currentPrompt;
  const config = getFeatureConfig(feature);

  const handleClose = () => {
    setOpen(false);
  };

  const handleSignUp = () => {
    navigate('/register');
    if (onSignUp) onSignUp();
    handleClose();
  };

  const handleSignIn = () => {
    navigate('/login');
    if (onSignIn) onSignIn();
    handleClose();
  };

  const handleContinue = () => {
    if (onContinue) onContinue();
    handleClose();
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      maxWidth="sm"
      fullWidth
      fullScreen={isMobile}
      PaperProps={{
        sx: {
          borderRadius: isMobile ? 0 : 2,
          m: isMobile ? 0 : 2
        }
      }}
    >
      <DialogTitle sx={{ textAlign: 'center', pb: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
          <StarIcon sx={{ color: 'primary.main', fontSize: 32, mr: 1 }} />
          <Typography variant="h5" component="span" sx={{ fontWeight: 'bold' }}>
            Unlock Your Full Learning Potential!
          </Typography>
        </Box>
        <Typography variant="body2" color="text.secondary">
          {config.icon} Ready to supercharge your {config.description}?
        </Typography>
      </DialogTitle>

      <DialogContent sx={{ pt: 1 }}>
        <Card variant="outlined" sx={{ mb: 3, bgcolor: 'primary.50' }}>
          <CardContent sx={{ py: 2 }}>
            <Typography variant="body1" sx={{ mb: 2, textAlign: 'center' }}>
              You've been exploring our <strong>{config.name}</strong> feature. 
              Create a free account to unlock these amazing benefits:
            </Typography>
          </CardContent>
        </Card>

        <List sx={{ py: 0 }}>
          {config.benefits.map((benefit, index) => (
            <ListItem key={index} sx={{ py: 0.5 }}>
              <ListItemIcon sx={{ minWidth: 36 }}>
                <CheckCircleIcon sx={{ color: 'success.main', fontSize: 20 }} />
              </ListItemIcon>
              <ListItemText 
                primary={benefit}
                primaryTypographyProps={{
                  variant: 'body2',
                  sx: { fontWeight: 500 }
                }}
              />
            </ListItem>
          ))}
        </List>

        <Divider sx={{ my: 2 }} />

        <Box sx={{ textAlign: 'center', py: 1 }}>
          <Typography variant="body2" color="text.secondary">
            Join thousands of students already learning with AI Math Tutor
          </Typography>
        </Box>
      </DialogContent>

      <DialogActions sx={{ p: 3, pt: 1, flexDirection: isMobile ? 'column' : 'row', gap: 1 }}>
        <Button
          variant="contained"
          size="large"
          onClick={handleSignUp}
          fullWidth={isMobile}
          sx={{ 
            minWidth: isMobile ? '100%' : 120,
            fontWeight: 'bold',
            textTransform: 'none'
          }}
        >
          Sign Up Free
        </Button>
        
        <Button
          variant="outlined"
          size="large"
          onClick={handleSignIn}
          fullWidth={isMobile}
          sx={{ 
            minWidth: isMobile ? '100%' : 100,
            textTransform: 'none'
          }}
        >
          Sign In
        </Button>
        
        <Button
          variant="text"
          size="large"
          onClick={handleContinue}
          fullWidth={isMobile}
          sx={{ 
            minWidth: isMobile ? '100%' : 140,
            textTransform: 'none',
            color: 'text.secondary'
          }}
        >
          Continue as Guest
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default GuestPromptModal;