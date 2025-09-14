import React, { useState, useRef } from 'react';
import {
  Box,
  TextField,
  IconButton,
  Grid,
  Paper,
  Typography,
  Fab,
  useTheme,
  useMediaQuery,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemText
} from '@mui/material';
import {
  Functions as FunctionsIcon,
  CameraAlt as CameraIcon,
  Mic as MicIcon,
  Keyboard as KeyboardIcon,
  Close as CloseIcon
} from '@mui/icons-material';

/**
 * Mobile-optimized input component for mathematical expressions
 * Includes touch-friendly keyboard, voice input, and camera features
 */
function MobileInput({ value, onChange, onSubmit, placeholder = "Enter mathematical expression" }) {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [showMathKeyboard, setShowMathKeyboard] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const inputRef = useRef(null);

  // Mathematical symbols for touch keyboard
  const mathSymbols = [
    { symbol: '+', label: 'Plus' },
    { symbol: '-', label: 'Minus' },
    { symbol: '×', label: 'Multiply', value: '*' },
    { symbol: '÷', label: 'Divide', value: '/' },
    { symbol: '=', label: 'Equals' },
    { symbol: '(', label: 'Left Parenthesis' },
    { symbol: ')', label: 'Right Parenthesis' },
    { symbol: '^', label: 'Power' },
    { symbol: '√', label: 'Square Root', value: 'sqrt(' },
    { symbol: '∫', label: 'Integral', value: 'integral(' },
    { symbol: '∑', label: 'Sum', value: 'sum(' },
    { symbol: 'π', label: 'Pi', value: 'pi' },
    { symbol: '∞', label: 'Infinity', value: 'infinity' },
    { symbol: 'sin', label: 'Sine', value: 'sin(' },
    { symbol: 'cos', label: 'Cosine', value: 'cos(' },
    { symbol: 'tan', label: 'Tangent', value: 'tan(' },
    { symbol: 'log', label: 'Logarithm', value: 'log(' },
    { symbol: 'ln', label: 'Natural Log', value: 'ln(' },
    { symbol: 'e', label: 'Euler Number', value: 'e' },
    { symbol: 'x', label: 'Variable x' },
    { symbol: 'y', label: 'Variable y' },
    { symbol: 'α', label: 'Alpha', value: 'alpha' },
    { symbol: 'β', label: 'Beta', value: 'beta' },
    { symbol: 'θ', label: 'Theta', value: 'theta' }
  ];

  const insertSymbol = (symbol, insertValue) => {
    const input = inputRef.current;
    if (input) {
      const start = input.selectionStart;
      const end = input.selectionEnd;
      const newValue = value.substring(0, start) + (insertValue || symbol) + value.substring(end);
      onChange({ target: { value: newValue } });
      
      // Set cursor position after inserted symbol
      setTimeout(() => {
        const newPosition = start + (insertValue || symbol).length;
        input.setSelectionRange(newPosition, newPosition);
        input.focus();
      }, 0);
    }
  };

  const handleVoiceInput = () => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      alert('Speech recognition not supported in this browser');
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      // Convert speech to mathematical notation
      const mathText = convertSpeechToMath(transcript);
      onChange({ target: { value: value + ' ' + mathText } });
      setIsListening(false);
    };

    recognition.onerror = () => {
      setIsListening(false);
      alert('Speech recognition error. Please try again.');
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognition.start();
  };

  const convertSpeechToMath = (speech) => {
    // Basic speech-to-math conversion
    return speech
      .replace(/\bplus\b/gi, '+')
      .replace(/\bminus\b/gi, '-')
      .replace(/\btimes\b/gi, '*')
      .replace(/\bmultiplied by\b/gi, '*')
      .replace(/\bdivided by\b/gi, '/')
      .replace(/\bover\b/gi, '/')
      .replace(/\bsquared\b/gi, '^2')
      .replace(/\bcubed\b/gi, '^3')
      .replace(/\bto the power of\b/gi, '^')
      .replace(/\bsquare root of\b/gi, 'sqrt(')
      .replace(/\bsine of\b/gi, 'sin(')
      .replace(/\bcosine of\b/gi, 'cos(')
      .replace(/\btangent of\b/gi, 'tan(')
      .replace(/\bpi\b/gi, 'pi')
      .replace(/\bequals\b/gi, '=');
  };

  const handleCameraInput = () => {
    // Placeholder for camera/OCR functionality
    // In a real implementation, this would integrate with an OCR service
    alert('Camera input feature will be implemented with OCR integration in upcoming tasks');
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      onSubmit();
    }
  };

  return (
    <Box>
      <TextField
        inputRef={inputRef}
        fullWidth
        multiline
        rows={isMobile ? 2 : 3}
        value={value}
        onChange={onChange}
        onKeyPress={handleKeyPress}
        placeholder={placeholder}
        variant="outlined"
        sx={{
          mb: 2,
          '& .MuiOutlinedInput-root': {
            fontSize: isMobile ? '16px' : '14px', // Prevent zoom on iOS
            '& fieldset': {
              borderWidth: 2,
            },
          },
        }}
        InputProps={{
          style: {
            fontSize: isMobile ? '16px' : '14px',
            lineHeight: 1.5,
          },
        }}
      />

      {isMobile && (
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1, mb: 2 }}>
          <Fab
            size="small"
            color="primary"
            onClick={() => setShowMathKeyboard(true)}
            aria-label="Math keyboard"
          >
            <FunctionsIcon />
          </Fab>
          
          <Fab
            size="small"
            color="secondary"
            onClick={handleVoiceInput}
            disabled={isListening}
            aria-label="Voice input"
            sx={{
              backgroundColor: isListening ? theme.palette.error.main : theme.palette.secondary.main,
            }}
          >
            <MicIcon />
          </Fab>
          
          <Fab
            size="small"
            color="info"
            onClick={handleCameraInput}
            aria-label="Camera input"
          >
            <CameraIcon />
          </Fab>
        </Box>
      )}

      {/* Mathematical Keyboard Drawer */}
      <Drawer
        anchor="bottom"
        open={showMathKeyboard}
        onClose={() => setShowMathKeyboard(false)}
        PaperProps={{
          sx: {
            maxHeight: '50vh',
            borderTopLeftRadius: 16,
            borderTopRightRadius: 16,
          },
        }}
      >
        <Box sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Mathematical Symbols</Typography>
            <IconButton onClick={() => setShowMathKeyboard(false)}>
              <CloseIcon />
            </IconButton>
          </Box>
          
          <Grid container spacing={1}>
            {mathSymbols.map((item, index) => (
              <Grid item xs={3} sm={2} key={index}>
                <Paper
                  elevation={1}
                  sx={{
                    p: 1,
                    textAlign: 'center',
                    cursor: 'pointer',
                    minHeight: 48,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    '&:hover': {
                      backgroundColor: theme.palette.action.hover,
                    },
                    '&:active': {
                      backgroundColor: theme.palette.action.selected,
                    },
                  }}
                  onClick={() => insertSymbol(item.symbol, item.value)}
                >
                  <Typography variant="h6" component="span">
                    {item.symbol}
                  </Typography>
                </Paper>
              </Grid>
            ))}
          </Grid>
          
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
            <Typography variant="caption" color="text.secondary">
              Tap symbols to insert into your expression
            </Typography>
          </Box>
        </Box>
      </Drawer>
    </Box>
  );
}

export default MobileInput;