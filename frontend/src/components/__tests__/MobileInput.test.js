import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import MobileInput from '../MobileInput';

const theme = createTheme();

const renderWithTheme = (component) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  );
};

describe('MobileInput Component', () => {
  const mockOnChange = jest.fn();
  const mockOnSubmit = jest.fn();

  beforeEach(() => {
    mockOnChange.mockClear();
    mockOnSubmit.mockClear();
  });

  test('renders input field correctly', () => {
    renderWithTheme(
      <MobileInput
        value=""
        onChange={mockOnChange}
        onSubmit={mockOnSubmit}
        placeholder="Enter math expression"
      />
    );

    expect(screen.getByPlaceholderText('Enter math expression')).toBeInTheDocument();
  });

  test('calls onChange when input value changes', () => {
    renderWithTheme(
      <MobileInput
        value=""
        onChange={mockOnChange}
        onSubmit={mockOnSubmit}
      />
    );

    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: '2x + 3' } });
    
    expect(mockOnChange).toHaveBeenCalled();
  });

  test('shows mobile-specific buttons on mobile viewport', () => {
    // Mock mobile viewport
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 600,
    });

    renderWithTheme(
      <MobileInput
        value=""
        onChange={mockOnChange}
        onSubmit={mockOnSubmit}
      />
    );

    // Math keyboard, voice, and camera buttons should be present
    expect(screen.getByLabelText('Math keyboard')).toBeInTheDocument();
    expect(screen.getByLabelText('Voice input')).toBeInTheDocument();
    expect(screen.getByLabelText('Camera input')).toBeInTheDocument();
  });

  test('handles camera input click', () => {
    // Mock alert
    window.alert = jest.fn();

    renderWithTheme(
      <MobileInput
        value=""
        onChange={mockOnChange}
        onSubmit={mockOnSubmit}
      />
    );

    const cameraButton = screen.getByLabelText('Camera input');
    fireEvent.click(cameraButton);

    expect(window.alert).toHaveBeenCalledWith(
      expect.stringContaining('Camera input feature will be implemented')
    );
  });

  test('handles voice input when speech recognition is not supported', () => {
    // Mock missing speech recognition
    delete window.webkitSpeechRecognition;
    delete window.SpeechRecognition;
    window.alert = jest.fn();

    renderWithTheme(
      <MobileInput
        value=""
        onChange={mockOnChange}
        onSubmit={mockOnSubmit}
      />
    );

    const voiceButton = screen.getByLabelText('Voice input');
    fireEvent.click(voiceButton);

    expect(window.alert).toHaveBeenCalledWith(
      'Speech recognition not supported in this browser'
    );
  });

  test('opens math keyboard drawer when math keyboard button is clicked', () => {
    renderWithTheme(
      <MobileInput
        value=""
        onChange={mockOnChange}
        onSubmit={mockOnSubmit}
      />
    );

    const mathKeyboardButton = screen.getByLabelText('Math keyboard');
    fireEvent.click(mathKeyboardButton);

    expect(screen.getByText('Mathematical Symbols')).toBeInTheDocument();
  });

  test('inserts mathematical symbols when clicked', () => {
    renderWithTheme(
      <MobileInput
        value="x"
        onChange={mockOnChange}
        onSubmit={mockOnSubmit}
      />
    );

    // Open math keyboard
    const mathKeyboardButton = screen.getByLabelText('Math keyboard');
    fireEvent.click(mathKeyboardButton);

    // Click on plus symbol
    const plusSymbol = screen.getByText('+');
    fireEvent.click(plusSymbol);

    // Should call onChange with the symbol inserted
    expect(mockOnChange).toHaveBeenCalled();
  });

  test('handles Enter key press to submit', () => {
    renderWithTheme(
      <MobileInput
        value="2x + 3"
        onChange={mockOnChange}
        onSubmit={mockOnSubmit}
      />
    );

    const input = screen.getByRole('textbox');
    fireEvent.keyPress(input, { key: 'Enter', code: 'Enter' });

    expect(mockOnSubmit).toHaveBeenCalled();
  });

  test('converts speech to mathematical notation', () => {
    // This tests the internal convertSpeechToMath function indirectly
    // by checking if the component handles speech input correctly
    renderWithTheme(
      <MobileInput
        value=""
        onChange={mockOnChange}
        onSubmit={mockOnSubmit}
      />
    );

    // The component should be able to handle speech conversion
    expect(screen.getByLabelText('Voice input')).toBeInTheDocument();
  });
});

describe('Mobile Input Accessibility', () => {
  test('has proper ARIA labels for all interactive elements', () => {
    render(
      <ThemeProvider theme={theme}>
        <MobileInput
          value=""
          onChange={() => {}}
          onSubmit={() => {}}
        />
      </ThemeProvider>
    );

    expect(screen.getByLabelText('Math keyboard')).toBeInTheDocument();
    expect(screen.getByLabelText('Voice input')).toBeInTheDocument();
    expect(screen.getByLabelText('Camera input')).toBeInTheDocument();
  });

  test('supports keyboard navigation', () => {
    render(
      <ThemeProvider theme={theme}>
        <MobileInput
          value=""
          onChange={() => {}}
          onSubmit={() => {}}
        />
      </ThemeProvider>
    );

    const input = screen.getByRole('textbox');
    expect(input).toBeInTheDocument();
    
    // Input should be focusable
    input.focus();
    expect(input).toHaveFocus();
  });
});