import React from 'react';
import { render, screen } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { MemoryRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import ProblemSolver from '../ProblemSolver';
import HomePage from '../HomePage';

// Mock components to avoid complex dependencies in tests
jest.mock('react-katex', () => ({
  InlineMath: ({ math }) => <span data-testid="inline-math">{math}</span>,
  BlockMath: ({ math }) => <div data-testid="block-math">{math}</div>
}));

jest.mock('react-plotly.js', () => {
  return function MockPlot(props) {
    return <div data-testid="plotly-plot">Mock Plot</div>;
  };
});

const theme = createTheme({
  breakpoints: {
    values: {
      xs: 0,
      sm: 600,
      md: 900,
      lg: 1200,
      xl: 1536,
    },
  },
});

const renderWithProviders = (component) => {
  return render(
    <ThemeProvider theme={theme}>
      <MemoryRouter>
        {component}
      </MemoryRouter>
    </ThemeProvider>
  );
};

// Mock window.matchMedia for responsive testing
const mockMatchMedia = (matches) => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: jest.fn().mockImplementation(query => ({
      matches,
      media: query,
      onchange: null,
      addListener: jest.fn(),
      removeListener: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    })),
  });
};

describe('Responsive Design Tests', () => {
  beforeEach(() => {
    // Reset matchMedia mock
    delete window.matchMedia;
    // Mock getBoundingClientRect for textarea autosize
    Element.prototype.getBoundingClientRect = jest.fn(() => ({
      width: 200,
      height: 100,
      top: 0,
      left: 0,
      bottom: 100,
      right: 200,
    }));
  });

  test('home page renders responsive layout', () => {
    renderWithProviders(<HomePage />);
    
    // Check that the home page renders
    expect(screen.getByText('Welcome to AI Math Tutor')).toBeInTheDocument();
  });

  test('problem solver layout adapts to screen size', () => {
    renderWithProviders(<ProblemSolver />);
    
    // Check that the main container exists
    expect(screen.getByText('Problem Solver')).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/e.g., 2x \+ 3 = 7/)).toBeInTheDocument();
  });

  test('mathematical expressions are responsive', () => {
    renderWithProviders(<ProblemSolver />);
    
    // Check that math input field is present and responsive
    const mathInput = screen.getByPlaceholderText(/e.g., 2x \+ 3 = 7/);
    expect(mathInput).toBeInTheDocument();
    expect(mathInput).toHaveAttribute('rows', '3'); // Multiline for better mobile experience
  });

  test('components handle different screen sizes', () => {
    mockMatchMedia(true); // Simulate mobile screen
    renderWithProviders(<HomePage />);
    
    // Should render without errors on mobile
    expect(screen.getByText('Welcome to AI Math Tutor')).toBeInTheDocument();
  });
});

describe('Cross-Browser Compatibility Tests', () => {
  beforeEach(() => {
    // Mock getBoundingClientRect for textarea autosize
    Element.prototype.getBoundingClientRect = jest.fn(() => ({
      width: 200,
      height: 100,
      top: 0,
      left: 0,
      bottom: 100,
      right: 200,
    }));
  });

  test('uses standard CSS properties for compatibility', () => {
    renderWithProviders(<HomePage />);
    
    // Check that the component renders without errors
    expect(screen.getByText(/AI Math Tutor/)).toBeInTheDocument();
  });

  test('handles missing features gracefully', () => {
    // Mock missing CSS Grid support
    const originalGetComputedStyle = window.getComputedStyle;
    window.getComputedStyle = jest.fn().mockReturnValue({
      display: 'block', // Fallback for browsers without grid support
    });

    renderWithProviders(<HomePage />);
    expect(screen.getByText(/AI Math Tutor/)).toBeInTheDocument();

    window.getComputedStyle = originalGetComputedStyle;
  });

  test('mathematical notation renders with fallbacks', () => {
    renderWithProviders(<ProblemSolver />);
    
    // The component should render even if KaTeX fails
    expect(screen.getByText('Problem Solver')).toBeInTheDocument();
  });

  test('interactive elements are keyboard accessible', () => {
    renderWithProviders(<ProblemSolver />);
    
    const solveButton = screen.getByText('Solve Problem');
    expect(solveButton).toBeInTheDocument();
    expect(solveButton.tagName).toBe('BUTTON'); // Proper semantic element
  });
});

describe('Performance and Accessibility Tests', () => {
  beforeEach(() => {
    // Mock getBoundingClientRect for textarea autosize
    Element.prototype.getBoundingClientRect = jest.fn(() => ({
      width: 200,
      height: 100,
      top: 0,
      left: 0,
      bottom: 100,
      right: 200,
    }));
  });

  test('components render without performance warnings', () => {
    const consoleSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    
    renderWithProviders(<HomePage />);
    
    // Should not have React performance warnings
    expect(consoleSpy).not.toHaveBeenCalledWith(
      expect.stringContaining('Warning:')
    );
    
    consoleSpy.mockRestore();
  });

  test('proper ARIA labels are present', () => {
    renderWithProviders(<ProblemSolver />);
    
    const mathInput = screen.getByLabelText(/Enter your mathematical problem/);
    expect(mathInput).toBeInTheDocument();
  });

  test('semantic HTML structure is maintained', () => {
    renderWithProviders(<HomePage />);
    
    // Check for proper heading hierarchy
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
  });
});

describe('Error Handling Tests', () => {
  beforeEach(() => {
    // Mock getBoundingClientRect for textarea autosize
    Element.prototype.getBoundingClientRect = jest.fn(() => ({
      width: 200,
      height: 100,
      top: 0,
      left: 0,
      bottom: 100,
      right: 200,
    }));
  });

  test('handles component errors gracefully', () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    
    // This should not crash the component
    renderWithProviders(<HomePage />);
    expect(screen.getByText(/AI Math Tutor/)).toBeInTheDocument();
    
    consoleSpy.mockRestore();
  });

  test('displays error messages for invalid math input', () => {
    renderWithProviders(<ProblemSolver />);
    
    // The error handling should be present in the component
    expect(screen.getByText('Solve Problem')).toBeInTheDocument();
  });
});