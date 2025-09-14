import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import MathRenderer from '../MathRenderer';

// Mock KaTeX components
jest.mock('react-katex', () => ({
  InlineMath: ({ math }) => <span data-testid="inline-math">{math}</span>,
  BlockMath: ({ math }) => <div data-testid="block-math">{math}</div>
}));

describe('MathRenderer Component', () => {
  test('renders inline math correctly', () => {
    render(<MathRenderer math="x^2 + 1" />);
    const mathElement = screen.getByTestId('inline-math');
    expect(mathElement).toBeInTheDocument();
    expect(mathElement).toHaveTextContent('x^2 + 1');
  });

  test('renders block math correctly', () => {
    render(<MathRenderer math="\\frac{x}{y}" block={true} />);
    const mathElement = screen.getByTestId('block-math');
    expect(mathElement).toBeInTheDocument();
    expect(mathElement).toHaveTextContent('\\frac{x}{y}');
  });

  test('applies custom className', () => {
    render(<MathRenderer math="x + y" className="custom-class" />);
    const mathContainer = screen.getByTestId('inline-math').parentElement;
    expect(mathContainer).toHaveClass('math-inline');
    expect(mathContainer).toHaveClass('custom-class');
  });

  test('handles empty math input', () => {
    render(<MathRenderer math="" />);
    expect(screen.queryByTestId('inline-math')).not.toBeInTheDocument();
    expect(screen.queryByTestId('block-math')).not.toBeInTheDocument();
  });

  test('handles null math input', () => {
    render(<MathRenderer math={null} />);
    expect(screen.queryByTestId('inline-math')).not.toBeInTheDocument();
    expect(screen.queryByTestId('block-math')).not.toBeInTheDocument();
  });

  test('displays error message for invalid LaTeX', () => {
    // Mock console.error to avoid test output pollution
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    
    // Mock KaTeX to throw an error
    jest.doMock('react-katex', () => ({
      InlineMath: () => { throw new Error('Invalid LaTeX'); },
      BlockMath: () => { throw new Error('Invalid LaTeX'); }
    }));

    render(<MathRenderer math="\\invalid{latex}" />);
    expect(screen.getByText('[Math Error: \\invalid{latex}]')).toBeInTheDocument();
    
    consoleSpy.mockRestore();
  });
});