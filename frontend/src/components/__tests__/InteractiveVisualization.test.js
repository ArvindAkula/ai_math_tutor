import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import InteractiveVisualization from '../InteractiveVisualization';

// Mock Plotly component
jest.mock('react-plotly.js', () => {
  return function MockPlot({ data, layout, config, ...props }) {
    return (
      <div 
        data-testid="plotly-plot"
        data-plot-type={data[0]?.type || 'unknown'}
        data-title={layout?.title || 'No title'}
        {...props}
      >
        Mock Plot Component
      </div>
    );
  };
});

describe('InteractiveVisualization Component', () => {
  test('renders with default title', () => {
    render(<InteractiveVisualization />);
    expect(screen.getByTestId('plotly-plot')).toBeInTheDocument();
    expect(screen.getByTestId('plotly-plot')).toHaveAttribute('data-title', 'Mathematical Visualization');
  });

  test('renders with custom title', () => {
    const customTitle = 'Custom Plot Title';
    render(<InteractiveVisualization title={customTitle} />);
    expect(screen.getByTestId('plotly-plot')).toHaveAttribute('data-title', customTitle);
  });

  test('renders with provided plot data', () => {
    const plotData = [{
      x: [1, 2, 3],
      y: [1, 4, 9],
      type: 'scatter',
      mode: 'lines'
    }];
    
    render(<InteractiveVisualization plotData={plotData} />);
    expect(screen.getByTestId('plotly-plot')).toHaveAttribute('data-plot-type', 'scatter');
  });

  test('shows interactive controls when interactive=true', () => {
    render(<InteractiveVisualization interactive={true} />);
    expect(screen.getByRole('combobox')).toBeInTheDocument();
    expect(screen.getAllByText('Plot Type')[0]).toBeInTheDocument();
  });

  test('hides interactive controls when interactive=false', () => {
    render(<InteractiveVisualization interactive={false} />);
    expect(screen.queryByText('Plot Type')).not.toBeInTheDocument();
  });

  test('shows range sliders for 2D plots', () => {
    render(<InteractiveVisualization interactive={true} />);
    
    expect(screen.getByText('X Range')).toBeInTheDocument();
    expect(screen.getByText('Y Range')).toBeInTheDocument();
  });

  test('renders plot type selector', () => {
    render(<InteractiveVisualization interactive={true} />);
    
    const plotTypeSelect = screen.getByRole('combobox');
    expect(plotTypeSelect).toBeInTheDocument();
    expect(plotTypeSelect).toHaveTextContent('2D Function');
  });

  test('renders default 2D scatter plot', () => {
    render(<InteractiveVisualization interactive={true} />);
    
    expect(screen.getByTestId('plotly-plot')).toHaveAttribute('data-plot-type', 'scatter');
  });

  test('renders responsive plot container', () => {
    render(<InteractiveVisualization />);
    const plotContainer = screen.getByTestId('plotly-plot').parentElement;
    expect(plotContainer).toHaveStyle({ height: '400px', width: '100%' });
  });
});