import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { 
  Box, 
  Typography, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem,
  Slider,
  Paper,
  Grid
} from '@mui/material';

/**
 * InteractiveVisualization component for mathematical plots
 * Supports 2D functions, vectors, and basic 3D surfaces
 */
function InteractiveVisualization({ 
  plotData, 
  title = "Mathematical Visualization",
  interactive = true 
}) {
  const [plotType, setPlotType] = useState('2d');
  const [xRange, setXRange] = useState([-10, 10]);
  const [yRange, setYRange] = useState([-10, 10]);

  // Default plot configuration
  const defaultLayout = {
    title: title,
    autosize: true,
    margin: { l: 50, r: 50, t: 50, b: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Arial, sans-serif', size: 12 },
    xaxis: {
      title: 'x',
      range: xRange,
      gridcolor: '#e0e0e0',
      zerolinecolor: '#666'
    },
    yaxis: {
      title: 'y', 
      range: yRange,
      gridcolor: '#e0e0e0',
      zerolinecolor: '#666'
    }
  };

  const config = {
    responsive: true,
    displayModeBar: interactive,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
    displaylogo: false
  };

  // Generate sample function plot if no data provided
  const generateSamplePlot = () => {
    const x = [];
    const y = [];
    for (let i = xRange[0]; i <= xRange[1]; i += 0.1) {
      x.push(i);
      y.push(Math.sin(i) * Math.cos(i * 0.5));
    }
    
    return [{
      x: x,
      y: y,
      type: 'scatter',
      mode: 'lines',
      name: 'f(x) = sin(x) * cos(0.5x)',
      line: { color: '#1976d2', width: 2 }
    }];
  };

  // Generate vector field visualization
  const generateVectorField = () => {
    const x = [];
    const y = [];
    const u = [];
    const v = [];
    
    for (let i = -5; i <= 5; i += 1) {
      for (let j = -5; j <= 5; j += 1) {
        x.push(i);
        y.push(j);
        u.push(-j); // Vector field: (-y, x)
        v.push(i);
      }
    }
    
    return [{
      x: x,
      y: y,
      u: u,
      v: v,
      type: 'cone',
      colorscale: 'Viridis',
      name: 'Vector Field'
    }];
  };

  // Generate 3D surface plot
  const generate3DSurface = () => {
    const size = 50;
    const x = [];
    const y = [];
    const z = [];
    
    for (let i = 0; i < size; i++) {
      x.push([]);
      y.push([]);
      z.push([]);
      for (let j = 0; j < size; j++) {
        const xVal = (i - size/2) * 0.5;
        const yVal = (j - size/2) * 0.5;
        x[i].push(xVal);
        y[i].push(yVal);
        z[i].push(Math.sin(Math.sqrt(xVal*xVal + yVal*yVal)));
      }
    }
    
    return [{
      x: x,
      y: y,
      z: z,
      type: 'surface',
      colorscale: 'Viridis',
      name: 'f(x,y) = sin(√(x² + y²))'
    }];
  };

  const getPlotData = () => {
    if (plotData) return plotData;
    
    switch (plotType) {
      case '2d':
        return generateSamplePlot();
      case 'vector':
        return generateVectorField();
      case '3d':
        return generate3DSurface();
      default:
        return generateSamplePlot();
    }
  };

  const getLayout = () => {
    if (plotType === '3d') {
      return {
        ...defaultLayout,
        scene: {
          xaxis: { title: 'x' },
          yaxis: { title: 'y' },
          zaxis: { title: 'z' },
          camera: {
            eye: { x: 1.5, y: 1.5, z: 1.5 }
          }
        }
      };
    }
    return defaultLayout;
  };

  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      {interactive && (
        <Box sx={{ mb: 2 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} sm={4}>
              <FormControl fullWidth size="small">
                <InputLabel>Plot Type</InputLabel>
                <Select
                  value={plotType}
                  label="Plot Type"
                  onChange={(e) => setPlotType(e.target.value)}
                >
                  <MenuItem value="2d">2D Function</MenuItem>
                  <MenuItem value="vector">Vector Field</MenuItem>
                  <MenuItem value="3d">3D Surface</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            {plotType === '2d' && (
              <>
                <Grid item xs={12} sm={4}>
                  <Typography gutterBottom>X Range</Typography>
                  <Slider
                    value={xRange}
                    onChange={(e, newValue) => setXRange(newValue)}
                    valueLabelDisplay="auto"
                    min={-20}
                    max={20}
                    step={1}
                  />
                </Grid>
                <Grid item xs={12} sm={4}>
                  <Typography gutterBottom>Y Range</Typography>
                  <Slider
                    value={yRange}
                    onChange={(e, newValue) => setYRange(newValue)}
                    valueLabelDisplay="auto"
                    min={-20}
                    max={20}
                    step={1}
                  />
                </Grid>
              </>
            )}
          </Grid>
        </Box>
      )}
      
      <Box sx={{ height: 400, width: '100%' }}>
        <Plot
          data={getPlotData()}
          layout={getLayout()}
          config={config}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler={true}
        />
      </Box>
    </Paper>
  );
}

export default InteractiveVisualization;