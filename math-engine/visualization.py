"""
Mathematical visualization engine using Matplotlib.
Generates 2D plots for mathematical functions, solutions, and interactive elements.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Dict, Any, Optional, Tuple, Union
import io
import base64
import sys
import os

# Add shared models to path
from models import (
    PlotData, Point, PlotStyle, InteractiveElement, AnimationData,
    ParsedProblem, StepSolution
)

import sympy as sp
from sympy import symbols, lambdify, sympify


class MathPlotter:
    """Main plotting engine for mathematical visualizations."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), dpi: int = 100):
        """Initialize the plotter with default settings."""
        self.figsize = figsize
        self.dpi = dpi
        self.default_style = PlotStyle(
            color='blue',
            line_width=2.0,
            marker_style=None,
            transparency=1.0
        )
        
        # Set matplotlib style for better-looking plots
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    def create_function_plot(self, 
                           expression: str, 
                           variable: str = 'x',
                           x_range: Tuple[float, float] = (-10, 10),
                           num_points: int = 1000,
                           style: Optional[PlotStyle] = None,
                           title: str = "",
                           show_grid: bool = True) -> PlotData:
        """
        Create a 2D plot of a mathematical function.
        
        Args:
            expression: Mathematical expression as string (e.g., 'x**2 + 2*x + 1')
            variable: Variable name (default: 'x')
            x_range: Range of x values to plot
            num_points: Number of points to generate
            style: Plot styling options
            title: Plot title
            show_grid: Whether to show grid
            
        Returns:
            PlotData: Structured plot data
        """
        if style is None:
            style = self.default_style
        
        try:
            # Parse the expression using SymPy
            expr = sympify(expression)
            var_symbol = symbols(variable)
            
            # Create numerical function
            func = lambdify(var_symbol, expr, 'numpy')
            
            # Generate x values
            x_vals = np.linspace(x_range[0], x_range[1], num_points)
            
            # Calculate y values, handling potential errors
            y_vals = []
            valid_points = []
            
            for x in x_vals:
                try:
                    y = func(x)
                    if np.isfinite(y):  # Check for NaN, inf, etc.
                        y_vals.append(float(y))
                        valid_points.append(Point(x=float(x), y=float(y)))
                    else:
                        y_vals.append(np.nan)
                except:
                    y_vals.append(np.nan)
            
            # Create plot data structure
            plot_data = PlotData(
                plot_type='function',
                data_points=valid_points,
                styling=style,
                interactive_elements=[],
                title=title or f"Graph of {expression}",
                axis_labels={'x': variable, 'y': f'f({variable})'}
            )
            
            return plot_data
            
        except Exception as e:
            raise ValueError(f"Error creating function plot: {str(e)}")
    
    def create_derivative_plot(self, 
                             function_expr: str,
                             derivative_expr: str,
                             variable: str = 'x',
                             x_range: Tuple[float, float] = (-5, 5),
                             highlight_point: Optional[float] = None) -> PlotData:
        """
        Create a plot showing both a function and its derivative.
        
        Args:
            function_expr: Original function expression
            derivative_expr: Derivative expression
            variable: Variable name
            x_range: Range of x values
            highlight_point: Point to highlight with tangent line
            
        Returns:
            PlotData: Plot data with function and derivative
        """
        try:
            # Parse expressions
            func_expr = sympify(function_expr)
            deriv_expr = sympify(derivative_expr)
            var_symbol = symbols(variable)
            
            # Create numerical functions
            func = lambdify(var_symbol, func_expr, 'numpy')
            deriv_func = lambdify(var_symbol, deriv_expr, 'numpy')
            
            # Generate points
            x_vals = np.linspace(x_range[0], x_range[1], 500)
            
            data_points = []
            interactive_elements = []
            
            # Function points
            for x in x_vals:
                try:
                    y = func(x)
                    if np.isfinite(y):
                        data_points.append(Point(x=float(x), y=float(y)))
                except:
                    continue
            
            # Add tangent line at highlight point if specified
            if highlight_point is not None:
                try:
                    y_point = func(highlight_point)
                    slope = deriv_func(highlight_point)
                    
                    # Create tangent line points
                    tangent_x = np.linspace(highlight_point - 2, highlight_point + 2, 100)
                    tangent_y = slope * (tangent_x - highlight_point) + y_point
                    
                    # Add interactive element for the tangent point
                    interactive_elements.append(InteractiveElement(
                        element_type='point',
                        position=Point(x=float(highlight_point), y=float(y_point)),
                        action='show_tangent',
                        tooltip=f'Tangent at x={highlight_point}, slope={slope:.3f}'
                    ))
                    
                except:
                    pass
            
            plot_data = PlotData(
                plot_type='derivative',
                data_points=data_points,
                styling=PlotStyle(color='blue', line_width=2.0),
                interactive_elements=interactive_elements,
                title=f"Function and Derivative: f(x) = {function_expr}",
                axis_labels={'x': variable, 'y': 'y'}
            )
            
            return plot_data
            
        except Exception as e:
            raise ValueError(f"Error creating derivative plot: {str(e)}")
    
    def create_integral_plot(self,
                           function_expr: str,
                           variable: str = 'x',
                           x_range: Tuple[float, float] = (-5, 5),
                           integral_bounds: Optional[Tuple[float, float]] = None,
                           show_area: bool = True) -> PlotData:
        """
        Create a plot for integral visualization with area shading.
        
        Args:
            function_expr: Function to integrate
            variable: Variable name
            x_range: Range for plotting
            integral_bounds: Bounds for definite integral (a, b)
            show_area: Whether to shade the area under curve
            
        Returns:
            PlotData: Plot data with integral visualization
        """
        try:
            # Parse expression
            expr = sympify(function_expr)
            var_symbol = symbols(variable)
            func = lambdify(var_symbol, expr, 'numpy')
            
            # Generate function points
            x_vals = np.linspace(x_range[0], x_range[1], 500)
            data_points = []
            
            for x in x_vals:
                try:
                    y = func(x)
                    if np.isfinite(y):
                        data_points.append(Point(x=float(x), y=float(y)))
                except:
                    continue
            
            interactive_elements = []
            
            # Add area shading for definite integral
            if integral_bounds and show_area:
                a, b = integral_bounds
                area_x = np.linspace(a, b, 200)
                
                # Add interactive element for the shaded area
                interactive_elements.append(InteractiveElement(
                    element_type='area',
                    position=Point(x=float((a + b) / 2), y=0),
                    action='show_area',
                    tooltip=f'Integral from {a} to {b}'
                ))
            
            title = f"Integral of {function_expr}"
            if integral_bounds:
                title += f" from {integral_bounds[0]} to {integral_bounds[1]}"
            
            plot_data = PlotData(
                plot_type='integral',
                data_points=data_points,
                styling=PlotStyle(color='green', line_width=2.0),
                interactive_elements=interactive_elements,
                title=title,
                axis_labels={'x': variable, 'y': f'f({variable})'}
            )
            
            return plot_data
            
        except Exception as e:
            raise ValueError(f"Error creating integral plot: {str(e)}")
    
    def create_optimization_plot(self,
                               function_expr: str,
                               variable: str = 'x',
                               x_range: Tuple[float, float] = (-5, 5),
                               critical_points: Optional[List[float]] = None,
                               extrema_type: Optional[List[str]] = None) -> PlotData:
        """
        Create a plot for optimization problems showing critical points.
        
        Args:
            function_expr: Function to optimize
            variable: Variable name
            x_range: Range for plotting
            critical_points: List of critical point x-values
            extrema_type: List of extrema types ('min', 'max', 'inflection')
            
        Returns:
            PlotData: Plot data with optimization visualization
        """
        try:
            # Parse expression
            expr = sympify(function_expr)
            var_symbol = symbols(variable)
            func = lambdify(var_symbol, expr, 'numpy')
            
            # Generate function points
            x_vals = np.linspace(x_range[0], x_range[1], 500)
            data_points = []
            
            for x in x_vals:
                try:
                    y = func(x)
                    if np.isfinite(y):
                        data_points.append(Point(x=float(x), y=float(y)))
                except:
                    continue
            
            interactive_elements = []
            
            # Add critical points as interactive elements
            if critical_points:
                for i, cp in enumerate(critical_points):
                    try:
                        y_val = func(cp)
                        extrema = extrema_type[i] if extrema_type and i < len(extrema_type) else 'critical'
                        
                        interactive_elements.append(InteractiveElement(
                            element_type='critical_point',
                            position=Point(x=float(cp), y=float(y_val)),
                            action='highlight_extremum',
                            tooltip=f'{extrema.title()} at x={cp:.3f}, y={y_val:.3f}'
                        ))
                    except:
                        continue
            
            plot_data = PlotData(
                plot_type='optimization',
                data_points=data_points,
                styling=PlotStyle(color='red', line_width=2.0),
                interactive_elements=interactive_elements,
                title=f"Optimization: f(x) = {function_expr}",
                axis_labels={'x': variable, 'y': f'f({variable})'}
            )
            
            return plot_data
            
        except Exception as e:
            raise ValueError(f"Error creating optimization plot: {str(e)}")
    
    def render_plot(self, plot_data: PlotData, save_path: Optional[str] = None) -> str:
        """
        Render a PlotData object to a matplotlib figure.
        
        Args:
            plot_data: Plot data to render
            save_path: Optional path to save the plot
            
        Returns:
            Base64 encoded image string for web display
        """
        try:
            # Handle 3D plots
            if plot_data.plot_type in ['vector_3d', 'surface_3d']:
                fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
                ax = fig.add_subplot(111, projection='3d')
                
                if plot_data.plot_type == 'vector_3d':
                    self._render_3d_vectors(ax, plot_data)
                elif plot_data.plot_type == 'surface_3d':
                    self._render_3d_surface(ax, plot_data)
                    
            else:
                # 2D plots
                fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
                
                if plot_data.plot_type == 'vector_2d':
                    self._render_2d_vectors(ax, plot_data)
                elif plot_data.plot_type == 'vector_field':
                    self._render_vector_field(ax, plot_data)
                elif plot_data.plot_type == 'matrix_transformation':
                    self._render_matrix_transformation(ax, plot_data)
                elif plot_data.plot_type == 'gradient_field':
                    self._render_gradient_field(ax, plot_data)
                elif plot_data.plot_type == 'neural_network':
                    self._render_neural_network(ax, plot_data)
                elif plot_data.plot_type == 'optimization_path':
                    self._render_optimization_path(ax, plot_data)
                elif plot_data.plot_type == 'loss_surface':
                    # Handle 3D loss surface
                    fig.clear()
                    ax = fig.add_subplot(111, projection='3d')
                    self._render_loss_surface(ax, plot_data)
                else:
                    # Standard function plots
                    self._render_standard_plot(ax, plot_data)
            
            # Set labels and title
            ax.set_xlabel(plot_data.axis_labels.get('x', 'x'))
            ax.set_ylabel(plot_data.axis_labels.get('y', 'y'))
            if plot_data.plot_type in ['vector_3d', 'surface_3d']:
                ax.set_zlabel(plot_data.axis_labels.get('z', 'z'))
            ax.set_title(plot_data.title)
            
            if plot_data.plot_type not in ['vector_3d', 'surface_3d']:
                ax.grid(True, alpha=0.3)
            
            # Save or return as base64
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
            # Convert to base64 for web display
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            plt.close(fig)
            return image_base64
            
        except Exception as e:
            plt.close('all')  # Clean up any open figures
            raise ValueError(f"Error rendering plot: {str(e)}")
    
    def _render_standard_plot(self, ax, plot_data: PlotData):
        """Render standard 2D function plots."""
        if plot_data.data_points:
            x_coords = [p.x for p in plot_data.data_points]
            y_coords = [p.y for p in plot_data.data_points]
            
            # Plot the main function
            ax.plot(x_coords, y_coords, 
                   color=plot_data.styling.color,
                   linewidth=plot_data.styling.line_width,
                   alpha=plot_data.styling.transparency,
                   marker=plot_data.styling.marker_style)
        
        # Add interactive elements
        for element in plot_data.interactive_elements:
            if element.element_type == 'point':
                ax.plot(element.position.x, element.position.y, 
                       'ro', markersize=8, zorder=5)
            elif element.element_type == 'critical_point':
                color = 'red' if 'max' in element.tooltip.lower() else 'blue'
                ax.plot(element.position.x, element.position.y, 
                       'o', color=color, markersize=10, zorder=5)
            elif element.element_type == 'area' and plot_data.plot_type == 'integral':
                ax.axvline(element.position.x, color='gray', linestyle='--', alpha=0.5)
    
    def _render_2d_vectors(self, ax, plot_data: PlotData):
        """Render 2D vector plots."""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        if len(plot_data.data_points) > 1:
            origin = plot_data.data_points[0]  # First point is origin
            vectors = plot_data.data_points[1:]  # Rest are vector endpoints
            
            for i, vector_end in enumerate(vectors):
                color = colors[i % len(colors)]
                ax.arrow(origin.x, origin.y, 
                        vector_end.x - origin.x, vector_end.y - origin.y,
                        head_width=0.2, head_length=0.3, fc=color, ec=color,
                        linewidth=2, zorder=5)
                
                # Add vector label
                mid_x = origin.x + (vector_end.x - origin.x) * 0.6
                mid_y = origin.y + (vector_end.y - origin.y) * 0.6
                ax.text(mid_x, mid_y, f'v{i+1}', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        # Set equal aspect ratio for vectors
        ax.set_aspect('equal', adjustable='box')
    
    def _render_3d_vectors(self, ax, plot_data: PlotData):
        """Render 3D vector plots."""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        if len(plot_data.data_points) > 1:
            origin = plot_data.data_points[0]
            vectors = plot_data.data_points[1:]
            
            for i, vector_end in enumerate(vectors):
                color = colors[i % len(colors)]
                ax.quiver(origin.x, origin.y, origin.z or 0,
                         vector_end.x - origin.x, 
                         vector_end.y - origin.y,
                         (vector_end.z or 0) - (origin.z or 0),
                         color=color, linewidth=3, arrow_length_ratio=0.1)
                
                # Add vector label
                ax.text(vector_end.x, vector_end.y, vector_end.z or 0, 
                       f'v{i+1}', fontsize=12)
    
    def _render_vector_field(self, ax, plot_data: PlotData):
        """Render vector field plots."""
        if not plot_data.data_points:
            return
            
        # Extract grid points and vector components
        x_coords = [p.x for p in plot_data.data_points]
        y_coords = [p.y for p in plot_data.data_points]
        u_components = [getattr(p, 'u', 0) for p in plot_data.data_points]
        v_components = [getattr(p, 'v', 0) for p in plot_data.data_points]
        
        # Create quiver plot
        ax.quiver(x_coords, y_coords, u_components, v_components,
                 color=plot_data.styling.color, alpha=0.7, scale=20)
        
        ax.set_aspect('equal', adjustable='box')
    
    def _render_matrix_transformation(self, ax, plot_data: PlotData):
        """Render matrix transformation plots."""
        if not plot_data.data_points:
            return
            
        # Plot transformed points
        x_coords = [p.x for p in plot_data.data_points]
        y_coords = [p.y for p in plot_data.data_points]
        
        ax.scatter(x_coords, y_coords, c=plot_data.styling.color, 
                  alpha=0.6, s=20)
        
        # Add vector pairs from interactive elements
        for element in plot_data.interactive_elements:
            if element.element_type == 'vector_pair':
                # Extract original and transformed coordinates from tooltip
                tooltip = element.tooltip
                if '→' in tooltip:
                    parts = tooltip.split('→')
                    if len(parts) == 2:
                        # Parse coordinates (simplified)
                        orig_str = parts[0].strip().strip('()')
                        trans_str = parts[1].strip().strip('()')
                        try:
                            orig_coords = [float(x.strip()) for x in orig_str.split(',')]
                            trans_coords = [float(x.strip()) for x in trans_str.split(',')]
                            
                            # Draw original vector in light color
                            ax.arrow(0, 0, orig_coords[0], orig_coords[1],
                                   head_width=0.1, head_length=0.15, 
                                   fc='lightblue', ec='lightblue', alpha=0.5)
                            
                            # Draw transformed vector in bold color
                            ax.arrow(0, 0, trans_coords[0], trans_coords[1],
                                   head_width=0.1, head_length=0.15,
                                   fc='red', ec='red', linewidth=2)
                        except:
                            pass
        
        ax.set_aspect('equal', adjustable='box')
    
    def _render_3d_surface(self, ax, plot_data: PlotData):
        """Render 3D surface plots."""
        if not plot_data.data_points:
            return
            
        # Extract coordinates
        x_coords = [p.x for p in plot_data.data_points]
        y_coords = [p.y for p in plot_data.data_points]
        z_coords = [p.z for p in plot_data.data_points if p.z is not None]
        
        if len(z_coords) != len(x_coords):
            return
            
        # Estimate grid size (assuming square grid)
        grid_size = int(np.sqrt(len(x_coords)))
        if grid_size * grid_size != len(x_coords):
            # Fallback to scatter plot if not a perfect grid
            ax.scatter(x_coords, y_coords, z_coords, 
                      c=z_coords, cmap=plot_data.styling.color)
            return
        
        # Reshape to grid
        X = np.array(x_coords).reshape(grid_size, grid_size)
        Y = np.array(y_coords).reshape(grid_size, grid_size)
        Z = np.array(z_coords).reshape(grid_size, grid_size)
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=plot_data.styling.color, 
                              alpha=0.8, linewidth=0.5, antialiased=True)
        
        # Add colorbar
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    def _render_gradient_field(self, ax, plot_data: PlotData):
        """Render gradient field plots with contours and gradient vectors."""
        if not plot_data.data_points:
            return
        
        # Separate contour points from gradient points
        contour_points = [p for p in plot_data.data_points if not hasattr(p, 'is_gradient')]
        gradient_points = [p for p in plot_data.data_points if hasattr(p, 'is_gradient')]
        
        # Plot contours if available
        if contour_points and hasattr(contour_points[0], 'z'):
            x_coords = [p.x for p in contour_points]
            y_coords = [p.y for p in contour_points]
            z_coords = [p.z for p in contour_points]
            
            # Estimate grid size for contour plotting
            unique_x = sorted(list(set(x_coords)))
            unique_y = sorted(list(set(y_coords)))
            
            if len(unique_x) > 1 and len(unique_y) > 1:
                try:
                    X = np.array(x_coords).reshape(len(unique_y), len(unique_x))
                    Y = np.array(y_coords).reshape(len(unique_y), len(unique_x))
                    Z = np.array(z_coords).reshape(len(unique_y), len(unique_x))
                    
                    # Create contour plot
                    contour = ax.contour(X, Y, Z, levels=15, colors='gray', alpha=0.5, linewidths=0.5)
                    ax.clabel(contour, inline=True, fontsize=8)
                except:
                    # Fallback to scatter plot
                    ax.scatter(x_coords, y_coords, c=z_coords, cmap='coolwarm', alpha=0.3, s=10)
        
        # Plot gradient vectors
        if gradient_points:
            x_coords = [p.x for p in gradient_points]
            y_coords = [p.y for p in gradient_points]
            u_components = [getattr(p, 'u', 0) for p in gradient_points]
            v_components = [getattr(p, 'v', 0) for p in gradient_points]
            
            # Create quiver plot for gradients
            ax.quiver(x_coords, y_coords, u_components, v_components,
                     color='red', alpha=0.8, scale=50, width=0.003)
        
        # Add gradient point markers
        for element in plot_data.interactive_elements:
            if element.element_type == 'gradient_point':
                ax.plot(element.position.x, element.position.y, 
                       'ro', markersize=8, zorder=5)
        
        ax.set_aspect('equal', adjustable='box')
    
    def _render_neural_network(self, ax, plot_data: PlotData):
        """Render neural network architecture."""
        if not plot_data.data_points:
            return
        
        # Separate neurons from connections
        neurons = [p for p in plot_data.data_points if not hasattr(p, 'is_connection')]
        connections = [p for p in plot_data.data_points if hasattr(p, 'is_connection')]
        
        # Draw connections first (so they appear behind neurons)
        for conn in connections:
            if hasattr(conn, 'from_pos') and hasattr(conn, 'to_pos'):
                weight = getattr(conn, 'weight', 1.0)
                
                # Line thickness based on weight magnitude
                line_width = min(abs(weight) * 2, 5)
                color = 'red' if weight < 0 else 'blue'
                alpha = min(abs(weight), 1.0)
                
                ax.plot([conn.from_pos.x, conn.to_pos.x], 
                       [conn.from_pos.y, conn.to_pos.y],
                       color=color, linewidth=line_width, alpha=alpha)
        
        # Draw neurons
        for neuron in neurons:
            # Color based on activation if available
            if hasattr(neuron, 'activation'):
                activation = neuron.activation
                color = plt.cm.RdYlBu(0.5 + activation * 0.5)  # Map activation to color
                size = 100 + abs(activation) * 200
            else:
                color = 'lightblue'
                size = 200
            
            ax.scatter(neuron.x, neuron.y, c=[color], s=size, 
                      edgecolors='black', linewidth=2, zorder=5)
        
        # Add highlighted neurons
        for element in plot_data.interactive_elements:
            if element.element_type == 'highlighted_neuron':
                ax.scatter(element.position.x, element.position.y, 
                          c='yellow', s=300, marker='*', 
                          edgecolors='red', linewidth=3, zorder=10)
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1, max([p.x for p in neurons]) + 1)
    
    def _render_optimization_path(self, ax, plot_data: PlotData):
        """Render optimization path with loss contours."""
        if not plot_data.data_points:
            return
        
        # Separate path points from contour points
        path_points = [p for p in plot_data.data_points if not hasattr(p, 'is_contour')]
        contour_points = [p for p in plot_data.data_points if hasattr(p, 'is_contour')]
        
        # Plot contours if available
        if contour_points:
            x_coords = [p.x for p in contour_points]
            y_coords = [p.y for p in contour_points]
            z_coords = [p.z for p in contour_points]
            
            # Estimate grid size
            unique_x = sorted(list(set(x_coords)))
            unique_y = sorted(list(set(y_coords)))
            
            if len(unique_x) > 1 and len(unique_y) > 1:
                try:
                    X = np.array(x_coords).reshape(len(unique_y), len(unique_x))
                    Y = np.array(y_coords).reshape(len(unique_y), len(unique_x))
                    Z = np.array(z_coords).reshape(len(unique_y), len(unique_x))
                    
                    # Create filled contour plot
                    contourf = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
                    plt.colorbar(contourf, ax=ax, label='Loss')
                    
                    # Add contour lines
                    contour = ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
                except:
                    # Fallback to scatter plot
                    scatter = ax.scatter(x_coords, y_coords, c=z_coords, cmap='viridis', alpha=0.6, s=20)
                    plt.colorbar(scatter, ax=ax, label='Loss')
        
        # Plot optimization path
        if path_points:
            x_path = [p.x for p in path_points]
            y_path = [p.y for p in path_points]
            
            # Plot path line
            ax.plot(x_path, y_path, 'r-', linewidth=3, alpha=0.8, label='Optimization Path')
            
            # Plot path points with step numbers
            for i, point in enumerate(path_points):
                # Color gradient from red to green (start to end)
                color = plt.cm.RdYlGn(i / max(len(path_points) - 1, 1))
                ax.scatter(point.x, point.y, c=[color], s=100, 
                          edgecolors='black', linewidth=1, zorder=5)
                
                # Add step number
                if i % max(len(path_points) // 10, 1) == 0:  # Show every nth step
                    ax.annotate(f'{i}', (point.x, point.y), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, fontweight='bold')
            
            # Highlight start and end points
            ax.scatter(x_path[0], y_path[0], c='red', s=200, marker='s', 
                      label='Start', zorder=10)
            ax.scatter(x_path[-1], y_path[-1], c='green', s=200, marker='*', 
                      label='End', zorder=10)
        
        # Add convergence info
        for element in plot_data.interactive_elements:
            if element.element_type == 'convergence_info':
                ax.text(0.02, 0.98, element.tooltip, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.8))
        
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
    
    def _render_loss_surface(self, ax, plot_data: PlotData):
        """Render 3D loss surface with optimization path."""
        if not plot_data.data_points:
            return
        
        # Extract coordinates
        x_coords = [p.x for p in plot_data.data_points]
        y_coords = [p.y for p in plot_data.data_points]
        z_coords = [p.z for p in plot_data.data_points if p.z is not None]
        
        if len(z_coords) != len(x_coords):
            return
        
        # Estimate grid size
        grid_size = int(np.sqrt(len(x_coords)))
        if grid_size * grid_size == len(x_coords):
            # Perfect grid - create surface
            X = np.array(x_coords).reshape(grid_size, grid_size)
            Y = np.array(y_coords).reshape(grid_size, grid_size)
            Z = np.array(z_coords).reshape(grid_size, grid_size)
            
            # Create surface plot
            surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.7, 
                                  linewidth=0.5, antialiased=True)
            plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Loss')
        else:
            # Fallback to scatter plot
            scatter = ax.scatter(x_coords, y_coords, z_coords, 
                               c=z_coords, cmap='plasma', s=20)
            plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Loss')
        
        # Add optimization path and special points
        for element in plot_data.interactive_elements:
            if element.element_type == 'optimization_step':
                ax.scatter(element.position.x, element.position.y, element.position.z,
                          c='red', s=50, alpha=0.8)
            elif element.element_type == 'global_minimum':
                ax.scatter(element.position.x, element.position.y, element.position.z,
                          c='green', s=200, marker='*', edgecolors='black', linewidth=2)
    
    def create_multi_function_plot(self, 
                                 expressions: List[str],
                                 labels: List[str],
                                 variable: str = 'x',
                                 x_range: Tuple[float, float] = (-10, 10),
                                 colors: Optional[List[str]] = None) -> PlotData:
        """
        Create a plot with multiple functions on the same axes.
        
        Args:
            expressions: List of mathematical expressions
            labels: Labels for each function
            variable: Variable name
            x_range: Range of x values
            colors: Optional list of colors for each function
            
        Returns:
            PlotData: Plot data with multiple functions
        """
        if colors is None:
            colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        try:
            all_points = []
            interactive_elements = []
            
            for i, (expr_str, label) in enumerate(zip(expressions, labels)):
                expr = sympify(expr_str)
                var_symbol = symbols(variable)
                func = lambdify(var_symbol, expr, 'numpy')
                
                x_vals = np.linspace(x_range[0], x_range[1], 500)
                
                for x in x_vals:
                    try:
                        y = func(x)
                        if np.isfinite(y):
                            # Add function index to distinguish different functions
                            point = Point(x=float(x), y=float(y))
                            point.function_index = i  # Custom attribute
                            all_points.append(point)
                    except:
                        continue
                
                # Add legend element
                interactive_elements.append(InteractiveElement(
                    element_type='legend',
                    position=Point(x=x_range[0] + 0.1 * (x_range[1] - x_range[0]), 
                                 y=0),  # Will be positioned properly in rendering
                    action='toggle_function',
                    tooltip=f'{label}: {expr_str}'
                ))
            
            plot_data = PlotData(
                plot_type='multi_function',
                data_points=all_points,
                styling=PlotStyle(color='blue', line_width=2.0),  # Default, will be overridden
                interactive_elements=interactive_elements,
                title="Multiple Functions",
                axis_labels={'x': variable, 'y': 'y'}
            )
            
            return plot_data
            
        except Exception as e:
            raise ValueError(f"Error creating multi-function plot: {str(e)}")
    
    def create_vector_plot(self,
                          vectors: List[Tuple[float, float, Optional[float]]],
                          labels: Optional[List[str]] = None,
                          colors: Optional[List[str]] = None,
                          origin: Tuple[float, float, Optional[float]] = (0, 0, None),
                          show_grid: bool = True,
                          is_3d: bool = False) -> PlotData:
        """
        Create a vector visualization plot.
        
        Args:
            vectors: List of vectors as (x, y, z) tuples (z can be None for 2D)
            labels: Optional labels for each vector
            colors: Optional colors for each vector
            origin: Origin point for vectors
            show_grid: Whether to show grid
            is_3d: Whether to create 3D plot
            
        Returns:
            PlotData: Vector plot data
        """
        if colors is None:
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        if labels is None:
            labels = [f'v{i+1}' for i in range(len(vectors))]
        
        try:
            data_points = []
            interactive_elements = []
            
            # Add origin point
            if is_3d and origin[2] is not None:
                data_points.append(Point(x=origin[0], y=origin[1], z=origin[2]))
            else:
                data_points.append(Point(x=origin[0], y=origin[1]))
            
            # Process each vector
            for i, vector in enumerate(vectors):
                if is_3d and len(vector) == 3 and vector[2] is not None:
                    # 3D vector
                    end_point = Point(
                        x=origin[0] + vector[0],
                        y=origin[1] + vector[1], 
                        z=origin[2] + vector[2] if origin[2] is not None else vector[2]
                    )
                else:
                    # 2D vector
                    end_point = Point(
                        x=origin[0] + vector[0],
                        y=origin[1] + vector[1]
                    )
                
                data_points.append(end_point)
                
                # Add interactive element for vector
                interactive_elements.append(InteractiveElement(
                    element_type='vector',
                    position=end_point,
                    action='show_vector_info',
                    tooltip=f'{labels[i] if i < len(labels) else f"v{i+1}"}: {vector}'
                ))
            
            plot_type = 'vector_3d' if is_3d else 'vector_2d'
            title = f"{'3D' if is_3d else '2D'} Vector Visualization"
            
            plot_data = PlotData(
                plot_type=plot_type,
                data_points=data_points,
                styling=PlotStyle(color='red', line_width=2.0),
                interactive_elements=interactive_elements,
                title=title,
                axis_labels={'x': 'X', 'y': 'Y', 'z': 'Z'} if is_3d else {'x': 'X', 'y': 'Y'}
            )
            
            return plot_data
            
        except Exception as e:
            raise ValueError(f"Error creating vector plot: {str(e)}")
    
    def create_vector_field_plot(self,
                                vector_function: str,
                                x_range: Tuple[float, float] = (-5, 5),
                                y_range: Tuple[float, float] = (-5, 5),
                                grid_density: int = 20,
                                normalize_arrows: bool = True) -> PlotData:
        """
        Create a vector field visualization.
        
        Args:
            vector_function: Vector function as string (e.g., "[-y, x]" for rotation)
            x_range: Range of x values
            y_range: Range of y values
            grid_density: Number of arrows per axis
            normalize_arrows: Whether to normalize arrow lengths
            
        Returns:
            PlotData: Vector field plot data
        """
        try:
            # Parse vector function
            x, y = symbols('x y')
            
            # Handle different input formats
            if vector_function.startswith('[') and vector_function.endswith(']'):
                # List format: "[-y, x]"
                components = vector_function[1:-1].split(',')
                fx_expr = sympify(components[0].strip())
                fy_expr = sympify(components[1].strip())
            else:
                # Assume comma-separated: "-y, x"
                components = vector_function.split(',')
                fx_expr = sympify(components[0].strip())
                fy_expr = sympify(components[1].strip())
            
            # Create numerical functions
            fx_func = lambdify((x, y), fx_expr, 'numpy')
            fy_func = lambdify((x, y), fy_expr, 'numpy')
            
            # Generate grid
            x_vals = np.linspace(x_range[0], x_range[1], grid_density)
            y_vals = np.linspace(y_range[0], y_range[1], grid_density)
            X, Y = np.meshgrid(x_vals, y_vals)
            
            # Calculate vector components
            U = fx_func(X, Y)
            V = fy_func(X, Y)
            
            # Normalize if requested
            if normalize_arrows:
                magnitude = np.sqrt(U**2 + V**2)
                magnitude[magnitude == 0] = 1  # Avoid division by zero
                U = U / magnitude
                V = V / magnitude
            
            # Convert to data points (store grid and vector data)
            data_points = []
            interactive_elements = []
            
            for i in range(len(x_vals)):
                for j in range(len(y_vals)):
                    point = Point(x=float(X[j, i]), y=float(Y[j, i]))
                    # Store vector components as custom attributes
                    point.u = float(U[j, i]) if np.isfinite(U[j, i]) else 0.0
                    point.v = float(V[j, i]) if np.isfinite(V[j, i]) else 0.0
                    data_points.append(point)
            
            plot_data = PlotData(
                plot_type='vector_field',
                data_points=data_points,
                styling=PlotStyle(color='blue', line_width=1.0),
                interactive_elements=interactive_elements,
                title=f"Vector Field: {vector_function}",
                axis_labels={'x': 'X', 'y': 'Y'}
            )
            
            return plot_data
            
        except Exception as e:
            raise ValueError(f"Error creating vector field plot: {str(e)}")
    
    def create_matrix_transformation_plot(self,
                                        matrix: List[List[float]],
                                        input_vectors: Optional[List[Tuple[float, float]]] = None,
                                        show_unit_circle: bool = True,
                                        show_grid_transformation: bool = True) -> PlotData:
        """
        Create a visualization of matrix transformation.
        
        Args:
            matrix: 2x2 transformation matrix as [[a, b], [c, d]]
            input_vectors: Optional input vectors to transform
            show_unit_circle: Whether to show unit circle transformation
            show_grid_transformation: Whether to show grid line transformation
            
        Returns:
            PlotData: Matrix transformation plot data
        """
        try:
            if len(matrix) != 2 or len(matrix[0]) != 2:
                raise ValueError("Matrix must be 2x2 for visualization")
            
            A = np.array(matrix)
            data_points = []
            interactive_elements = []
            
            # Default input vectors if none provided
            if input_vectors is None:
                input_vectors = [(1, 0), (0, 1)]  # Standard basis vectors
            
            # Transform input vectors
            for i, (x, y) in enumerate(input_vectors):
                input_vec = np.array([x, y])
                transformed = A @ input_vec
                
                # Add original vector
                data_points.append(Point(x=float(x), y=float(y)))
                # Add transformed vector
                data_points.append(Point(x=float(transformed[0]), y=float(transformed[1])))
                
                # Add interactive elements
                interactive_elements.append(InteractiveElement(
                    element_type='vector_pair',
                    position=Point(x=float(transformed[0]), y=float(transformed[1])),
                    action='show_transformation',
                    tooltip=f'({x}, {y}) → ({transformed[0]:.2f}, {transformed[1]:.2f})'
                ))
            
            # Add unit circle transformation if requested
            if show_unit_circle:
                theta = np.linspace(0, 2*np.pi, 100)
                unit_circle = np.array([np.cos(theta), np.sin(theta)])
                transformed_circle = A @ unit_circle
                
                for i in range(len(theta)):
                    data_points.append(Point(
                        x=float(transformed_circle[0, i]),
                        y=float(transformed_circle[1, i])
                    ))
            
            # Add grid transformation if requested
            if show_grid_transformation:
                # Create grid lines
                grid_range = np.linspace(-2, 2, 5)
                
                # Vertical lines
                for x_val in grid_range:
                    y_vals = np.linspace(-2, 2, 20)
                    for y_val in y_vals:
                        point = np.array([x_val, y_val])
                        transformed = A @ point
                        data_points.append(Point(
                            x=float(transformed[0]),
                            y=float(transformed[1])
                        ))
                
                # Horizontal lines  
                for y_val in grid_range:
                    x_vals = np.linspace(-2, 2, 20)
                    for x_val in x_vals:
                        point = np.array([x_val, y_val])
                        transformed = A @ point
                        data_points.append(Point(
                            x=float(transformed[0]),
                            y=float(transformed[1])
                        ))
            
            # Calculate determinant for title
            det = np.linalg.det(A)
            
            plot_data = PlotData(
                plot_type='matrix_transformation',
                data_points=data_points,
                styling=PlotStyle(color='purple', line_width=2.0),
                interactive_elements=interactive_elements,
                title=f"Matrix Transformation (det = {det:.2f})",
                axis_labels={'x': 'X', 'y': 'Y'}
            )
            
            return plot_data
            
        except Exception as e:
            raise ValueError(f"Error creating matrix transformation plot: {str(e)}")
    
    def create_3d_surface_plot(self,
                              function_expr: str,
                              x_range: Tuple[float, float] = (-5, 5),
                              y_range: Tuple[float, float] = (-5, 5),
                              resolution: int = 50,
                              colormap: str = 'viridis') -> PlotData:
        """
        Create a 3D surface plot for functions of two variables.
        
        Args:
            function_expr: Function expression (e.g., "x**2 + y**2")
            x_range: Range of x values
            y_range: Range of y values
            resolution: Number of points per axis
            colormap: Matplotlib colormap name
            
        Returns:
            PlotData: 3D surface plot data
        """
        try:
            # Parse expression
            x, y = symbols('x y')
            expr = sympify(function_expr)
            func = lambdify((x, y), expr, 'numpy')
            
            # Generate grid
            x_vals = np.linspace(x_range[0], x_range[1], resolution)
            y_vals = np.linspace(y_range[0], y_range[1], resolution)
            X, Y = np.meshgrid(x_vals, y_vals)
            
            # Calculate Z values
            Z = func(X, Y)
            
            # Convert to data points
            data_points = []
            for i in range(resolution):
                for j in range(resolution):
                    if np.isfinite(Z[i, j]):
                        data_points.append(Point(
                            x=float(X[i, j]),
                            y=float(Y[i, j]),
                            z=float(Z[i, j])
                        ))
            
            interactive_elements = [
                InteractiveElement(
                    element_type='surface',
                    position=Point(x=0, y=0, z=0),
                    action='rotate_3d',
                    tooltip='Click and drag to rotate'
                )
            ]
            
            plot_data = PlotData(
                plot_type='surface_3d',
                data_points=data_points,
                styling=PlotStyle(color=colormap, line_width=1.0),
                interactive_elements=interactive_elements,
                title=f"3D Surface: f(x,y) = {function_expr}",
                axis_labels={'x': 'X', 'y': 'Y', 'z': 'Z'}
            )
            
            return plot_data
            
        except Exception as e:
            raise ValueError(f"Error creating 3D surface plot: {str(e)}")
    
    def create_loss_surface_plot(self,
                               loss_function: str,
                               parameter_ranges: Dict[str, Tuple[float, float]],
                               optimization_path: Optional[List[Tuple[float, float]]] = None,
                               global_minimum: Optional[Tuple[float, float]] = None,
                               resolution: int = 50) -> PlotData:
        """
        Create a loss surface visualization for optimization problems.
        
        Args:
            loss_function: Loss function expression (e.g., "(x-2)**2 + (y-1)**2")
            parameter_ranges: Dict with parameter ranges (e.g., {'x': (-5, 5), 'y': (-5, 5)})
            optimization_path: Optional list of (x, y) points showing optimization trajectory
            global_minimum: Optional global minimum point
            resolution: Grid resolution for surface
            
        Returns:
            PlotData: Loss surface plot data
        """
        try:
            # Get parameter names and ranges
            param_names = list(parameter_ranges.keys())
            if len(param_names) != 2:
                raise ValueError("Loss surface visualization requires exactly 2 parameters")
            
            param1, param2 = param_names
            range1, range2 = parameter_ranges[param1], parameter_ranges[param2]
            
            # Parse loss function
            symbols_dict = {param1: symbols(param1), param2: symbols(param2)}
            expr = sympify(loss_function)
            func = lambdify((symbols_dict[param1], symbols_dict[param2]), expr, 'numpy')
            
            # Generate parameter grid
            p1_vals = np.linspace(range1[0], range1[1], resolution)
            p2_vals = np.linspace(range2[0], range2[1], resolution)
            P1, P2 = np.meshgrid(p1_vals, p2_vals)
            
            # Calculate loss values
            Loss = func(P1, P2)
            
            # Convert to data points
            data_points = []
            for i in range(resolution):
                for j in range(resolution):
                    if np.isfinite(Loss[i, j]):
                        data_points.append(Point(
                            x=float(P1[i, j]),
                            y=float(P2[i, j]),
                            z=float(Loss[i, j])
                        ))
            
            interactive_elements = []
            
            # Add optimization path
            if optimization_path:
                for i, (p1, p2) in enumerate(optimization_path):
                    loss_val = func(p1, p2)
                    interactive_elements.append(InteractiveElement(
                        element_type='optimization_step',
                        position=Point(x=float(p1), y=float(p2), z=float(loss_val)),
                        action='show_step_info',
                        tooltip=f'Step {i+1}: {param1}={p1:.3f}, {param2}={p2:.3f}, Loss={loss_val:.3f}'
                    ))
            
            # Add global minimum
            if global_minimum:
                p1_min, p2_min = global_minimum
                loss_min = func(p1_min, p2_min)
                interactive_elements.append(InteractiveElement(
                    element_type='global_minimum',
                    position=Point(x=float(p1_min), y=float(p2_min), z=float(loss_min)),
                    action='highlight_minimum',
                    tooltip=f'Global minimum: {param1}={p1_min:.3f}, {param2}={p2_min:.3f}, Loss={loss_min:.3f}'
                ))
            
            plot_data = PlotData(
                plot_type='loss_surface',
                data_points=data_points,
                styling=PlotStyle(color='plasma', line_width=1.0),
                interactive_elements=interactive_elements,
                title=f"Loss Surface: {loss_function}",
                axis_labels={'x': param1, 'y': param2, 'z': 'Loss'}
            )
            
            return plot_data
            
        except Exception as e:
            raise ValueError(f"Error creating loss surface plot: {str(e)}")
    
    def create_gradient_visualization(self,
                                    function_expr: str,
                                    x_range: Tuple[float, float] = (-5, 5),
                                    y_range: Tuple[float, float] = (-5, 5),
                                    gradient_points: Optional[List[Tuple[float, float]]] = None,
                                    show_contours: bool = True,
                                    grid_density: int = 20) -> PlotData:
        """
        Create gradient visualization with contour lines and gradient vectors.
        
        Args:
            function_expr: Function expression (e.g., "x**2 + y**2")
            x_range: Range of x values
            y_range: Range of y values
            gradient_points: Specific points to show gradients at
            show_contours: Whether to show contour lines
            grid_density: Density of gradient arrows
            
        Returns:
            PlotData: Gradient visualization plot data
        """
        try:
            # Parse function and compute gradient
            x, y = symbols('x y')
            expr = sympify(function_expr)
            
            # Compute partial derivatives
            grad_x = sp.diff(expr, x)
            grad_y = sp.diff(expr, y)
            
            # Create numerical functions
            func = lambdify((x, y), expr, 'numpy')
            grad_x_func = lambdify((x, y), grad_x, 'numpy')
            grad_y_func = lambdify((x, y), grad_y, 'numpy')
            
            # Generate grid for contours
            x_vals = np.linspace(x_range[0], x_range[1], 100)
            y_vals = np.linspace(y_range[0], y_range[1], 100)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = func(X, Y)
            
            data_points = []
            interactive_elements = []
            
            # Add contour data points if requested
            if show_contours:
                for i in range(len(x_vals)):
                    for j in range(len(y_vals)):
                        if np.isfinite(Z[j, i]):
                            point = Point(x=float(X[j, i]), y=float(Y[j, i]))
                            point.z = float(Z[j, i])  # Store function value
                            data_points.append(point)
            
            # Generate gradient field
            grad_x_vals = np.linspace(x_range[0], x_range[1], grid_density)
            grad_y_vals = np.linspace(y_range[0], y_range[1], grid_density)
            GX, GY = np.meshgrid(grad_x_vals, grad_y_vals)
            
            # Calculate gradient components
            U = grad_x_func(GX, GY)
            V = grad_y_func(GX, GY)
            
            # Add gradient vectors as data points
            for i in range(grid_density):
                for j in range(grid_density):
                    if np.isfinite(U[i, j]) and np.isfinite(V[i, j]):
                        point = Point(x=float(GX[i, j]), y=float(GY[i, j]))
                        point.u = float(U[i, j])  # Gradient x-component
                        point.v = float(V[i, j])  # Gradient y-component
                        point.is_gradient = True  # Mark as gradient point
                        data_points.append(point)
            
            # Add specific gradient points if provided
            if gradient_points:
                for i, (px, py) in enumerate(gradient_points):
                    grad_x_val = grad_x_func(px, py)
                    grad_y_val = grad_y_func(px, py)
                    func_val = func(px, py)
                    
                    interactive_elements.append(InteractiveElement(
                        element_type='gradient_point',
                        position=Point(x=float(px), y=float(py)),
                        action='show_gradient_info',
                        tooltip=f'∇f({px:.2f}, {py:.2f}) = ({grad_x_val:.3f}, {grad_y_val:.3f}), f = {func_val:.3f}'
                    ))
            
            plot_data = PlotData(
                plot_type='gradient_field',
                data_points=data_points,
                styling=PlotStyle(color='coolwarm', line_width=1.5),
                interactive_elements=interactive_elements,
                title=f"Gradient Field: ∇({function_expr})",
                axis_labels={'x': 'x', 'y': 'y'}
            )
            
            return plot_data
            
        except Exception as e:
            raise ValueError(f"Error creating gradient visualization: {str(e)}")
    
    def create_neural_network_visualization(self,
                                          layer_sizes: List[int],
                                          weights: Optional[List[np.ndarray]] = None,
                                          activations: Optional[List[np.ndarray]] = None,
                                          highlight_path: Optional[List[int]] = None) -> PlotData:
        """
        Create neural network architecture visualization.
        
        Args:
            layer_sizes: List of neurons per layer (e.g., [3, 4, 2])
            weights: Optional weight matrices between layers
            activations: Optional activation values for each layer
            highlight_path: Optional path through network to highlight
            
        Returns:
            PlotData: Neural network visualization plot data
        """
        try:
            data_points = []
            interactive_elements = []
            
            # Calculate positions for neurons
            max_layer_size = max(layer_sizes)
            layer_spacing = 3.0
            neuron_spacing = 1.0
            
            neuron_positions = []
            
            for layer_idx, layer_size in enumerate(layer_sizes):
                layer_positions = []
                x_pos = layer_idx * layer_spacing
                
                # Center neurons vertically
                start_y = -(layer_size - 1) * neuron_spacing / 2
                
                for neuron_idx in range(layer_size):
                    y_pos = start_y + neuron_idx * neuron_spacing
                    pos = Point(x=x_pos, y=y_pos)
                    
                    # Add activation value if provided
                    if activations and layer_idx < len(activations):
                        if neuron_idx < len(activations[layer_idx]):
                            pos.activation = float(activations[layer_idx][neuron_idx])
                    
                    layer_positions.append(pos)
                    data_points.append(pos)
                
                neuron_positions.append(layer_positions)
            
            # Add connections (weights) between layers
            if weights:
                for layer_idx in range(len(layer_sizes) - 1):
                    if layer_idx < len(weights):
                        weight_matrix = weights[layer_idx]
                        
                        for i, from_neuron in enumerate(neuron_positions[layer_idx]):
                            for j, to_neuron in enumerate(neuron_positions[layer_idx + 1]):
                                if i < weight_matrix.shape[0] and j < weight_matrix.shape[1]:
                                    weight_val = weight_matrix[i, j]
                                    
                                    # Create connection point
                                    connection = Point(
                                        x=(from_neuron.x + to_neuron.x) / 2,
                                        y=(from_neuron.y + to_neuron.y) / 2
                                    )
                                    connection.weight = float(weight_val)
                                    connection.from_pos = from_neuron
                                    connection.to_pos = to_neuron
                                    connection.is_connection = True
                                    data_points.append(connection)
            
            # Add interactive elements for neurons
            for layer_idx, layer_positions in enumerate(neuron_positions):
                for neuron_idx, pos in enumerate(layer_positions):
                    activation_text = ""
                    if hasattr(pos, 'activation'):
                        activation_text = f", activation={pos.activation:.3f}"
                    
                    interactive_elements.append(InteractiveElement(
                        element_type='neuron',
                        position=pos,
                        action='show_neuron_info',
                        tooltip=f'Layer {layer_idx}, Neuron {neuron_idx}{activation_text}'
                    ))
            
            # Add highlight path if provided
            if highlight_path and len(highlight_path) == len(layer_sizes):
                for layer_idx in range(len(highlight_path)):
                    neuron_idx = highlight_path[layer_idx]
                    if neuron_idx < len(neuron_positions[layer_idx]):
                        pos = neuron_positions[layer_idx][neuron_idx]
                        
                        interactive_elements.append(InteractiveElement(
                            element_type='highlighted_neuron',
                            position=pos,
                            action='highlight_path',
                            tooltip=f'Highlighted path: Layer {layer_idx}, Neuron {neuron_idx}'
                        ))
            
            plot_data = PlotData(
                plot_type='neural_network',
                data_points=data_points,
                styling=PlotStyle(color='blue', line_width=1.0),
                interactive_elements=interactive_elements,
                title=f"Neural Network Architecture: {' → '.join(map(str, layer_sizes))}",
                axis_labels={'x': 'Layer', 'y': 'Neuron Position'}
            )
            
            return plot_data
            
        except Exception as e:
            raise ValueError(f"Error creating neural network visualization: {str(e)}")
    
    def create_optimization_path_plot(self,
                                    loss_function: str,
                                    optimization_steps: List[Tuple[float, float, float]],
                                    algorithm_name: str = "Gradient Descent",
                                    parameter_names: Tuple[str, str] = ("w1", "w2"),
                                    show_loss_contours: bool = True) -> PlotData:
        """
        Create optimization path visualization showing parameter updates over time.
        
        Args:
            loss_function: Loss function expression
            optimization_steps: List of (param1, param2, loss) tuples for each step
            algorithm_name: Name of optimization algorithm
            parameter_names: Names of the two parameters
            show_loss_contours: Whether to show loss contours in background
            
        Returns:
            PlotData: Optimization path plot data
        """
        try:
            data_points = []
            interactive_elements = []
            
            # Add optimization path points
            for i, (p1, p2, loss) in enumerate(optimization_steps):
                point = Point(x=float(p1), y=float(p2))
                point.loss = float(loss)
                point.step = i
                data_points.append(point)
                
                # Add interactive element for each step
                interactive_elements.append(InteractiveElement(
                    element_type='optimization_step',
                    position=point,
                    action='show_step_details',
                    tooltip=f'Step {i}: {parameter_names[0]}={p1:.4f}, {parameter_names[1]}={p2:.4f}, Loss={loss:.4f}'
                ))
            
            # Add contour background if requested
            if show_loss_contours and len(optimization_steps) > 0:
                # Determine plot range based on optimization path
                x_coords = [step[0] for step in optimization_steps]
                y_coords = [step[1] for step in optimization_steps]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Expand range slightly
                x_range = x_max - x_min
                y_range = y_max - y_min
                x_min -= 0.2 * x_range
                x_max += 0.2 * x_range
                y_min -= 0.2 * y_range
                y_max += 0.2 * y_range
                
                # Generate contour grid
                x_vals = np.linspace(x_min, x_max, 50)
                y_vals = np.linspace(y_min, y_max, 50)
                X, Y = np.meshgrid(x_vals, y_vals)
                
                # Parse and evaluate loss function
                p1_sym, p2_sym = symbols(f'{parameter_names[0]} {parameter_names[1]}')
                expr = sympify(loss_function)
                func = lambdify((p1_sym, p2_sym), expr, 'numpy')
                Z = func(X, Y)
                
                # Add contour data points
                for i in range(len(x_vals)):
                    for j in range(len(y_vals)):
                        if np.isfinite(Z[j, i]):
                            contour_point = Point(x=float(X[j, i]), y=float(Y[j, i]))
                            contour_point.z = float(Z[j, i])
                            contour_point.is_contour = True
                            data_points.append(contour_point)
            
            # Add convergence info
            if len(optimization_steps) > 1:
                start_loss = optimization_steps[0][2]
                end_loss = optimization_steps[-1][2]
                improvement = start_loss - end_loss
                
                interactive_elements.append(InteractiveElement(
                    element_type='convergence_info',
                    position=Point(x=optimization_steps[-1][0], y=optimization_steps[-1][1]),
                    action='show_convergence',
                    tooltip=f'Converged in {len(optimization_steps)} steps. Loss: {start_loss:.4f} → {end_loss:.4f} (Δ={improvement:.4f})'
                ))
            
            plot_data = PlotData(
                plot_type='optimization_path',
                data_points=data_points,
                styling=PlotStyle(color='red', line_width=2.0),
                interactive_elements=interactive_elements,
                title=f"{algorithm_name} Optimization Path",
                axis_labels={'x': parameter_names[0], 'y': parameter_names[1]}
            )
            
            return plot_data
            
        except Exception as e:
            raise ValueError(f"Error creating optimization path plot: {str(e)}")


class VisualizationEngine:
    """High-level visualization engine that integrates with the math solver."""
    
    def __init__(self):
        """Initialize the visualization engine."""
        self.plotter = MathPlotter()
    
    def generate_problem_visualization(self, 
                                     problem: ParsedProblem, 
                                     solution: StepSolution) -> PlotData:
        """
        Generate appropriate visualization for a mathematical problem and its solution.
        
        Args:
            problem: Parsed mathematical problem
            solution: Step-by-step solution
            
        Returns:
            PlotData: Visualization data for the problem
        """
        try:
            if problem.problem_type == 'derivative':
                return self._create_derivative_visualization(problem, solution)
            elif problem.problem_type == 'integral':
                return self._create_integral_visualization(problem, solution)
            elif problem.problem_type == 'optimization':
                return self._create_optimization_visualization(problem, solution)
            elif problem.problem_type == 'limit':
                return self._create_limit_visualization(problem, solution)
            elif problem.problem_type in ['linear_equation', 'quadratic_equation']:
                return self._create_equation_visualization(problem, solution)
            else:
                # Default function plot
                return self._create_default_visualization(problem, solution)
                
        except Exception as e:
            # Return empty plot data on error
            return PlotData(
                plot_type='error',
                data_points=[],
                styling=PlotStyle(color='red', line_width=1.0),
                interactive_elements=[],
                title=f"Visualization Error: {str(e)}",
                axis_labels={'x': 'x', 'y': 'y'}
            )
    
    def _create_derivative_visualization(self, problem: ParsedProblem, solution: StepSolution) -> PlotData:
        """Create visualization for derivative problems."""
        # Extract function from problem expressions
        function_expr = self._extract_function_expression(problem)
        if not function_expr:
            function_expr = "x**2"  # Default
        
        # Extract derivative from solution
        derivative_expr = solution.final_answer
        
        return self.plotter.create_derivative_plot(
            function_expr=function_expr,
            derivative_expr=derivative_expr,
            highlight_point=0.0  # Highlight tangent at x=0
        )
    
    def _create_integral_visualization(self, problem: ParsedProblem, solution: StepSolution) -> PlotData:
        """Create visualization for integral problems."""
        function_expr = self._extract_function_expression(problem)
        if not function_expr:
            function_expr = "x**2"
        
        # Check if it's a definite integral
        integral_bounds = self._extract_integral_bounds(problem)
        
        return self.plotter.create_integral_plot(
            function_expr=function_expr,
            integral_bounds=integral_bounds,
            show_area=integral_bounds is not None
        )
    
    def _create_optimization_visualization(self, problem: ParsedProblem, solution: StepSolution) -> PlotData:
        """Create visualization for optimization problems."""
        function_expr = self._extract_function_expression(problem)
        if not function_expr:
            function_expr = "x**2"
        
        # Extract critical points from solution steps
        critical_points = self._extract_critical_points(solution)
        
        return self.plotter.create_optimization_plot(
            function_expr=function_expr,
            critical_points=critical_points
        )
    
    def _create_limit_visualization(self, problem: ParsedProblem, solution: StepSolution) -> PlotData:
        """Create visualization for limit problems."""
        function_expr = self._extract_function_expression(problem)
        if not function_expr:
            function_expr = "1/x"
        
        return self.plotter.create_function_plot(
            expression=function_expr,
            title=f"Limit Problem: {function_expr}"
        )
    
    def _create_equation_visualization(self, problem: ParsedProblem, solution: StepSolution) -> PlotData:
        """Create visualization for equation problems."""
        # For equations, plot the function and highlight the solution
        if problem.expressions:
            expr = problem.expressions[0]
            # Convert equation to function (e.g., "2x + 3 = 7" -> "2x + 3 - 7")
            if '=' in expr:
                left, right = expr.split('=')
                function_expr = f"({left.strip()}) - ({right.strip()})"
            else:
                function_expr = expr
        else:
            function_expr = "x"
        
        return self.plotter.create_function_plot(
            expression=function_expr,
            title=f"Equation: {problem.original_text}"
        )
    
    def _create_default_visualization(self, problem: ParsedProblem, solution: StepSolution) -> PlotData:
        """Create default visualization for unsupported problem types."""
        return self.plotter.create_function_plot(
            expression="x",
            title=f"Problem: {problem.original_text}"
        )
    
    def _extract_function_expression(self, problem: ParsedProblem) -> Optional[str]:
        """Extract the main function expression from a problem."""
        if problem.expressions:
            # Return the longest expression (likely the main function)
            return max(problem.expressions, key=len)
        return None
    
    def _extract_integral_bounds(self, problem: ParsedProblem) -> Optional[Tuple[float, float]]:
        """Extract integral bounds from problem text."""
        import re
        text = problem.original_text.lower()
        
        # Look for patterns like "from 0 to 2" or "∫₀²"
        patterns = [
            r'from\s+([+-]?\d*\.?\d+)\s+to\s+([+-]?\d*\.?\d+)',
            r'∫\[([+-]?\d*\.?\d+)\s+to\s+([+-]?\d*\.?\d+)\]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    a, b = float(match.group(1)), float(match.group(2))
                    return (a, b)
                except:
                    continue
        
        return None
    
    def _extract_critical_points(self, solution: StepSolution) -> List[float]:
        """Extract critical points from solution steps."""
        critical_points = []
        
        for step in solution.steps:
            if 'critical' in step.operation.lower():
                # Try to extract numbers from the step result
                import re
                numbers = re.findall(r'[+-]?\d*\.?\d+', step.intermediate_result)
                for num_str in numbers:
                    try:
                        critical_points.append(float(num_str))
                    except:
                        continue
        
        return critical_points