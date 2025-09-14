import React from 'react';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

/**
 * MathRenderer component for displaying mathematical expressions using KaTeX
 * Supports both inline and block math rendering
 */
function MathRenderer({ math, block = false, className = '' }) {
  if (!math) return null;

  try {
    if (block) {
      return (
        <div className={`math-block ${className}`}>
          <BlockMath math={math} />
        </div>
      );
    } else {
      return (
        <span className={`math-inline ${className}`}>
          <InlineMath math={math} />
        </span>
      );
    }
  } catch (error) {
    // Fallback for invalid LaTeX
    return (
      <span className={`math-error ${className}`} style={{ color: 'red' }}>
        [Math Error: {math}]
      </span>
    );
  }
}

export default MathRenderer;