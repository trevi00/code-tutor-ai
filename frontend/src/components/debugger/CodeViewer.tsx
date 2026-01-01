/**
 * Code Viewer Component for Debugger
 * Displays code with line numbers and highlights current execution line
 */

import { useEffect, useRef } from 'react';
import { Circle } from 'lucide-react';

interface CodeViewerProps {
  code: string;
  currentLine?: number;
  breakpoints?: number[];
  onToggleBreakpoint?: (lineNumber: number) => void;
  className?: string;
}

export default function CodeViewer({
  code,
  currentLine,
  breakpoints = [],
  onToggleBreakpoint,
  className = '',
}: CodeViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const lines = code.split('\n');

  // Scroll to current line
  useEffect(() => {
    if (currentLine && containerRef.current) {
      const lineElement = containerRef.current.querySelector(
        `[data-line="${currentLine}"]`
      );
      if (lineElement) {
        lineElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [currentLine]);

  return (
    <div
      ref={containerRef}
      className={`bg-gray-900 rounded-lg overflow-auto font-mono text-sm ${className}`}
    >
      <table className="w-full border-collapse">
        <tbody>
          {lines.map((line, index) => {
            const lineNumber = index + 1;
            const isCurrentLine = lineNumber === currentLine;
            const hasBreakpoint = breakpoints.includes(lineNumber);

            return (
              <tr
                key={lineNumber}
                data-line={lineNumber}
                className={`${
                  isCurrentLine
                    ? 'bg-yellow-500/20'
                    : hasBreakpoint
                    ? 'bg-red-500/10'
                    : 'hover:bg-gray-800/50'
                }`}
              >
                {/* Breakpoint column */}
                <td className="w-6 px-1 select-none">
                  <button
                    onClick={() => onToggleBreakpoint?.(lineNumber)}
                    className="w-4 h-4 flex items-center justify-center"
                    title={hasBreakpoint ? '브레이크포인트 제거' : '브레이크포인트 추가'}
                  >
                    {hasBreakpoint ? (
                      <Circle className="w-3 h-3 fill-red-500 text-red-500" />
                    ) : (
                      <span className="w-3 h-3 rounded-full hover:bg-red-500/50" />
                    )}
                  </button>
                </td>

                {/* Line number column */}
                <td
                  className={`w-10 px-2 text-right select-none ${
                    isCurrentLine ? 'text-yellow-400' : 'text-gray-500'
                  }`}
                >
                  {lineNumber}
                </td>

                {/* Current line indicator */}
                <td className="w-4 select-none">
                  {isCurrentLine && (
                    <span className="text-yellow-400">▶</span>
                  )}
                </td>

                {/* Code content */}
                <td className="px-2 py-0.5">
                  <pre
                    className={`whitespace-pre ${
                      isCurrentLine ? 'text-yellow-100' : 'text-gray-300'
                    }`}
                  >
                    {highlightSyntax(line)}
                  </pre>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// Simple syntax highlighting for Python
function highlightSyntax(code: string): React.ReactNode {
  // Keywords
  const keywords = [
    'def', 'class', 'if', 'elif', 'else', 'for', 'while', 'try', 'except',
    'finally', 'with', 'as', 'import', 'from', 'return', 'yield', 'raise',
    'break', 'continue', 'pass', 'lambda', 'and', 'or', 'not', 'in', 'is',
    'True', 'False', 'None', 'global', 'nonlocal', 'assert', 'async', 'await',
  ];

  // Built-in functions
  const builtins = [
    'print', 'len', 'range', 'int', 'str', 'float', 'list', 'dict', 'set',
    'tuple', 'bool', 'input', 'open', 'type', 'isinstance', 'hasattr',
    'getattr', 'setattr', 'enumerate', 'zip', 'map', 'filter', 'sorted',
    'reversed', 'sum', 'min', 'max', 'abs', 'round', 'pow', 'divmod',
  ];

  const parts: React.ReactNode[] = [];
  let remaining = code;
  let key = 0;

  while (remaining.length > 0) {
    let matched = false;

    // String (double quote)
    const stringMatch = remaining.match(/^("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/);
    if (stringMatch) {
      parts.push(
        <span key={key++} className="text-green-400">
          {stringMatch[1]}
        </span>
      );
      remaining = remaining.slice(stringMatch[1].length);
      matched = true;
      continue;
    }

    // Comment
    const commentMatch = remaining.match(/^(#.*)/);
    if (commentMatch) {
      parts.push(
        <span key={key++} className="text-gray-500 italic">
          {commentMatch[1]}
        </span>
      );
      remaining = remaining.slice(commentMatch[1].length);
      matched = true;
      continue;
    }

    // Number
    const numberMatch = remaining.match(/^(\d+\.?\d*)/);
    if (numberMatch) {
      parts.push(
        <span key={key++} className="text-blue-400">
          {numberMatch[1]}
        </span>
      );
      remaining = remaining.slice(numberMatch[1].length);
      matched = true;
      continue;
    }

    // Identifier or keyword
    const identMatch = remaining.match(/^([a-zA-Z_][a-zA-Z0-9_]*)/);
    if (identMatch) {
      const word = identMatch[1];
      if (keywords.includes(word)) {
        parts.push(
          <span key={key++} className="text-purple-400 font-semibold">
            {word}
          </span>
        );
      } else if (builtins.includes(word)) {
        parts.push(
          <span key={key++} className="text-cyan-400">
            {word}
          </span>
        );
      } else {
        parts.push(<span key={key++}>{word}</span>);
      }
      remaining = remaining.slice(word.length);
      matched = true;
      continue;
    }

    // Other characters
    if (!matched) {
      parts.push(remaining[0]);
      remaining = remaining.slice(1);
    }
  }

  return <>{parts}</>;
}

// Compact code viewer for inline display
interface CompactCodeViewerProps {
  code: string;
  lineNumber: number;
  className?: string;
}

export function CompactCodeViewer({ code, lineNumber, className = '' }: CompactCodeViewerProps) {
  return (
    <div className={`flex items-center gap-2 font-mono text-sm ${className}`}>
      <span className="text-gray-500 select-none">{lineNumber}</span>
      <pre className="text-gray-300 whitespace-pre">{highlightSyntax(code)}</pre>
    </div>
  );
}
