import React, { useState, useEffect } from 'react';
import Prism from 'prismjs';
import 'prismjs/themes/prism.css';
import 'prismjs/components/prism-java';

const CodeViewer = ({ originalCode, comparisonCode }) => {
  const [typingOriginal, setTypingOriginal] = useState('');
  const [typingComparison, setTypingComparison] = useState('');

  const CHARS_PER_ITERATION = 5; // Procesa más caracteres por iteración
  const TYPING_SPEED = 1; // Menor número = más rápido

  // Efecto de escritura para código original
  useEffect(() => {
    if (originalCode) {
      let currentText = '';
      const lines = originalCode.split('\n');
      let currentLine = 0;
      let currentChar = 0;
      
      const interval = setInterval(() => {
        if (currentLine < lines.length) {
          // Procesa múltiples caracteres por iteración
          for (let i = 0; i < CHARS_PER_ITERATION; i++) {
            if (currentChar <= lines[currentLine].length) {
              currentText = lines.slice(0, currentLine).join('\n') + 
                           '\n' + 
                           lines[currentLine].slice(0, currentChar);
              currentChar += 1;
            } else {
              currentLine++;
              currentChar = 0;
              currentText += '\n';
              break;
            }
          }
          setTypingOriginal(currentText);
        } else {
          clearInterval(interval);
          setTypingOriginal(originalCode);
        }
      }, TYPING_SPEED);

      return () => clearInterval(interval);
    }
  }, [originalCode]);

  // Efecto de escritura para código de comparación
  useEffect(() => {
    if (comparisonCode) {
      let currentText = '';
      const lines = comparisonCode.split('\n');
      let currentLine = 0;
      let currentChar = 0;
      
      const interval = setInterval(() => {
        if (currentLine < lines.length) {
          // Procesa múltiples caracteres por iteración
          for (let i = 0; i < CHARS_PER_ITERATION; i++) {
            if (currentChar <= lines[currentLine].length) {
              currentText = lines.slice(0, currentLine).join('\n') + 
                           '\n' + 
                           lines[currentLine].slice(0, currentChar);
              currentChar += 1;
            } else {
              currentLine++;
              currentChar = 0;
              currentText += '\n';
              break;
            }
          }
          setTypingComparison(currentText);
        } else {
          clearInterval(interval);
          setTypingComparison(comparisonCode);
        }
      }, TYPING_SPEED);

      return () => clearInterval(interval);
    }
  }, [comparisonCode]);

  // Efecto para aplicar syntax highlighting
  useEffect(() => {
    Prism.highlightAll();
  }, [typingOriginal, typingComparison]);

  return (
    <div className="mt-8 bg-white p-6 rounded-lg border border-gray-200">
      <h2 className="text-xl font-semibold mb-4 text-gray-800">Análisis de Código</h2>
      <div className="grid md:grid-cols-2 gap-6">
        {/* Código Original */}
        <div className="relative">
          <h3 className="text-lg font-medium mb-2 text-gray-700">Código Original</h3>
          <div className="bg-white rounded-lg overflow-auto max-h-[600px] border border-gray-200">
            <pre className="p-4 m-0 min-h-[200px]" style={{ background: 'white' }}>
              <code className="language-java" style={{ background: 'white' }}>
                {typingOriginal || ' '}
              </code>
            </pre>
          </div>
        </div>

        {/* Código a Comparar */}
        <div className="relative">
          <h3 className="text-lg font-medium mb-2 text-gray-700">Código a Comparar</h3>
          <div className="bg-white rounded-lg overflow-auto max-h-[600px] border border-gray-200">
            <pre className="p-4 m-0 min-h-[200px]" style={{ background: 'white' }}>
              <code className="language-java" style={{ background: 'white' }}>
                {typingComparison || ' '}
              </code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CodeViewer;