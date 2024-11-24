import React, { useState, useEffect } from 'react';
import { ArrowLeftRight } from 'lucide-react';
import Prism from 'prismjs';
import 'prismjs/themes/prism.css';
import 'prismjs/components/prism-java';

const CodeViewer = ({ originalCode, comparisonCode, similarSegments }) => {
  const [typingOriginal, setTypingOriginal] = useState('');
  const [typingComparison, setTypingComparison] = useState('');
  const [hoveredSegment, setHoveredSegment] = useState(null);
  const [selectedSegment, setSelectedSegment] = useState(null);

  // Efecto de escritura para código original
  useEffect(() => {
    if (originalCode) {
      let currentText = '';
      const lines = originalCode.split('\n');
      let currentLine = 0;
      let currentChar = 0;
      
      const interval = setInterval(() => {
        if (currentLine < lines.length) {
          if (currentChar <= lines[currentLine].length) {
            currentText = lines.slice(0, currentLine).join('\n') + 
                         '\n' + 
                         lines[currentLine].slice(0, currentChar);
            setTypingOriginal(currentText);
            currentChar++;
          } else {
            currentLine++;
            currentChar = 0;
            currentText += '\n';
          }
        } else {
          clearInterval(interval);
          setTypingOriginal(originalCode);
        }
      }, 10);

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
          if (currentChar <= lines[currentLine].length) {
            currentText = lines.slice(0, currentLine).join('\n') + 
                         '\n' + 
                         lines[currentLine].slice(0, currentChar);
            setTypingComparison(currentText);
            currentChar++;
          } else {
            currentLine++;
            currentChar = 0;
            currentText += '\n';
          }
        } else {
          clearInterval(interval);
          setTypingComparison(comparisonCode);
        }
      }, 10);

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

      {/* Información del segmento */}
      {(hoveredSegment || selectedSegment) && (
        <div className="fixed bottom-4 right-4 bg-white text-gray-800 p-4 rounded-lg shadow-lg border border-gray-200">
          <div className="flex items-center gap-2">
            <ArrowLeftRight className="text-blue-600" size={20} />
            <span>
              Similitud: {((hoveredSegment || selectedSegment).similarity * 100).toFixed(1)}%
            </span>
          </div>
          <div className="text-sm text-gray-600 mt-1">
            Tipo: {(hoveredSegment || selectedSegment).type}
          </div>
        </div>
      )}
    </div>
  );
};

export default CodeViewer;
