import React, { useState } from 'react';
import { Upload } from 'lucide-react';
import ResultsDisplay from './ResultsDisplay';

const PlagiarismChecker = () => {
  const [originalCode, setOriginalCode] = useState('');
  const [comparisonCode, setComparisonCode] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const analyzeCode = async () => {
    if (!originalCode || !comparisonCode) {
      setError('Por favor, proporciona ambos códigos para comparar');
      return;
    }

    setLoading(true);
    setError('');

    // Simulamos la llamada al backend
    try {
      // Aquí irá tu llamada real a la API
      setTimeout(() => {
        setResults({
          tokenOverlap: 75.5,
          astSimilarity: 82.3,
          semanticSimilarity: 79.8,
          overallPlagiarismScore: 79.2,
          isPlagiarism: true
        });
        setLoading(false);
      }, 1500);
    } catch (err) {
      setError('Error al analizar el código');
      setLoading(false);
    }
  };

  return (
    <div className="bg-white p-8 rounded-lg shadow-lg">
      <div className="grid md:grid-cols-2 gap-6">
        <div>
          <h2 className="text-xl font-semibold mb-4">Código Original</h2>
          <textarea
            className="w-full h-64 p-4 border rounded-lg font-mono text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={originalCode}
            onChange={(e) => setOriginalCode(e.target.value)}
            placeholder="Pega aquí el código Java original..."
          />
        </div>
        
        <div>
          <h2 className="text-xl font-semibold mb-4">Código a Comparar</h2>
          <textarea
            className="w-full h-64 p-4 border rounded-lg font-mono text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={comparisonCode}
            onChange={(e) => setComparisonCode(e.target.value)}
            placeholder="Pega aquí el código Java a comparar..."
          />
        </div>
      </div>

      {error && (
        <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-lg">
          {error}
        </div>
      )}

      <div className="mt-6 flex justify-center">
        <button
          onClick={analyzeCode}
          disabled={loading}
          className={`
            flex items-center gap-2 px-6 py-3 rounded-lg
            ${loading 
              ? 'bg-gray-400 cursor-not-allowed' 
              : 'bg-blue-600 hover:bg-blue-700'
            }
            text-white font-semibold transition-colors
          `}
        >
          <Upload className="h-5 w-5" />
          {loading ? 'Analizando...' : 'Analizar Código'}
        </button>
      </div>

      {results && <ResultsDisplay results={results} />}
    </div>
  );
};

export default PlagiarismChecker;