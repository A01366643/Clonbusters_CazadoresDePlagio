import React from 'react';
import { AlertTriangle, Check, FileText } from 'lucide-react';

const ComparisonResult = ({ result }) => {
  const getScoreColor = (score) => {
    if (score > 70) return 'text-red-600';
    if (score > 40) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getProgressBarColor = (score) => {
    if (score > 70) return 'bg-red-600';
    if (score > 40) return 'bg-yellow-600';
    return 'bg-green-600';
  };

  return (
    <div className="mb-8 p-6 bg-white rounded-lg shadow-sm border">
      <div className="flex items-center gap-2 mb-6">
        <FileText className="text-blue-500 h-5 w-5" />
        <h3 className="text-lg font-medium">
          Comparación con: {result.fileName}
        </h3>
      </div>

      <div className="space-y-6">
        <div className="space-y-2">
          <p className="font-medium">Similitud de Tokens</p>
          <div className="w-full bg-gray-100 rounded-full h-2.5">
            <div 
              className="bg-blue-600 h-2.5 rounded-full transition-all duration-500" 
              style={{ width: `${result.tokenOverlap}%` }}
            />
          </div>
          <p className="text-sm text-gray-600">{result.tokenOverlap}%</p>
        </div>

        <div className="space-y-2">
          <p className="font-medium">Similitud de Estructura (AST)</p>
          <div className="w-full bg-gray-100 rounded-full h-2.5">
            <div 
              className="bg-purple-600 h-2.5 rounded-full transition-all duration-500" 
              style={{ width: `${result.astSimilarity}%` }}
            />
          </div>
          <p className="text-sm text-gray-600">{result.astSimilarity}%</p>
        </div>

        <div className="space-y-2">
          <p className="font-medium">Puntuación Global</p>
          <div className="w-full bg-gray-100 rounded-full h-2.5">
            <div 
              className={`h-2.5 rounded-full transition-all duration-500 ${getProgressBarColor(result.overallScore)}`}
              style={{ width: `${result.overallScore}%` }}
            />
          </div>
          <p className={`text-sm font-medium ${getScoreColor(result.overallScore)}`}>
            {result.overallScore}%
          </p>
        </div>

        {result.overallScore > 70 && (
          <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-lg">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5" />
              <p className="font-medium">Advertencia de Plagio</p>
            </div>
            <p className="mt-2 text-sm">
              Se ha detectado un alto nivel de similitud con el archivo original.
              Se recomienda una revisión detallada.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

const ResultsDisplay = ({ results }) => {
  if (!results || !results.length) return null;

  return (
    <div className="mt-8">
      <div className="mb-6 flex items-center gap-2">
        <Check className="text-green-500 h-6 w-6" />
        <h2 className="text-xl font-semibold">
          Análisis Completado
        </h2>
      </div>

      <div className="space-y-4">
        {results.map((result, index) => (
          <ComparisonResult key={index} result={result} />
        ))}
      </div>
    </div>
  );
};

export default ResultsDisplay;
