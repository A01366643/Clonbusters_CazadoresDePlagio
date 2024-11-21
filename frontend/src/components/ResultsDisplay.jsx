import React from 'react';
import { AlertTriangle, Check } from 'lucide-react';
const ResultsDisplay = ({ results }) => {
  if (!results) return null;
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
    <div className="mt-8 p-6 bg-gray-50 rounded-lg border">
      <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
        {results.isPlagiarism ? (
          <>
            <AlertTriangle className="text-red-500" />
            <span>Alto nivel de similitud detectado</span>
          </>
        ) : (
          <>
            <Check className="text-green-500" />
            <span>Análisis Completado</span>
          </>
        )}
      </h3>
      <div className="grid md:grid-cols-2 gap-6">
        <div className="space-y-2">
          <p className="font-medium">Similitud de Tokens</p>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className="bg-blue-600 h-2.5 rounded-full transition-all duration-500" 
              style={{ width: ${results.tokenOverlap}% }}
            />
          </div>
          <p className="text-sm text-gray-600">{results.tokenOverlap}%</p>
        </div>
        <div className="space-y-2">
          <p className="font-medium">Similitud de Estructura (AST)</p>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className="bg-purple-600 h-2.5 rounded-full transition-all duration-500" 
              style={{ width: ${results.astSimilarity}% }}
            />
          </div>
          <p className="text-sm text-gray-600">{results.astSimilarity}%</p>
        </div>
        <div className="space-y-2">
          <p className="font-medium">Puntuación Global</p>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className={h-2.5 rounded-full transition-all duration-500 ${getProgressBarColor(results.overallPlagiarismScore)}}
              style={{ width: ${results.overallPlagiarismScore}% }}
            />
          </div>
          <p className={text-sm font-medium ${getScoreColor(results.overallPlagiarismScore)}}>
            {results.overallPlagiarismScore}%
          </p>
        </div>
      </div>
      {results.isPlagiarism && (
        <div className="mt-6 p-4 bg-red-50 text-red-700 rounded-lg">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5" />
            <p className="font-medium">Advertencia de Plagio</p>
          </div>
          <p className="mt-2 text-sm">
            Se ha detectado un alto nivel de similitud entre los códigos proporcionados.
            Se recomienda una revisión detallada.
          </p>
        </div>
      )}
    </div>
  );
};
export default ResultsDisplay;
