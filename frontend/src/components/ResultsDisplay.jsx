import React from 'react';
import { AlertTriangle, Check, FileText } from 'lucide-react';

const FileResultsDisplay = ({ fileResult }) => {
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
    <div className="p-4 bg-white rounded-lg border mb-4">
      <div className="flex items-center gap-2 mb-4">
        <FileText className="text-blue-500" />
        <h4 className="text-lg font-semibold">{fileResult.fileName}</h4>
      </div>
      
      <div className="space-y-4">
        <div className="space-y-2">
          <p className="font-medium">Similitud de Tokens</p>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className="bg-blue-600 h-2.5 rounded-full transition-all duration-500" 
              style={{ width: `${fileResult.tokenOverlap}%` }}
            />
          </div>
          <p className="text-sm text-gray-600">{fileResult.tokenOverlap}%</p>
        </div>
        
        <div className="space-y-2">
          <p className="font-medium">Similitud de Estructura (AST)</p>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className="bg-purple-600 h-2.5 rounded-full transition-all duration-500" 
              style={{ width: `${fileResult.astSimilarity}%` }}
            />
          </div>
          <p className="text-sm text-gray-600">{fileResult.astSimilarity}%</p>
        </div>
        
        <div className="space-y-2">
          <p className="font-medium">Puntuación Global</p>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className={`h-2.5 rounded-full transition-all duration-500 ${getProgressBarColor(fileResult.overallScore)}`}
              style={{ width: `${fileResult.overallScore}%` }}
            />
          </div>
          <p className={`text-sm font-medium ${getScoreColor(fileResult.overallScore)}`}>
            {fileResult.overallScore}%
          </p>
        </div>
      </div>

      {fileResult.isPlagiarism && (
        <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-lg">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            <p className="font-medium text-sm">Alto nivel de similitud detectado</p>
          </div>
        </div>
      )}
    </div>
  );
};

const DetailedResultsDisplay = ({ results }) => {
  if (!results || !results.fileResults || results.fileResults.length === 0) return null;

  return (
    <div className="mt-8 p-6 bg-gray-50 rounded-lg border">
      <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
        <Check className="text-green-500" />
        <span>Resultados del Análisis</span>
      </h3>

      <div className="space-y-2 mb-6">
        <h4 className="font-medium">Archivo Original</h4>
        <div className="p-2 bg-blue-50 rounded-lg flex items-center gap-2">
          <FileText className="text-blue-500" />
          <span className="text-sm">{results.originalFileName}</span>
        </div>
      </div>

      <div className="space-y-4">
        <h4 className="font-medium">Comparaciones</h4>
        {results.fileResults.map((fileResult, index) => (
          <FileResultsDisplay key={index} fileResult={fileResult} />
        ))}
      </div>

      {results.fileResults.some(r => r.isPlagiarism) && (
        <div className="mt-6 p-4 bg-red-50 text-red-700 rounded-lg">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5" />
            <p className="font-medium">Advertencia de Plagio</p>
          </div>
          <p className="mt-2 text-sm">
            Se ha detectado un alto nivel de similitud en uno o más archivos.
            Se recomienda una revisión detallada.
          </p>
        </div>
      )}
    </div>
  );
};

export default DetailedResultsDisplay;
