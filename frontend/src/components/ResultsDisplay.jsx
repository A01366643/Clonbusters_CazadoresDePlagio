import React from 'react';
import { AlertTriangle, Check, ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';

const ResultsDisplay = ({ results }) => {
  const [expandedFile, setExpandedFile] = useState(null);
  
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

  const ComparisonMetrics = ({ data, fileName }) => (
    <div className="space-y-4 bg-white p-4 rounded-lg shadow-sm">
      <div className="font-medium text-gray-700">{fileName}</div>
      <div className="space-y-2">
        <p className="text-sm font-medium">Similitud de Tokens</p>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div 
            className="bg-blue-600 h-2.5 rounded-full transition-all duration-500" 
            style={{ width: `${data.tokenOverlap}%` }}
          />
        </div>
        <p className="text-sm text-gray-600">{data.tokenOverlap}%</p>
      </div>
      
      <div className="space-y-2">
        <p className="text-sm font-medium">Similitud de Estructura (AST)</p>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div 
            className="bg-purple-600 h-2.5 rounded-full transition-all duration-500" 
            style={{ width: `${data.astSimilarity}%` }}
          />
        </div>
        <p className="text-sm text-gray-600">{data.astSimilarity}%</p>
      </div>
      
      <div className="space-y-2">
        <p className="text-sm font-medium">Puntuación Global</p>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div 
            className={`h-2.5 rounded-full transition-all duration-500 ${getProgressBarColor(data.overallPlagiarismScore)}`}
            style={{ width: `${data.overallPlagiarismScore}%` }}
          />
        </div>
        <p className={`text-sm font-medium ${getScoreColor(data.overallPlagiarismScore)}`}>
          {data.overallPlagiarismScore}%
        </p>
      </div>
    </div>
  );

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

      {/* Overall Summary */}
      <div className="mb-6">
        <h4 className="text-lg font-medium mb-3">Resumen General</h4>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <p className="text-sm font-medium">Promedio de Similitud Global</p>
            <div className="mt-2">
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div 
                  className={`h-2.5 rounded-full transition-all duration-500 ${getProgressBarColor(results.overallPlagiarismScore)}`}
                  style={{ width: `${results.overallPlagiarismScore}%` }}
                />
              </div>
              <p className={`mt-1 text-sm font-medium ${getScoreColor(results.overallPlagiarismScore)}`}>
                {results.overallPlagiarismScore}%
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Individual Comparisons */}
      <div className="space-y-4">
        <h4 className="text-lg font-medium">Comparaciones Individuales</h4>
        {results.comparisons && results.comparisons.map((comparison, index) => (
          <div key={index} className="border rounded-lg overflow-hidden">
            <button 
              className="w-full px-4 py-3 flex items-center justify-between bg-white hover:bg-gray-50"
              onClick={() => setExpandedFile(expandedFile === index ? null : index)}
            >
              <span className="font-medium">{comparison.fileName}</span>
              {expandedFile === index ? <ChevronDown className="h-5 w-5" /> : <ChevronRight className="h-5 w-5" />}
            </button>
            
            {expandedFile === index && (
              <div className="p-4 border-t bg-gray-50">
                <ComparisonMetrics data={comparison} fileName={comparison.fileName} />
              </div>
            )}
          </div>
        ))}
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
