import React, { useState, useRef, useEffect } from 'react';
import { Upload, X, FileText, AlertCircle, AlertTriangle, Check } from 'lucide-react';

// Componente ResultsDisplay integrado
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
              style={{ width: `${results.tokenOverlap}%` }}
            />
          </div>
          <p className="text-sm text-gray-600">{results.tokenOverlap}%</p>
        </div>
        <div className="space-y-2">
          <p className="font-medium">Similitud de Estructura (AST)</p>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className="bg-purple-600 h-2.5 rounded-full transition-all duration-500" 
              style={{ width: `${results.astSimilarity}%` }}
            />
          </div>
          <p className="text-sm text-gray-600">{results.astSimilarity}%</p>
        </div>
        
        <div className="space-y-2">
          <p className="font-medium">Puntuación Global</p>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className={`h-2.5 rounded-full transition-all duration-500 ${getProgressBarColor(results.overallPlagiarismScore)}`}
              style={{ width: `${results.overallPlagiarismScore}%` }}
            />
          </div>
          <p className={`text-sm font-medium ${getScoreColor(results.overallPlagiarismScore)}`}>
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

// Componente principal PlagiarismChecker
const PlagiarismChecker = () => {
  const [originalFile, setOriginalFile] = useState(null);
  const [comparisonFiles, setComparisonFiles] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [dragActive, setDragActive] = useState({ original: false, comparison: false })
  const [apiStatus, setApiStatus] = useState('checking'); // Agregar esta línea
  
  const originalInputRef = useRef(null);
  const comparisonInputRef = useRef(null);

  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
          setApiStatus('ready');
        } else {
          setApiStatus('error');
        }
      } catch (err) {
        console.error('Error checking API status:', err);
        setApiStatus('error');
      }
    };

    checkApiStatus();
  }, []);

  const handleDrag = (e, type) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(prev => ({ ...prev, [type]: true }));
    } else if (e.type === "dragleave") {
      setDragActive(prev => ({ ...prev, [type]: false }));
    }
  };

  const handleOriginalDrop = async (e) => {
    e.preventDefault();
    setDragActive(prev => ({ ...prev, original: false }));
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleOriginalFile(e.dataTransfer.files[0]);
    }
  };

  const handleComparisonDrop = async (e) => {
    e.preventDefault();
    setDragActive(prev => ({ ...prev, comparison: false }));
    
    if (e.dataTransfer.files) {
      handleComparisonFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleOriginalFile = (file) => {
    if (!file.name.endsWith('.java')) {
      setError('Por favor, sube solo archivos Java (.java)');
      return;
    }
    setOriginalFile(file);
    setError('');
  };

  const handleComparisonFiles = (files) => {
    const invalidFiles = files.filter(file => !file.name.endsWith('.java'));
    if (invalidFiles.length > 0) {
      setError('Por favor, sube solo archivos Java (.java)');
      return;
    }
    setComparisonFiles(prev => [...prev, ...files]);
    setError('');
  };

  const removeComparisonFile = (index) => {
    setComparisonFiles(prev => prev.filter((_, i) => i !== index));
  };

  // Primero, agrega esta constante en la parte superior del archivo, después de los imports
  const API_URL = import.meta.env.VITE_API_URL;
  
  // Reemplaza la función analyzeCode actual con esta versión:
  const analyzeCode = async () => {
    const formData = new FormData();
    formData.append('original', originalFile);
    comparisonFiles.forEach((file) => {
      formData.append('comparison_files', file);
    });
  
    try {
      const response = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        body: formData,
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Error al analizar el código');
      }
  
      const data = await response.json();
      console.log('Datos recibidos del backend:', data);
      
      // Asegurar que los valores sean números
      const tokenSim = parseFloat(data.token_similarity);
      const astSim = parseFloat(data.ast_similarity);
      
      // Calcular el overall score como el promedio
      const overallScore = (tokenSim + astSim) / 2;
      
      const mappedResults = {
        tokenOverlap: tokenSim,
        astSimilarity: astSim,
        overallPlagiarismScore: overallScore,
        isPlagiarism: overallScore > 70 
      };
  
      console.log('Resultados mapeados:', mappedResults);
      setResults(mappedResults);
    } catch (error) {
      console.error('Error:', error);
      setError('Error al analizar el código');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white p-8 rounded-lg shadow-lg">
      {apiStatus === 'error' && (
        <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-lg flex items-center gap-2">
          <AlertCircle className="h-5 w-5" />
          No se puede conectar con el servicio de análisis
        </div>
      )}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Área para código original */}
        <div>
          <h2 className="text-xl font-semibold mb-4">Código Original</h2>
          <div
            className={`
              w-full h-64 border-2 border-dashed rounded-lg
              ${dragActive.original ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}
              transition-colors duration-200 ease-in-out
              flex flex-col items-center justify-center
              cursor-pointer
            `}
            onClick={() => originalInputRef.current?.click()}
            onDragEnter={e => handleDrag(e, 'original')}
            onDragLeave={e => handleDrag(e, 'original')}
            onDragOver={e => handleDrag(e, 'original')}
            onDrop={handleOriginalDrop}
          >
            <input
              ref={originalInputRef}
              type="file"
              className="hidden"
              accept=".java"
              onChange={e => handleOriginalFile(e.target.files[0])}
            />
            
            {originalFile ? (
              <div className="flex items-center gap-2 p-4">
                <FileText className="text-blue-500" />
                <span className="font-medium">{originalFile.name}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setOriginalFile(null);
                  }}
                  className="p-1 hover:bg-gray-100 rounded-full"
                >
                  <X className="h-4 w-4 text-gray-500" />
                </button>
              </div>
            ) : (
              <>
                <Upload className="h-10 w-10 text-blue-500 mb-2" />
                <p className="text-sm text-gray-600">
                  Arrastra y suelta tu archivo Java aquí o haz clic para seleccionarlo
                </p>
              </>
            )}
          </div>
        </div>
        
        {/* Área para códigos a comparar */}
        <div>
          <h2 className="text-xl font-semibold mb-4">Códigos a Comparar</h2>
          <div
            className={`
              w-full h-64 border-2 border-dashed rounded-lg
              ${dragActive.comparison ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}
              transition-colors duration-200 ease-in-out
              overflow-y-auto
            `}
            onClick={() => comparisonInputRef.current?.click()}
            onDragEnter={e => handleDrag(e, 'comparison')}
            onDragLeave={e => handleDrag(e, 'comparison')}
            onDragOver={e => handleDrag(e, 'comparison')}
            onDrop={handleComparisonDrop}
          >
            <input
              ref={comparisonInputRef}
              type="file"
              className="hidden"
              accept=".java"
              multiple
              onChange={e => handleComparisonFiles(Array.from(e.target.files))}
            />
            
            {comparisonFiles.length > 0 ? (
              <div className="p-4 space-y-2">
                {comparisonFiles.map((file, index) => (
                  <div key={index} className="flex items-center justify-between bg-gray-50 p-2 rounded">
                    <div className="flex items-center gap-2">
                      <FileText className="text-blue-500" />
                      <span className="font-medium">{file.name}</span>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        removeComparisonFile(index);
                      }}
                      className="p-1 hover:bg-gray-100 rounded-full"
                    >
                      <X className="h-4 w-4 text-gray-500" />
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center">
                <Upload className="h-10 w-10 text-blue-500 mb-2" />
                <p className="text-sm text-gray-600 text-center">
                  Arrastra y suelta tus archivos Java aquí<br />
                  o haz clic para seleccionarlos
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {error && (
        <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-lg flex items-center gap-2">
          <AlertCircle className="h-5 w-5" />
          {error}
        </div>
      )}

      <div className="mt-6 flex justify-center">
        <button
          onClick={analyzeCode}
          disabled={loading || apiStatus !== 'ready'}
          className={`
            flex items-center gap-2 px-6 py-3 rounded-lg
            ${loading || apiStatus !== 'ready'
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
