import React, { useState, useRef, useEffect } from 'react';
import { Upload, X, FileText, AlertTriangle, Check, AlertCircle } from 'lucide-react';

// Definimos la URL base directamente - ajusta esto según tu entorno
const API_URL = import.meta.env.VITE_API_URL;

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
              style={{ width: `${results.token_similarity}%` }}
            />
          </div>
          <p className="text-sm text-gray-600">{results.token_similarity}%</p>
        </div>
        <div className="space-y-2">
          <p className="font-medium">Similitud de Estructura (AST)</p>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className="bg-purple-600 h-2.5 rounded-full transition-all duration-500" 
              style={{ width: `${results.ast_similarity}%` }}
            />
          </div>
          <p className="text-sm text-gray-600">{results.ast_similarity}%</p>
        </div>
        <div className="space-y-2">
          <p className="font-medium">Similitud Semántica</p>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className="bg-indigo-600 h-2.5 rounded-full transition-all duration-500" 
              style={{ width: `${results.semantic_similarity}%` }}
            />
          </div>
          <p className="text-sm text-gray-600">{results.semantic_similarity}%</p>
        </div>
        <div className="space-y-2">
          <p className="font-medium">Puntuación Global</p>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className={`h-2.5 rounded-full transition-all duration-500 ${getProgressBarColor(results.overall_score)}`}
              style={{ width: `${results.overall_score}%` }}
            />
          </div>
          <p className={`text-sm font-medium ${getScoreColor(results.overall_score)}`}>
            {results.overall_score}%
          </p>
        </div>
      </div>
      {results.is_plagiarism && (
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

const PlagiarismChecker = () => {
  const [originalFile, setOriginalFile] = useState(null);
  const [comparisonFiles, setComparisonFiles] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState('checking');
  const [dragActive, setDragActive] = useState({ original: false, comparison: false });
  
  const originalInputRef = useRef(null);
  const comparisonInputRef = useRef(null);

  // Verificar que el backend esté funcionando
  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
          const data = await response.json();
          setApiStatus(data.models_loaded ? 'ready' : 'no-models');
        } else {
          setApiStatus('error');
        }
      } catch (err) {
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

  const analyzeCode = async () => {
    if (!originalFile || comparisonFiles.length === 0) {
      setError('Por favor, sube el código original y al menos un archivo para comparar');
      return;
    }

    if (apiStatus !== 'ready') {
      setError('El servicio de análisis no está disponible');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('original', originalFile);
      comparisonFiles.forEach((file) => {
        formData.append('comparison_files', file);
      });

      const response = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Error al analizar el código');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const renderApiStatus = () => {
    switch (apiStatus) {
      case 'checking':
        return (
          <div className="text-blue-600 text-sm flex items-center gap-2 mb-4">
            <AlertCircle className="h-4 w-4" />
            Verificando conexión con el servicio...
          </div>
        );
      case 'error':
        return (
          <div className="text-red-600 text-sm flex items-center gap-2 mb-4">
            <AlertTriangle className="h-4 w-4" />
            No se puede conectar con el servicio de análisis
          </div>
        );
      case 'no-models':
        return (
          <div className="text-yellow-600 text-sm flex items-center gap-2 mb-4">
            <AlertTriangle className="h-4 w-4" />
            Servicio conectado pero modelos no disponibles
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="bg-white p-8 rounded-lg shadow-lg">
      {renderApiStatus()}
      
      <div className="grid md:grid-cols-2 gap-6">
        <div>
          <h2 className="text-xl font-semibold mb-4">Código Original</h2>
          <div
            className={`
              w-full h-64 border-2 border-dashed rounded-lg
              ${dragActive.original ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}
              transition-colors duration-200 ease-in-out
              flex flex-col items-center justify-center
              cursor-pointer
              ${apiStatus !== 'ready' ? 'opacity-50' : ''}
            `}
            onClick={() => apiStatus === 'ready' && originalInputRef.current?.click()}
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
              disabled={apiStatus !== 'ready'}
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
                  disabled={apiStatus !== 'ready'}
                >
                  <X className="h-4 w-4 text-gray-500" />
                </button>
              </div>
            ) : (
              <>
                <Upload className="h-10 w-10 text-blue-500 mb-2" />
                <p className="text-sm text-gray-600 text-center">
                  Arrastra y suelta tu archivo Java aquí<br />
                  o haz clic para seleccionarlo
                </p>
              </>
            )}
          </div>
        </div>
        
        <div>
          <h2 className="text-xl font-semibold mb-4">Códigos a Comparar</h2>
          <div
            className={`
              w-full h-64 border-2 border-dashed rounded-lg
              ${dragActive.comparison ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}
              transition-colors duration-200 ease-in-out
              overflow-y-auto
              ${apiStatus !== 'ready' ? 'opacity-50' : ''}
            `}
            onClick={() => apiStatus === 'ready' && comparisonInputRef.current?.click()}
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
              disabled={apiStatus !== 'ready'}
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
                      disabled={apiStatus !== 'ready'}
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
              ? '
