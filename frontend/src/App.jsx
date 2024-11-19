import React from 'react'
import PlagiarismChecker from './components/PlagiarismChecker'

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-blue-600 text-white p-4">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-2xl font-bold">Clonbusters</h1>
          <p className="text-sm">Cazadores de Plagio</p>
        </div>
      </nav>
      
      <main className="max-w-6xl mx-auto mt-8">
        <PlagiarismChecker />
      </main>
      
      <footer className="mt-12 py-6 bg-gray-100">
        <div className="max-w-6xl mx-auto text-center text-gray-600">
          © 2024 Clonbusters - Todos los derechos reservados
        </div>
      </footer>
    </div>
  )
}

export default App