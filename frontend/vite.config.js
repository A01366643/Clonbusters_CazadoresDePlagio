import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/Clonbusters_CazadoresDePlagio/', // Debe coincidir exactamente con el nombre de tu repositorio
})
