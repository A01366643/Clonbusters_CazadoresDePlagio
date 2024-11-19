# Frontend/Dockerfile
# Etapa de construcción
FROM node:16-alpine as builder

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de configuración primero
COPY package*.json ./
COPY vite.config.js ./
COPY postcss.config.js ./
COPY tailwind.config.js ./

# Instalar dependencias
RUN npm ci

# Copiar código fuente
COPY src/ ./src/
COPY public/ ./public/
COPY index.html ./

# Construir la aplicación
RUN npm run build

# Etapa de producción
FROM nginx:alpine

# Copiar configuración de nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copiar archivos construidos
COPY --from=builder /app/dist /usr/share/nginx/html

# Exponer puerto
EXPOSE 80

# Iniciar nginx
CMD ["nginx", "-g", "daemon off;"]
