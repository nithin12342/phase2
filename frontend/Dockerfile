
# Stage 1: Build
FROM node:18-alpine as build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

# Stage 2: Serve
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
# Add custom nginx config if trying to handle React Router (SPA) 
# For now, default is fine as we only have one route mostly.
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
