# Development Dockerfile for frontend
FROM node:22.9.0

WORKDIR /app

# Copy package files
COPY package.json ./
COPY yarn.lock ./

# Install dependencies
RUN yarn install --frozen-lockfile

# Copy source code
COPY . .

# Expose Vite dev server port
EXPOSE 5173

# Start development server
CMD ["yarn", "dev", "--host", "0.0.0.0", "--port", "5173"]
