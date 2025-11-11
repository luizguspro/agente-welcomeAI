#!/usr/bin/env node

/**
 * Fix Dependencies Script
 * Execute este script dentro da pasta greencardclub-app
 */

const fs = require('fs');
const { execSync } = require('child_process');

console.log(`
╔═══════════════════════════════════════════════════════╗
║  🔧 CORRIGINDO DEPENDÊNCIAS DO PROJETO                ║
╚═══════════════════════════════════════════════════════╝
`);

// Package.json completo com todas as dependências
const packageJson = {
  "name": "greencardclub-app",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.1.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@radix-ui/react-avatar": "^1.0.4",
    "@radix-ui/react-checkbox": "^1.0.4",
    "@radix-ui/react-dialog": "^1.0.5",
    "@radix-ui/react-label": "^2.0.2",
    "@radix-ui/react-progress": "^1.0.3",
    "@radix-ui/react-select": "^2.0.0",
    "@radix-ui/react-slot": "^1.0.2",
    "@radix-ui/react-tabs": "^1.0.4",
    "@radix-ui/react-toast": "^1.1.5",
    "@hookform/resolvers": "^3.3.4",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.0",
    "date-fns": "^3.3.1",
    "framer-motion": "^11.0.3",
    "html2canvas": "^1.4.1",
    "lucide-react": "^0.314.0",
    "qrcode": "^1.5.3",
    "react-dropzone": "^14.2.3",
    "react-hook-form": "^7.49.3",
    "react-webcam": "^7.2.0",
    "tailwind-merge": "^2.2.0",
    "tailwindcss-animate": "^1.0.7",
    "zod": "^3.22.4",
    "zustand": "^4.5.0"
  },
  "devDependencies": {
    "@types/node": "^20.11.5",
    "@types/qrcode": "^1.5.5",
    "@types/react": "^18.2.48",
    "@types/react-dom": "^18.2.18",
    "autoprefixer": "^10.4.17",
    "eslint": "^8.56.0",
    "eslint-config-next": "14.1.0",
    "postcss": "^8.4.33",
    "tailwindcss": "^3.4.1",
    "typescript": "^5.3.3"
  }
};

// Salvar o package.json corrigido
console.log('📝 Atualizando package.json...');
fs.writeFileSync('package.json', JSON.stringify(packageJson, null, 2));
console.log('✅ package.json atualizado!');

// Limpar cache do npm
console.log('\n🧹 Limpando cache do npm...');
try {
  execSync('npm cache clean --force', { stdio: 'inherit' });
} catch (error) {
  console.log('⚠️ Não foi possível limpar o cache, continuando...');
}

// Remover node_modules e package-lock.json
console.log('\n🗑️ Removendo arquivos antigos...');
if (fs.existsSync('node_modules')) {
  console.log('Removendo node_modules...');
  if (process.platform === 'win32') {
    execSync('rmdir /s /q node_modules', { shell: true });
  } else {
    execSync('rm -rf node_modules');
  }
}

if (fs.existsSync('package-lock.json')) {
  console.log('Removendo package-lock.json...');
  fs.unlinkSync('package-lock.json');
}

// Instalar dependências
console.log('\n📦 Instalando todas as dependências...\n');
console.log('Isso pode levar alguns minutos...\n');

try {
  execSync('npm install', { stdio: 'inherit' });
  console.log('\n✅ Todas as dependências foram instaladas com sucesso!');
} catch (error) {
  console.error('\n❌ Erro ao instalar dependências');
  console.log('\nTente executar manualmente:');
  console.log('npm install');
}

console.log(`
╔═══════════════════════════════════════════════════════╗
║  ✨ PROJETO CORRIGIDO!                                ║
╚═══════════════════════════════════════════════════════╝

🚀 Agora você pode executar:

   npm run dev

E acessar: http://localhost:3000

Se ainda houver problemas, tente:
   1. npm install next react react-dom
   2. npm install
   3. npm run dev
`);