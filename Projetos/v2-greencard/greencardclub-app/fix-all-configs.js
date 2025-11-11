#!/usr/bin/env node

/**
 * Fix All Configurations for Next.js 14
 * Execute este script na pasta greencardclub-app
 */

const fs = require('fs');
const path = require('path');

console.log(`
╔═══════════════════════════════════════════════════════╗
║  🔧 CORRIGINDO TODAS AS CONFIGURAÇÕES                 ║
╚═══════════════════════════════════════════════════════╝
`);

// 1. tsconfig.json
const tsconfigContent = {
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [
      {
        "name": "next"
      }
    ],
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
};

console.log('📝 Criando tsconfig.json...');
fs.writeFileSync('tsconfig.json', JSON.stringify(tsconfigContent, null, 2));

// 2. next.config.js
const nextConfigContent = `/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  images: {
    domains: ['localhost'],
    unoptimized: true,
  },
  experimental: {
    serverActions: true,
  },
}

module.exports = nextConfig
`;

console.log('📝 Criando next.config.js...');
fs.writeFileSync('next.config.js', nextConfigContent);

// 3. tailwind.config.ts
const tailwindContent = `import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        greencard: {
          primary: '#084734',
          secondary: '#CEF17B',
          tertiary: '#CEEDB2',
          background: '#F9FFF3',
        }
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'float': 'float 3s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        slideUp: {
          from: { transform: 'translateY(10px)', opacity: '0' },
          to: { transform: 'translateY(0)', opacity: '1' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
    },
  },
  plugins: [],
}
export default config
`;

console.log('📝 Criando tailwind.config.ts...');
fs.writeFileSync('tailwind.config.ts', tailwindContent);

// 4. postcss.config.js
const postcssContent = `module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
`;

console.log('📝 Criando postcss.config.js...');
fs.writeFileSync('postcss.config.js', postcssContent);

// 5. next-env.d.ts
const nextEnvContent = `/// <reference types="next" />
/// <reference types="next/image-types/global" />

// NOTE: This file should not be edited
// see https://nextjs.org/docs/basic-features/typescript for more information.
`;

console.log('📝 Criando next-env.d.ts...');
fs.writeFileSync('next-env.d.ts', nextEnvContent);

// 6. .eslintrc.json
const eslintContent = {
  "extends": "next/core-web-vitals"
};

console.log('📝 Criando .eslintrc.json...');
fs.writeFileSync('.eslintrc.json', JSON.stringify(eslintContent, null, 2));

// 7. package.json atualizado
const packageJsonContent = {
  "name": "greencardclub-app",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "type-check": "tsc --noEmit"
  },
  "dependencies": {
    "next": "14.1.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@hookform/resolvers": "^3.3.4",
    "@radix-ui/react-avatar": "^1.0.4",
    "@radix-ui/react-checkbox": "^1.0.4",
    "@radix-ui/react-dialog": "^1.0.5",
    "@radix-ui/react-label": "^2.0.2",
    "@radix-ui/react-progress": "^1.0.3",
    "@radix-ui/react-select": "^2.0.0",
    "@radix-ui/react-slot": "^1.0.2",
    "@radix-ui/react-tabs": "^1.0.4",
    "@radix-ui/react-toast": "^1.1.5",
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

console.log('📝 Atualizando package.json...');
fs.writeFileSync('package.json', JSON.stringify(packageJsonContent, null, 2));

// Verificar se as pastas necessárias existem
const requiredDirs = ['app', 'components', 'lib', 'public'];
requiredDirs.forEach(dir => {
  if (!fs.existsSync(dir)) {
    console.log(`📁 Criando pasta ${dir}...`);
    fs.mkdirSync(dir, { recursive: true });
  }
});

// Criar arquivo app/globals.css se não existir
if (!fs.existsSync('app/globals.css')) {
  const globalsCss = `@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --primary: #084734;
  --secondary: #CEF17B;
  --tertiary: #CEEDB2;
  --background: #F9FFF3;
}

body {
  background-color: var(--background);
  color: var(--primary);
}
`;
  console.log('📝 Criando app/globals.css...');
  fs.writeFileSync('app/globals.css', globalsCss);
}

// Criar um app/page.tsx básico se não existir
if (!fs.existsSync('app/page.tsx')) {
  const pageContent = `export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <h1 className="text-4xl font-bold">Greencardclub</h1>
      <p className="mt-4 text-xl">Sistema de Carteirinha Digital</p>
    </main>
  );
}
`;
  console.log('📝 Criando app/page.tsx...');
  fs.writeFileSync('app/page.tsx', pageContent);
}

// Criar app/layout.tsx se não existir
if (!fs.existsSync('app/layout.tsx')) {
  const layoutContent = `import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Greencardclub',
  description: 'Sistema de Carteirinha Digital',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="pt-BR">
      <body className={inter.className}>{children}</body>
    </html>
  )
}
`;
  console.log('📝 Criando app/layout.tsx...');
  fs.writeFileSync('app/layout.tsx', layoutContent);
}

console.log(`
╔═══════════════════════════════════════════════════════╗
║  ✅ TODAS AS CONFIGURAÇÕES FORAM CORRIGIDAS!          ║
╚═══════════════════════════════════════════════════════╝

📌 Agora execute:

1. npm install
2. npm run dev

🌐 Acesse: http://localhost:3000

Se ainda houver erros, tente:
   - Deletar node_modules e package-lock.json
   - npm cache clean --force  
   - npm install
   - npm run dev

🚀 Projeto está pronto para desenvolvimento!
`);