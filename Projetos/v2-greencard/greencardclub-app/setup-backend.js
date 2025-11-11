#!/usr/bin/env node

/**
 * 🚀 GREENCARDCLUB - Setup Backend Completo
 * Este script cria toda estrutura de backend automaticamente
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log(`
╔═══════════════════════════════════════════════════════╗
║     🚀 GREENCARDCLUB - SETUP BACKEND                  ║
║     Criando estrutura completa...                     ║
╚═══════════════════════════════════════════════════════╝
`);

// 1. CRIAR ESTRUTURA DE PASTAS
console.log('\n📁 Criando estrutura de pastas...\n');

const folders = [
  'app/api/users/register',
  'app/api/cards/create',
  'app/api/cards/list',
  'app/api/auth/login',
  'app/api/appointments/create',
  'app/api/appointments/list',
  'lib',
  'prisma'
];

folders.forEach(folder => {
  const folderPath = path.join(process.cwd(), folder);
  if (!fs.existsSync(folderPath)) {
    fs.mkdirSync(folderPath, { recursive: true });
    console.log(`✅ Criada: ${folder}`);
  } else {
    console.log(`⏭️  Já existe: ${folder}`);
  }
});

// 2. CRIAR ARQUIVO PRISMA SCHEMA
console.log('\n📋 Criando schema do banco de dados...\n');

const prismaSchema = `generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id            String      @id @default(cuid())
  email         String      @unique
  password      String
  name          String
  cpf           String      @unique
  phone         String?
  birthDate     DateTime?
  userType      UserType    @default(PATIENT)
  createdAt     DateTime    @default(now())
  updatedAt     DateTime    @updatedAt
  
  card          Card?
  appointments  Appointment[] @relation("PatientAppointments")
  doctorAppointments Appointment[] @relation("DoctorAppointments")
}

model Card {
  id            String      @id @default(cuid())
  userId        String      @unique
  cardNumber    String      @unique
  photoUrl      String?
  validUntil    DateTime
  status        CardStatus  @default(ACTIVE)
  createdAt     DateTime    @default(now())
  
  user          User        @relation(fields: [userId], references: [id])
}

model Appointment {
  id            String      @id @default(cuid())
  patientId     String
  doctorId      String
  date          DateTime
  time          String
  status        AppointmentStatus @default(PENDING)
  notes         String?
  createdAt     DateTime    @default(now())
  
  patient       User        @relation("PatientAppointments", fields: [patientId], references: [id])
  doctor        User        @relation("DoctorAppointments", fields: [doctorId], references: [id])
}

enum UserType {
  PATIENT
  DOCTOR
}

enum CardStatus {
  ACTIVE
  INACTIVE
  EXPIRED
}

enum AppointmentStatus {
  PENDING
  CONFIRMED
  CANCELLED
  COMPLETED
}`;

fs.writeFileSync('prisma/schema.prisma', prismaSchema);
console.log('✅ prisma/schema.prisma criado!');

// 3. CRIAR ARQUIVO DE CONEXÃO PRISMA
console.log('\n🔌 Criando conexão Prisma...\n');

const prismaClient = `import { PrismaClient } from '@prisma/client';

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

export const prisma = globalForPrisma.prisma ?? new PrismaClient();

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = prisma;`;

fs.writeFileSync('lib/prisma.ts', prismaClient);
console.log('✅ lib/prisma.ts criado!');

// 4. CRIAR API ROUTES

console.log('\n🛠️ Criando API Routes...\n');

// API - Register User
const registerAPI = `import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';
import bcrypt from 'bcryptjs';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { email, password, name, cpf, phone, birthDate, userType } = body;

    // Verificar se usuário já existe
    const existingUser = await prisma.user.findFirst({
      where: {
        OR: [
          { email },
          { cpf }
        ]
      }
    });

    if (existingUser) {
      return NextResponse.json(
        { error: 'Usuário já cadastrado' },
        { status: 400 }
      );
    }

    // Hash da senha
    const hashedPassword = await bcrypt.hash(password, 10);

    // Criar usuário
    const user = await prisma.user.create({
      data: {
        email,
        password: hashedPassword,
        name,
        cpf,
        phone,
        birthDate: birthDate ? new Date(birthDate) : null,
        userType: userType || 'PATIENT',
      },
    });

    return NextResponse.json({
      id: user.id,
      email: user.email,
      name: user.name,
      userType: user.userType
    });
  } catch (error) {
    console.error('Erro ao criar usuário:', error);
    return NextResponse.json(
      { error: 'Erro ao criar usuário' },
      { status: 500 }
    );
  }
}`;

fs.writeFileSync('app/api/users/register/route.ts', registerAPI);
console.log('✅ app/api/users/register/route.ts criado!');

// API - Create Card
const createCardAPI = `import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { userId, photoUrl } = body;

    // Verificar se usuário existe
    const user = await prisma.user.findUnique({
      where: { id: userId },
      include: { card: true }
    });

    if (!user) {
      return NextResponse.json(
        { error: 'Usuário não encontrado' },
        { status: 404 }
      );
    }

    // Verificar se já tem carteirinha
    if (user.card) {
      return NextResponse.json(
        { error: 'Usuário já possui carteirinha' },
        { status: 400 }
      );
    }

    // Gerar número único da carteirinha
    const cardNumber = \`GC\${Date.now().toString().slice(-8)}\`;
    
    // Criar carteirinha válida por 1 ano
    const validUntil = new Date();
    validUntil.setFullYear(validUntil.getFullYear() + 1);

    const card = await prisma.card.create({
      data: {
        userId,
        cardNumber,
        photoUrl,
        validUntil,
        status: 'ACTIVE'
      },
      include: {
        user: {
          select: {
            name: true,
            cpf: true,
            email: true
          }
        }
      }
    });

    return NextResponse.json(card);
  } catch (error) {
    console.error('Erro ao criar carteirinha:', error);
    return NextResponse.json(
      { error: 'Erro ao criar carteirinha' },
      { status: 500 }
    );
  }
}`;

fs.writeFileSync('app/api/cards/create/route.ts', createCardAPI);
console.log('✅ app/api/cards/create/route.ts criado!');

// API - List Cards
const listCardsAPI = `import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';

export async function GET() {
  try {
    const cards = await prisma.card.findMany({
      include: {
        user: {
          select: {
            name: true,
            email: true,
            cpf: true,
          }
        }
      },
      orderBy: {
        createdAt: 'desc'
      }
    });

    return NextResponse.json(cards);
  } catch (error) {
    console.error('Erro ao listar carteirinhas:', error);
    return NextResponse.json(
      { error: 'Erro ao listar carteirinhas' },
      { status: 500 }
    );
  }
}`;

fs.writeFileSync('app/api/cards/list/route.ts', listCardsAPI);
console.log('✅ app/api/cards/list/route.ts criado!');

// API - Login
const loginAPI = `import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';
import bcrypt from 'bcryptjs';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { email, password } = body;

    const user = await prisma.user.findUnique({
      where: { email },
      include: { card: true }
    });

    if (!user) {
      return NextResponse.json(
        { error: 'Usuário não encontrado' },
        { status: 401 }
      );
    }

    const passwordMatch = await bcrypt.compare(password, user.password);

    if (!passwordMatch) {
      return NextResponse.json(
        { error: 'Senha incorreta' },
        { status: 401 }
      );
    }

    return NextResponse.json({
      id: user.id,
      email: user.email,
      name: user.name,
      userType: user.userType,
      hasCard: !!user.card,
      card: user.card
    });
  } catch (error) {
    console.error('Erro no login:', error);
    return NextResponse.json(
      { error: 'Erro ao fazer login' },
      { status: 500 }
    );
  }
}`;

fs.writeFileSync('app/api/auth/login/route.ts', loginAPI);
console.log('✅ app/api/auth/login/route.ts criado!');

// API - Create Appointment
const createAppointmentAPI = `import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { patientId, doctorId, date, time, notes } = body;

    const appointment = await prisma.appointment.create({
      data: {
        patientId,
        doctorId,
        date: new Date(date),
        time,
        notes,
        status: 'PENDING'
      },
      include: {
        patient: {
          select: {
            name: true,
            email: true
          }
        },
        doctor: {
          select: {
            name: true,
            email: true
          }
        }
      }
    });

    return NextResponse.json(appointment);
  } catch (error) {
    console.error('Erro ao criar agendamento:', error);
    return NextResponse.json(
      { error: 'Erro ao criar agendamento' },
      { status: 500 }
    );
  }
}`;

fs.writeFileSync('app/api/appointments/create/route.ts', createAppointmentAPI);
console.log('✅ app/api/appointments/create/route.ts criado!');

// API - List Appointments
const listAppointmentsAPI = `import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const userId = searchParams.get('userId');
  const userType = searchParams.get('userType');

  try {
    let appointments;
    
    if (userType === 'DOCTOR') {
      appointments = await prisma.appointment.findMany({
        where: { doctorId: userId || undefined },
        include: {
          patient: {
            select: {
              name: true,
              email: true,
              phone: true
            }
          }
        },
        orderBy: {
          date: 'asc'
        }
      });
    } else {
      appointments = await prisma.appointment.findMany({
        where: { patientId: userId || undefined },
        include: {
          doctor: {
            select: {
              name: true,
              email: true
            }
          }
        },
        orderBy: {
          date: 'asc'
        }
      });
    }

    return NextResponse.json(appointments);
  } catch (error) {
    console.error('Erro ao listar agendamentos:', error);
    return NextResponse.json(
      { error: 'Erro ao listar agendamentos' },
      { status: 500 }
    );
  }
}`;

fs.writeFileSync('app/api/appointments/list/route.ts', listAppointmentsAPI);
console.log('✅ app/api/appointments/list/route.ts criado!');

// 5. CRIAR ARQUIVO .ENV
console.log('\n🔐 Criando arquivo .env...\n');

const envExample = `# Pegue a URL no Neon Console
# https://console.neon.tech/app/projects/YOUR_PROJECT/dashboard
DATABASE_URL="postgresql://USER:PASSWORD@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require"

# Exemplo:
# DATABASE_URL="postgresql://luizgustavo:AbC123@ep-falling-thunder-73668518.us-east-2.aws.neon.tech/neondb?sslmode=require"
`;

if (!fs.existsSync('.env')) {
  fs.writeFileSync('.env', envExample);
  console.log('✅ .env criado! (Configure com sua URL do Neon)');
} else {
  console.log('⚠️  .env já existe - adicione DATABASE_URL manualmente');
}

// 6. ADICIONAR .ENV AO GITIGNORE
console.log('\n🚫 Atualizando .gitignore...\n');

const gitignoreContent = `
# Environment variables
.env
.env.local
.env.production
`;

if (fs.existsSync('.gitignore')) {
  const currentGitignore = fs.readFileSync('.gitignore', 'utf8');
  if (!currentGitignore.includes('.env')) {
    fs.appendFileSync('.gitignore', gitignoreContent);
    console.log('✅ .gitignore atualizado!');
  }
} else {
  fs.writeFileSync('.gitignore', gitignoreContent);
  console.log('✅ .gitignore criado!');
}

// 7. INSTALAR DEPENDÊNCIAS
console.log('\n📦 Instalando dependências...\n');

try {
  console.log('Instalando @prisma/client, prisma, bcryptjs...');
  execSync('npm install @prisma/client prisma bcryptjs', { stdio: 'inherit' });
  
  console.log('Instalando @types/bcryptjs...');
  execSync('npm install -D @types/bcryptjs', { stdio: 'inherit' });
  
  console.log('✅ Dependências instaladas!');
} catch (error) {
  console.log('⚠️  Erro ao instalar dependências. Execute manualmente:');
  console.log('   npm install @prisma/client prisma bcryptjs');
  console.log('   npm install -D @types/bcryptjs');
}

// RESUMO FINAL
console.log(`
╔═══════════════════════════════════════════════════════╗
║     ✅ BACKEND CRIADO COM SUCESSO!                    ║
╚═══════════════════════════════════════════════════════╝

📁 ESTRUTURA CRIADA:
   ✅ app/api/users/register
   ✅ app/api/cards/create
   ✅ app/api/cards/list
   ✅ app/api/auth/login
   ✅ app/api/appointments/create
   ✅ app/api/appointments/list
   ✅ lib/prisma.ts
   ✅ prisma/schema.prisma
   ✅ .env (exemplo)

🎯 PRÓXIMOS PASSOS:

1. CONFIGURE O BANCO DE DADOS:
   • Abra o arquivo .env
   • Cole sua DATABASE_URL do Neon Console
   
2. EXECUTE AS MIGRAÇÕES:
   npx prisma migrate dev --name init
   npx prisma generate

3. VISUALIZE O BANCO (OPCIONAL):
   npx prisma studio

4. TESTE O SISTEMA:
   npm run dev
   
📌 APIs DISPONÍVEIS:
   POST /api/users/register     - Criar usuário
   POST /api/cards/create       - Criar carteirinha
   GET  /api/cards/list         - Listar carteirinhas
   POST /api/auth/login         - Login
   POST /api/appointments/create - Criar agendamento
   GET  /api/appointments/list  - Listar agendamentos

💡 DICA: Acesse http://localhost:3000/carteirinha
         para testar a criação de carteirinhas!

🚀 Backend pronto para uso!
`);