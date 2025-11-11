import { NextResponse } from 'next/server';
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
}