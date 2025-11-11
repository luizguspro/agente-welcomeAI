import { NextResponse } from 'next/server';
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
    const cardNumber = `GC${Date.now().toString().slice(-8)}`;
    
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
}