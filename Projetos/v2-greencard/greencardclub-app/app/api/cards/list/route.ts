import { NextResponse } from 'next/server';
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
}