import { NextResponse } from 'next/server';
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
}