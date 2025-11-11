import { NextResponse } from 'next/server';
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
}