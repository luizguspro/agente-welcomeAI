"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Calendar, Clock, MapPin, Star, ArrowLeft, Check } from "lucide-react";
import Link from "next/link";

export default function AgendarConsulta() {
  const [selectedDoctor, setSelectedDoctor] = useState<number | null>(null);
  const [selectedDate, setSelectedDate] = useState("");
  const [selectedTime, setSelectedTime] = useState("");
  const [step, setStep] = useState(1);

  // Dados mockados de médicos
  const doctors = [
    {
      id: 1,
      name: "Dr. Roberto Mendes",
      specialty: "Clínico Geral",
      rating: 4.8,
      reviews: 127,
      location: "São Paulo - SP",
      photo: "👨‍⚕️",
      availableDates: ["2024-01-20", "2024-01-21", "2024-01-22"],
      availableTimes: ["09:00", "10:00", "11:00", "14:00", "15:00"],
    },
    {
      id: 2,
      name: "Dra. Ana Carolina",
      specialty: "Neurologista",
      rating: 4.9,
      reviews: 89,
      location: "São Paulo - SP",
      photo: "👩‍⚕️",
      availableDates: ["2024-01-20", "2024-01-21", "2024-01-23"],
      availableTimes: ["08:00", "09:00", "10:00", "14:00", "16:00"],
    },
    {
      id: 3,
      name: "Dr. Carlos Alberto",
      specialty: "Psiquiatra",
      rating: 4.7,
      reviews: 156,
      location: "São Paulo - SP",
      photo: "👨‍⚕️",
      availableDates: ["2024-01-21", "2024-01-22", "2024-01-24"],
      availableTimes: ["10:00", "11:00", "14:00", "15:00", "16:00"],
    },
  ];

  const handleNext = () => {
    if (step < 3) setStep(step + 1);
  };

  const handleBack = () => {
    if (step > 1) setStep(step - 1);
  };

  const handleConfirm = () => {
    alert("Consulta agendada com sucesso!");
  };

  const selectedDoctorData = doctors.find(d => d.id === selectedDoctor);

  return (
    <div className="min-h-screen bg-gradient-to-br from-greencard-background via-white to-greencard-tertiary/10">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-md border-b border-greencard-tertiary/20">
        <div className="container mx-auto px-4 sm:px-6 py-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2 text-greencard-primary hover:text-greencard-secondary">
              <ArrowLeft className="h-5 w-5" />
              <span className="font-semibold">Voltar</span>
            </Link>
            <h1 className="text-xl sm:text-2xl font-black">
              GREENCARD<span className="text-greencard-secondary italic">club</span>
            </h1>
            <div className="w-20"></div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 sm:px-6 py-8 max-w-6xl">
        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-center gap-4">
            {[1, 2, 3].map((s) => (
              <div key={s} className="flex items-center">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                  step >= s ? "bg-greencard-primary text-white" : "bg-gray-200 text-gray-500"
                }`}>
                  {step > s ? <Check className="h-5 w-5" /> : s}
                </div>
                {s < 3 && (
                  <div className={`w-20 sm:w-32 h-1 ${step > s ? "bg-greencard-primary" : "bg-gray-200"}`} />
                )}
              </div>
            ))}
          </div>
          <div className="flex justify-center gap-8 sm:gap-16 mt-3">
            <span className={`text-xs sm:text-sm ${step >= 1 ? "text-greencard-primary font-semibold" : "text-gray-500"}`}>
              Escolher Médico
            </span>
            <span className={`text-xs sm:text-sm ${step >= 2 ? "text-greencard-primary font-semibold" : "text-gray-500"}`}>
              Data e Hora
            </span>
            <span className={`text-xs sm:text-sm ${step >= 3 ? "text-greencard-primary font-semibold" : "text-gray-500"}`}>
              Confirmar
            </span>
          </div>
        </div>

        {/* Step 1: Escolher Médico */}
        {step === 1 && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <h2 className="text-2xl font-bold text-greencard-primary mb-6">
              Escolha um Médico Especialista
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {doctors.map((doctor) => (
                <motion.div
                  key={doctor.id}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setSelectedDoctor(doctor.id)}
                  className={`bg-white rounded-xl shadow-lg p-6 cursor-pointer transition-all ${
                    selectedDoctor === doctor.id ? "ring-2 ring-greencard-primary" : ""
                  }`}
                >
                  <div className="flex items-start gap-4">
                    <div className="text-4xl">{doctor.photo}</div>
                    <div className="flex-1">
                      <h3 className="font-bold text-lg">{doctor.name}</h3>
                      <p className="text-sm text-gray-600">{doctor.specialty}</p>
                      <div className="flex items-center gap-1 mt-2">
                        <Star className="h-4 w-4 text-yellow-500 fill-current" />
                        <span className="text-sm font-semibold">{doctor.rating}</span>
                        <span className="text-xs text-gray-500">({doctor.reviews} avaliações)</span>
                      </div>
                      <div className="flex items-center gap-1 mt-2">
                        <MapPin className="h-4 w-4 text-gray-500" />
                        <span className="text-sm text-gray-600">{doctor.location}</span>
                      </div>
                    </div>
                  </div>
                  {selectedDoctor === doctor.id && (
                    <div className="mt-4 bg-greencard-secondary/20 rounded-lg p-2 text-center">
                      <span className="text-sm font-semibold text-greencard-primary">Selecionado</span>
                    </div>
                  )}
                </motion.div>
              ))}
            </div>

            <div className="flex justify-end mt-8">
              <button
                onClick={handleNext}
                disabled={!selectedDoctor}
                className={`px-8 py-3 rounded-full font-bold transition-colors ${
                  selectedDoctor
                    ? "bg-greencard-primary hover:bg-greencard-primary/90 text-white"
                    : "bg-gray-200 text-gray-500 cursor-not-allowed"
                }`}
              >
                Próximo
              </button>
            </div>
          </motion.div>
        )}

        {/* Step 2: Escolher Data e Hora */}
        {step === 2 && selectedDoctorData && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <h2 className="text-2xl font-bold text-greencard-primary mb-6">
              Escolha a Data e Horário
            </h2>

            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center gap-4 mb-6">
                <div className="text-3xl">{selectedDoctorData.photo}</div>
                <div>
                  <h3 className="font-bold text-lg">{selectedDoctorData.name}</h3>
                  <p className="text-gray-600">{selectedDoctorData.specialty}</p>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label className="block font-semibold text-greencard-primary mb-3">
                    <Calendar className="inline h-5 w-5 mr-2" />
                    Selecione a Data
                  </label>
                  <div className="space-y-2">
                    {selectedDoctorData.availableDates.map((date) => (
                      <button
                        key={date}
                        onClick={() => setSelectedDate(date)}
                        className={`w-full p-3 rounded-lg border-2 transition-colors ${
                          selectedDate === date
                            ? "border-greencard-primary bg-greencard-secondary/20"
                            : "border-gray-200 hover:border-greencard-tertiary"
                        }`}
                      >
                        {new Date(date).toLocaleDateString("pt-BR", {
                          weekday: "long",
                          day: "numeric",
                          month: "long"
                        })}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block font-semibold text-greencard-primary mb-3">
                    <Clock className="inline h-5 w-5 mr-2" />
                    Selecione o Horário
                  </label>
                  <div className="grid grid-cols-3 gap-2">
                    {selectedDoctorData.availableTimes.map((time) => (
                      <button
                        key={time}
                        onClick={() => setSelectedTime(time)}
                        className={`p-3 rounded-lg border-2 transition-colors ${
                          selectedTime === time
                            ? "border-greencard-primary bg-greencard-secondary/20"
                            : "border-gray-200 hover:border-greencard-tertiary"
                        }`}
                      >
                        {time}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div className="flex justify-between mt-8">
              <button
                onClick={handleBack}
                className="px-6 py-3 text-greencard-primary hover:bg-greencard-background rounded-lg transition-colors"
              >
                Voltar
              </button>
              <button
                onClick={handleNext}
                disabled={!selectedDate || !selectedTime}
                className={`px-8 py-3 rounded-full font-bold transition-colors ${
                  selectedDate && selectedTime
                    ? "bg-greencard-primary hover:bg-greencard-primary/90 text-white"
                    : "bg-gray-200 text-gray-500 cursor-not-allowed"
                }`}
              >
                Próximo
              </button>
            </div>
          </motion.div>
        )}

        {/* Step 3: Confirmar */}
        {step === 3 && selectedDoctorData && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="max-w-2xl mx-auto"
          >
            <h2 className="text-2xl font-bold text-greencard-primary mb-6">
              Confirmar Agendamento
            </h2>

            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="text-center mb-6">
                <div className="w-20 h-20 bg-greencard-secondary/20 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Check className="h-10 w-10 text-greencard-primary" />
                </div>
                <h3 className="font-bold text-xl mb-2">Revise os Detalhes da Consulta</h3>
              </div>

              <div className="space-y-4 bg-gray-50 rounded-lg p-4">
                <div className="flex items-center gap-4">
                  <div className="text-3xl">{selectedDoctorData.photo}</div>
                  <div>
                    <p className="font-bold">{selectedDoctorData.name}</p>
                    <p className="text-sm text-gray-600">{selectedDoctorData.specialty}</p>
                  </div>
                </div>
                
                <div className="border-t pt-4">
                  <p className="flex items-center gap-2 mb-2">
                    <Calendar className="h-5 w-5 text-greencard-primary" />
                    <span className="font-semibold">Data:</span>
                    {new Date(selectedDate).toLocaleDateString("pt-BR", {
                      weekday: "long",
                      day: "numeric",
                      month: "long",
                      year: "numeric"
                    })}
                  </p>
                  <p className="flex items-center gap-2 mb-2">
                    <Clock className="h-5 w-5 text-greencard-primary" />
                    <span className="font-semibold">Horário:</span>
                    {selectedTime}
                  </p>
                  <p className="flex items-center gap-2">
                    <MapPin className="h-5 w-5 text-greencard-primary" />
                    <span className="font-semibold">Local:</span>
                    {selectedDoctorData.location}
                  </p>
                </div>
              </div>

              <div className="mt-6 p-4 bg-yellow-50 rounded-lg">
                <p className="text-sm text-yellow-800">
                  ⚠️ <strong>Importante:</strong> Leve sua carteirinha digital Greencardclub e documentos pessoais no dia da consulta.
                </p>
              </div>

              <div className="flex gap-4 mt-6">
                <button
                  onClick={handleBack}
                  className="flex-1 px-6 py-3 border-2 border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Voltar
                </button>
                <button
                  onClick={handleConfirm}
                  className="flex-1 px-6 py-3 bg-greencard-primary hover:bg-greencard-primary/90 text-white font-bold rounded-lg transition-colors"
                >
                  Confirmar Agendamento
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}