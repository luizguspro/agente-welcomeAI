"use client";

import { useState } from "react";
import { useFormStore } from "@/lib/store";
import { formatCPF, formatPhone } from "@/lib/formatters";
import { personalDataSchema } from "@/lib/validators";
import { z } from "zod";

export default function PersonalDataStep() {
  const { formData, updateFormData, setStep } = useFormStore();
  const [errors, setErrors] = useState<any>({});

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      const validData = personalDataSchema.parse(formData);
      updateFormData(validData);
      setStep(2);
      setErrors({});
    } catch (error) {
      if (error instanceof z.ZodError) {
        const fieldErrors: any = {};
        error.errors.forEach((err) => {
          if (err.path[0]) {
            fieldErrors[err.path[0]] = err.message;
          }
        });
        setErrors(fieldErrors);
      }
    }
  };

  const handleCPFChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const formatted = formatCPF(e.target.value);
    updateFormData({ cpf: formatted });
  };

  const handlePhoneChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const formatted = formatPhone(e.target.value);
    updateFormData({ phone: formatted });
  };

  return (
    <div className="card">
      <h2 className="text-2xl font-bold mb-6">Dados Pessoais</h2>
      <p className="text-gray-600 mb-8">
        Preencha seus dados para criar sua carteirinha digital
      </p>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="name" className="label">
            Nome completo
          </label>
          <input
            id="name"
            type="text"
            className="input"
            placeholder="Digite seu nome completo"
            value={formData.name}
            onChange={(e) => updateFormData({ name: e.target.value })}
          />
          {errors.name && (
            <p className="text-red-500 text-sm mt-1">{errors.name}</p>
          )}
        </div>

        <div>
          <label htmlFor="cpf" className="label">
            CPF
          </label>
          <input
            id="cpf"
            type="text"
            className="input"
            placeholder="000.000.000-00"
            value={formData.cpf}
            onChange={handleCPFChange}
            maxLength={14}
          />
          {errors.cpf && (
            <p className="text-red-500 text-sm mt-1">{errors.cpf}</p>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="birthDate" className="label">
              Data de nascimento
            </label>
            <input
              id="birthDate"
              type="date"
              className="input"
              value={formData.birthDate}
              onChange={(e) => updateFormData({ birthDate: e.target.value })}
            />
            {errors.birthDate && (
              <p className="text-red-500 text-sm mt-1">{errors.birthDate}</p>
            )}
          </div>

          <div>
            <label htmlFor="phone" className="label">
              Telefone
            </label>
            <input
              id="phone"
              type="tel"
              className="input"
              placeholder="(00) 00000-0000"
              value={formData.phone}
              onChange={handlePhoneChange}
              maxLength={15}
            />
            {errors.phone && (
              <p className="text-red-500 text-sm mt-1">{errors.phone}</p>
            )}
          </div>
        </div>

        <div>
          <label htmlFor="email" className="label">
            E-mail
          </label>
          <input
            id="email"
            type="email"
            className="input"
            placeholder="seu@email.com"
            value={formData.email}
            onChange={(e) => updateFormData({ email: e.target.value })}
          />
          {errors.email && (
            <p className="text-red-500 text-sm mt-1">{errors.email}</p>
          )}
        </div>

        <div className="flex justify-end">
          <button type="submit" className="btn-primary px-8 py-3">
            Próximo
          </button>
        </div>
      </form>
    </div>
  );
}