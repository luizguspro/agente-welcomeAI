"use client";

import { useState } from "react";
import { useFormStore } from "@/lib/store";
import { formatCPF, formatPhone, formatDate } from "@/lib/formatters";

export default function ReviewStep() {
  const { formData, updateFormData, setStep } = useFormStore();
  const [acceptTerms, setAcceptTerms] = useState(formData.acceptTerms);

  const handleGenerate = () => {
    if (acceptTerms) {
      updateFormData({ acceptTerms });
      setStep(4);
    }
  };

  const handleBack = () => {
    setStep(2);
  };

  return (
    <div className="card">
      <h2 className="text-2xl font-bold mb-6">Revisão</h2>
      <p className="text-gray-600 mb-8">
        Confirme seus dados antes de gerar a carteirinha
      </p>

      <div className="space-y-6">
        <div className="flex gap-6">
          {formData.photo && (
            <img
              src={formData.photo}
              alt="Sua foto"
              className="w-32 h-40 object-cover rounded-lg"
            />
          )}
          
          <div className="flex-1 space-y-3">
            <div>
              <span className="text-sm text-gray-500">Nome</span>
              <p className="font-semibold">{formData.name}</p>
            </div>
            
            <div>
              <span className="text-sm text-gray-500">CPF</span>
              <p className="font-semibold">{formatCPF(formData.cpf)}</p>
            </div>
            
            <div>
              <span className="text-sm text-gray-500">Data de nascimento</span>
              <p className="font-semibold">{formatDate(formData.birthDate)}</p>
            </div>
            
            <div>
              <span className="text-sm text-gray-500">Telefone</span>
              <p className="font-semibold">{formatPhone(formData.phone)}</p>
            </div>
            
            <div>
              <span className="text-sm text-gray-500">E-mail</span>
              <p className="font-semibold">{formData.email}</p>
            </div>
          </div>
        </div>

        <div className="border-t pt-6">
          <label className="flex items-start gap-3">
            <input
              type="checkbox"
              checked={acceptTerms}
              onChange={(e) => setAcceptTerms(e.target.checked)}
              className="mt-1"
            />
            <span className="text-sm">
              Declaro que as informações fornecidas são verdadeiras e autorizo
              o uso dos meus dados para emissão da carteirinha digital de
              cannabis medicinal.
            </span>
          </label>
        </div>

        <div className="flex justify-between">
          <button onClick={handleBack} className="btn-secondary px-8 py-3">
            Voltar
          </button>
          <button
            onClick={handleGenerate}
            disabled={!acceptTerms}
            className="btn-primary px-8 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Gerar Carteirinha
          </button>
        </div>
      </div>
    </div>
  );
}