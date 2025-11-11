"use client";

import { useState, useRef } from "react";
import { useFormStore } from "@/lib/store";
import { Camera, Upload, X } from "lucide-react";

export default function PhotoStep() {
  const { formData, updateFormData, setStep } = useFormStore();
  const [photo, setPhoto] = useState<string | null>(formData.photo);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.size <= 5 * 1024 * 1024) {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        setPhoto(result);
      };
      reader.readAsDataURL(file);
    } else {
      alert("A foto deve ter no máximo 5MB");
    }
  };

  const handleNext = () => {
    if (photo) {
      updateFormData({ photo });
      setStep(3);
    }
  };

  const handleBack = () => {
    setStep(1);
  };

  return (
    <div className="card">
      <h2 className="text-2xl font-bold mb-6">Foto</h2>
      <p className="text-gray-600 mb-8">
        Adicione uma foto para sua carteirinha digital
      </p>

      <div className="space-y-6">
        {!photo ? (
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-greencard-tertiary rounded-lg p-12 text-center cursor-pointer hover:border-greencard-primary transition-colors"
          >
            <Upload className="h-12 w-12 text-greencard-primary mx-auto mb-4" />
            <p className="text-gray-600">
              Clique para enviar ou arraste sua foto aqui
            </p>
            <p className="text-sm text-gray-500 mt-2">Máximo 5MB</p>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
          </div>
        ) : (
          <div className="relative">
            <img
              src={photo}
              alt="Sua foto"
              className="w-48 h-60 object-cover rounded-lg mx-auto"
            />
            <button
              onClick={() => setPhoto(null)}
              className="absolute top-0 right-0 bg-red-500 text-white p-2 rounded-full transform translate-x-1/2 -translate-y-1/2"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        <div className="bg-greencard-tertiary/20 rounded-lg p-4">
          <h4 className="font-semibold mb-2">Instruções para a foto:</h4>
          <ul className="text-sm text-gray-600 space-y-1">
            <li>• Foto recente, colorida e nítida</li>
            <li>• Fundo claro e uniforme</li>
            <li>• Rosto centralizado e visível</li>
            <li>• Sem óculos escuros ou acessórios</li>
          </ul>
        </div>

        <div className="flex justify-between">
          <button onClick={handleBack} className="btn-secondary px-8 py-3">
            Voltar
          </button>
          <button
            onClick={handleNext}
            disabled={!photo}
            className="btn-primary px-8 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Próximo
          </button>
        </div>
      </div>
    </div>
  );
}