"use client";

import { useEffect, useState } from "react";
import { useFormStore } from "@/lib/store";
import { CheckCircle2, Download, RotateCcw } from "lucide-react";
import { generateCard } from "@/lib/card-generator";

export default function SuccessStep() {
  const { formData, cardUrl, setCardUrl, resetForm } = useFormStore();
  const [isGenerating, setIsGenerating] = useState(true);

  useEffect(() => {
    const generate = async () => {
      try {
        const url = await generateCard(formData);
        setCardUrl(url);
      } catch (error) {
        console.error("Error generating card:", error);
      } finally {
        setIsGenerating(false);
      }
    };

    if (!cardUrl) {
      generate();
    } else {
      setIsGenerating(false);
    }
  }, [formData, cardUrl, setCardUrl]);

  const handleDownload = () => {
    if (cardUrl) {
      const link = document.createElement("a");
      link.download = "greencardclub_carteirinha.png";
      link.href = cardUrl;
      link.click();
    }
  };

  const handleNewCard = () => {
    resetForm();
  };

  return (
    <div className="card">
      <div className="text-center">
        <CheckCircle2 className="h-16 w-16 text-green-500 mx-auto mb-4" />
        <h2 className="text-2xl font-bold mb-2">Carteirinha Gerada com Sucesso!</h2>
        <p className="text-gray-600 mb-8">
          Sua carteirinha digital está pronta para download
        </p>
      </div>

      {isGenerating ? (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-greencard-primary mx-auto mb-4"></div>
          <p className="text-gray-600">Gerando sua carteirinha...</p>
        </div>
      ) : (
        <>
          {cardUrl && (
            <div className="mb-8">
              <img
                src={cardUrl}
                alt="Sua carteirinha"
                className="w-full max-w-md mx-auto rounded-lg shadow-lg"
              />
            </div>
          )}

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button onClick={handleDownload} className="btn-primary px-8 py-3 flex items-center justify-center gap-2">
              <Download className="h-5 w-5" />
              Baixar Carteirinha
            </button>
            
            <button onClick={handleNewCard} className="btn-secondary px-8 py-3 flex items-center justify-center gap-2">
              <RotateCcw className="h-5 w-5" />
              Nova Carteirinha
            </button>
          </div>
        </>
      )}
    </div>
  );
}