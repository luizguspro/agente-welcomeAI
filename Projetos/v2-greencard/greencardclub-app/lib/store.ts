import { create } from 'zustand';

interface FormData {
  name: string;
  cpf: string;
  birthDate: string;
  phone: string;
  email: string;
  photo: string | null;
  acceptTerms: boolean;
}

interface FormStore {
  currentStep: number;
  formData: FormData;
  isGenerating: boolean;
  cardUrl: string | null;
  
  setStep: (step: number) => void;
  updateFormData: (data: Partial<FormData>) => void;
  resetForm: () => void;
  setIsGenerating: (value: boolean) => void;
  setCardUrl: (url: string | null) => void;
}

const initialFormData: FormData = {
  name: '',
  cpf: '',
  birthDate: '',
  phone: '',
  email: '',
  photo: null,
  acceptTerms: false,
};

export const useFormStore = create<FormStore>((set) => ({
  currentStep: 1,
  formData: initialFormData,
  isGenerating: false,
  cardUrl: null,
  
  setStep: (step) => set({ currentStep: step }),
  updateFormData: (data) =>
    set((state) => ({
      formData: { ...state.formData, ...data },
    })),
  resetForm: () =>
    set({
      currentStep: 1,
      formData: initialFormData,
      cardUrl: null,
    }),
  setIsGenerating: (value) => set({ isGenerating: value }),
  setCardUrl: (url) => set({ cardUrl: url }),
}));