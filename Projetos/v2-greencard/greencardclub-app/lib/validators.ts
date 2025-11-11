import { z } from "zod";

// CPF validation function - FIXED
function validateCPF(cpf: string): boolean {
  cpf = cpf.replace(/[^\d]+/g, "");
  
  if (cpf.length !== 11 || /^(\d)\1{10}$/.test(cpf)) {
    return false;
  }
  
  let sum = 0;
  let remainder;
  
  for (let i = 1; i <= 9; i++) {
    sum += parseInt(cpf.substring(i - 1, i)) * (11 - i);
  }
  
  remainder = (sum * 10) % 11;
  if (remainder === 10 || remainder === 11) remainder = 0;
  if (remainder !== parseInt(cpf.substring(9, 10))) return false;
  
  sum = 0;
  for (let i = 1; i <= 10; i++) {
    sum += parseInt(cpf.substring(i - 1, i)) * (12 - i);
  }
  
  remainder = (sum * 10) % 11;
  if (remainder === 10 || remainder === 11) remainder = 0;
  if (remainder !== parseInt(cpf.substring(10, 11))) return false;
  
  return true;
}

// Form schemas
export const personalDataSchema = z.object({
  name: z.string().min(3, "Nome deve ter no mínimo 3 caracteres"),
  cpf: z.string().refine(validateCPF, "CPF inválido"),
  birthDate: z.string().min(1, "Data de nascimento é obrigatória"),
  phone: z.string().min(10, "Telefone inválido"),
  email: z.string().email("E-mail inválido"),
});

export const photoSchema = z.object({
  photo: z.string().min(1, "Foto é obrigatória"),
});

export const termsSchema = z.object({
  acceptTerms: z.boolean().refine((val) => val === true, {
    message: "Você deve aceitar os termos",
  }),
});

export type PersonalData = z.infer<typeof personalDataSchema>;
export type PhotoData = z.infer<typeof photoSchema>;
export type TermsData = z.infer<typeof termsSchema>;