"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { User, Stethoscope, ArrowRight, Leaf, ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter();
  const [userType, setUserType] = useState<"patient" | "doctor" | null>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (userType === "doctor") {
      router.push("/medico/dashboard");
    } else {
      router.push("/agendar");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-greencard-background via-white to-greencard-tertiary/10">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-md border-b border-greencard-tertiary/20">
        <div className="container mx-auto px-4 sm:px-6 py-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2">
              <ArrowLeft className="h-5 w-5 text-greencard-primary" />
              <span className="font-semibold text-greencard-primary">Voltar</span>
            </Link>
            <div className="flex items-center gap-2">
              <Leaf className="h-8 w-8 text-greencard-primary" />
              <span className="text-2xl font-black">
                GREENCARD<span className="text-greencard-secondary italic">club</span>
              </span>
            </div>
            <div className="w-20"></div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 sm:px-6 py-12 max-w-md">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl shadow-xl p-8"
        >
          {!userType ? (
            <>
              <h2 className="text-2xl font-bold text-center text-greencard-primary mb-8">
                Como deseja entrar?
              </h2>
              
              <div className="space-y-4">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setUserType("patient")}
                  className="w-full bg-greencard-primary hover:bg-greencard-primary/90 text-white rounded-xl p-6 flex items-center justify-between transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <User className="h-8 w-8" />
                    <div className="text-left">
                      <p className="font-bold text-lg">Sou Paciente</p>
                      <p className="text-sm opacity-90">Agendar consulta médica</p>
                    </div>
                  </div>
                  <ArrowRight className="h-6 w-6" />
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setUserType("doctor")}
                  className="w-full bg-greencard-secondary hover:bg-greencard-secondary/90 text-greencard-primary rounded-xl p-6 flex items-center justify-between transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <Stethoscope className="h-8 w-8" />
                    <div className="text-left">
                      <p className="font-bold text-lg">Sou Médico</p>
                      <p className="text-sm">Gerenciar agenda e pacientes</p>
                    </div>
                  </div>
                  <ArrowRight className="h-6 w-6" />
                </motion.button>
              </div>
            </>
          ) : (
            <>
              <div className="text-center mb-8">
                <div className="w-20 h-20 bg-greencard-secondary/20 rounded-full flex items-center justify-center mx-auto mb-4">
                  {userType === "doctor" ? (
                    <Stethoscope className="h-10 w-10 text-greencard-primary" />
                  ) : (
                    <User className="h-10 w-10 text-greencard-primary" />
                  )}
                </div>
                <h2 className="text-2xl font-bold text-greencard-primary">
                  {userType === "doctor" ? "Portal Médico" : "Portal do Paciente"}
                </h2>
              </div>

              <form onSubmit={handleLogin} className="space-y-6">
                <div>
                  <label className="block text-sm font-semibold text-greencard-primary mb-2">
                    E-mail
                  </label>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-greencard-primary focus:outline-none"
                    placeholder="seu@email.com"
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-greencard-primary mb-2">
                    Senha
                  </label>
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-greencard-primary focus:outline-none"
                    placeholder="••••••••"
                    required
                  />
                </div>

                <button
                  type="submit"
                  className="w-full bg-greencard-primary hover:bg-greencard-primary/90 text-white font-bold py-3 rounded-xl transition-colors"
                >
                  Entrar
                </button>

                <button
                  type="button"
                  onClick={() => setUserType(null)}
                  className="w-full text-greencard-primary hover:text-greencard-secondary transition-colors text-sm"
                >
                  ← Voltar para seleção
                </button>
              </form>
            </>
          )}
        </motion.div>
      </div>
    </div>
  );
}