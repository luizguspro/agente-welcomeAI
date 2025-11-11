"use client";

import { useState, useEffect } from "react";
import { motion, useMotionValue, useTransform } from "framer-motion";
import { Leaf, Shield, Clock, Users, ArrowRight, QrCode, User } from "lucide-react";
import Link from "next/link";

export default function Home() {
  const [isHovered, setIsHovered] = useState(false);
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);
  const rotateX = useTransform(mouseY, [-300, 300], [25, -25]);
  const rotateY = useTransform(mouseX, [-300, 300], [-25, 25]);

  useEffect(() => {
    const handleMouseMove = (e) => {
      const rect = document.body.getBoundingClientRect();
      const x = e.clientX - rect.left - rect.width / 2;
      const y = e.clientY - rect.top - rect.height / 2;
      mouseX.set(x);
      mouseY.set(y);
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, [mouseX, mouseY]);

  const stats = [
    { icon: "👥", value: "1.234", label: "Usuários Ativos" },
    { icon: "⏱️", value: "5min", label: "Tempo Médio" },
    { icon: "✅", value: "100%", label: "Digital" }
  ];

  const features = [
    { icon: Shield, title: "100% Legal", desc: "Processo regulamentado" },
    { icon: Clock, title: "5 Minutos", desc: "Carteirinha digital rápida" },
    { icon: Users, title: "Suporte Médico", desc: "Profissionais especializados" }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-greencard-background via-white to-greencard-tertiary/10">
      {/* Background Effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-greencard-secondary/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-greencard-tertiary/20 rounded-full blur-3xl" />
      </div>

      {/* Header */}
      <header className="relative z-50 bg-white/80 backdrop-blur-md border-b border-greencard-tertiary/20">
        <nav className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2">
              <motion.div
                animate={{ rotate: isHovered ? 360 : 0 }}
                transition={{ duration: 0.5 }}
                onHoverStart={() => setIsHovered(true)}
                onHoverEnd={() => setIsHovered(false)}
              >
                <Leaf className="h-8 w-8 text-greencard-primary" />
              </motion.div>
              <span className="text-2xl font-black">
                GREENCARD<span className="text-greencard-secondary italic font-light">club</span>
              </span>
            </Link>
            
            <div className="flex items-center gap-4">
              <Link href="/login" className="text-greencard-primary hover:text-greencard-secondary font-medium transition-colors">
                Entrar
              </Link>
              <Link href="/carteirinha" className="bg-greencard-primary hover:bg-greencard-primary/90 text-white font-bold px-6 py-2.5 rounded-full transition-all hover:shadow-lg flex items-center gap-2">
                Criar Carteirinha
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
          </div>
        </nav>
      </header>

      {/* Hero Section - Centralizado */}
      <main className="relative">
        <section className="min-h-[calc(100vh-73px)] flex items-center justify-center py-12">
          <div className="container mx-auto px-6">
            <div className="max-w-5xl mx-auto">
              {/* Stats no topo */}
              <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                className="flex justify-center gap-12 mb-12"
              >
                {stats.map((stat, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.1 }}
                    className="text-center"
                  >
                    <div className="text-3xl mb-1">{stat.icon}</div>
                    <p className="text-2xl font-black text-greencard-primary">{stat.value}</p>
                    <p className="text-sm text-gray-600">{stat.label}</p>
                  </motion.div>
                ))}
              </motion.div>

              {/* Badge */}
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.3 }}
                className="flex justify-center mb-8"
              >
                <div className="inline-flex items-center gap-2 px-6 py-3 bg-greencard-secondary/20 backdrop-blur-sm rounded-full">
                  <Shield className="h-5 w-5 text-greencard-primary" />
                  <span className="text-sm font-bold text-greencard-primary">100% Legal e Seguro</span>
                </div>
              </motion.div>

              {/* Título e Descrição */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="text-center mb-8"
              >
                <h1 className="text-5xl md:text-6xl lg:text-7xl font-black leading-tight mb-6">
                  Sua Carteirinha<br />
                  Digital para<br />
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-greencard-primary to-greencard-secondary">
                    Cannabis Medicinal
                  </span>
                </h1>
                
                <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-8">
                  Cadastre-se, obtenha sua autorização digital e tenha acesso legal ao tratamento. 
                  Processo 100% online, rápido e seguro.
                </p>

                {/* CTAs */}
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                    <Link 
                      href="/carteirinha" 
                      className="bg-greencard-primary hover:bg-greencard-primary/90 text-white font-bold text-lg px-10 py-4 rounded-full transition-all shadow-xl hover:shadow-2xl flex items-center gap-2 justify-center"
                    >
                      Criar Carteirinha Agora
                      <ArrowRight className="h-5 w-5" />
                    </Link>
                  </motion.div>
                  <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                    <Link 
                      href="/login" 
                      className="bg-white hover:bg-gray-50 text-greencard-primary font-bold text-lg px-10 py-4 rounded-full transition-all border-2 border-greencard-primary shadow-lg hover:shadow-xl"
                    >
                      Já Tenho Conta
                    </Link>
                  </motion.div>
                </div>
              </motion.div>

              {/* Carteirinha 3D Ultra Realista */}
              <motion.div
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6, duration: 0.8 }}
                className="flex justify-center mt-16"
                style={{ perspective: "1000px" }}
              >
                <motion.div
                  style={{
                    rotateX,
                    rotateY,
                    transformStyle: "preserve-3d"
                  }}
                  whileHover={{ scale: 1.05 }}
                  className="relative"
                >
                  {/* Shadow */}
                  <div className="absolute -bottom-4 left-1/2 -translate-x-1/2 w-[90%] h-8 bg-black/20 blur-2xl rounded-full" />
                  
                  {/* Card Container */}
                  <div className="relative w-[420px] h-[270px] bg-gradient-to-br from-[#0a5a42] via-[#084734] to-[#063d2b] rounded-2xl shadow-2xl overflow-hidden">
                    {/* Texture Pattern */}
                    <div className="absolute inset-0 opacity-5">
                      <div 
                        className="absolute inset-0" 
                        style={{
                          backgroundImage: "repeating-linear-gradient(45deg, transparent, transparent 35px, rgba(255,255,255,.1) 35px, rgba(255,255,255,.1) 70px)"
                        }} 
                      />
                    </div>

                    {/* Holographic Effect */}
                    <motion.div
                      className="absolute inset-0 opacity-30"
                      animate={{
                        background: [
                          "linear-gradient(135deg, transparent 20%, rgba(206, 241, 123, 0.3) 40%, transparent 60%)",
                          "linear-gradient(135deg, transparent 40%, rgba(206, 241, 123, 0.3) 60%, transparent 80%)",
                          "linear-gradient(135deg, transparent 60%, rgba(206, 241, 123, 0.3) 80%, transparent 100%)"
                        ]
                      }}
                      transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                    />

                    {/* Card Content */}
                    <div className="relative z-10 p-6 h-full flex flex-col">
                      {/* Header */}
                      <div className="flex justify-between items-start mb-6">
                        <div>
                          <h3 className="text-white text-2xl font-black tracking-wide">
                            GREENCARD<span className="text-greencard-secondary italic font-light">club</span>
                          </h3>
                          <p className="text-white/60 text-xs tracking-widest mt-1">CARTEIRA DIGITAL</p>
                        </div>
                        <div className="bg-white/20 backdrop-blur-md px-3 py-1.5 rounded-lg">
                          <span className="text-white font-bold text-sm">GC</span>
                        </div>
                      </div>

                      {/* Main Content */}
                      <div className="flex gap-6 flex-1">
                        {/* Photo */}
                        <div className="relative">
                          <div className="w-20 h-24 bg-gradient-to-br from-white/20 to-white/10 rounded-lg backdrop-blur-sm overflow-hidden">
                            <div className="w-full h-full flex items-center justify-center">
                              <User className="w-12 h-12 text-white/40" />
                            </div>
                            {/* Photo Shine Effect */}
                            <motion.div
                              className="absolute inset-0 bg-gradient-to-tr from-transparent via-white/20 to-transparent"
                              animate={{ x: [-100, 100] }}
                              transition={{ duration: 3, repeat: Infinity, repeatDelay: 3 }}
                            />
                          </div>
                        </div>

                        {/* User Info */}
                        <div className="flex-1 flex flex-col justify-center">
                          <div className="space-y-1">
                            <div className="h-2 bg-white/30 rounded-full w-3/4" />
                            <div className="h-2 bg-white/20 rounded-full w-1/2" />
                            <div className="h-2 bg-white/20 rounded-full w-2/3" />
                          </div>
                          <div className="mt-3">
                            <p className="text-white/60 text-xs">AUTORIZAÇÃO MÉDICA</p>
                            <p className="text-white font-mono text-sm">THC/CBD</p>
                          </div>
                        </div>

                        {/* Right Side Info */}
                        <div className="flex flex-col justify-between items-end">
                          <div>
                            <QrCode className="w-16 h-16 text-white/30" />
                          </div>
                          <div className="text-right">
                            <p className="text-white/60 text-xs">VÁLIDA ATÉ</p>
                            <p className="text-white font-bold">12/2025</p>
                            <div className="flex items-center gap-1 mt-1">
                              <motion.div
                                animate={{ scale: [1, 1.2, 1] }}
                                transition={{ duration: 2, repeat: Infinity }}
                                className="w-2 h-2 bg-greencard-secondary rounded-full"
                              />
                              <span className="text-white/80 text-xs">ATIVA</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Card Chip */}
                    <div className="absolute bottom-6 left-6 w-12 h-10 bg-gradient-to-br from-yellow-400/80 to-yellow-600/80 rounded-md">
                      <div 
                        className="w-full h-full rounded-md border border-yellow-700/30" 
                        style={{
                          backgroundImage: "linear-gradient(90deg, transparent 40%, rgba(255,255,255,0.1) 50%, transparent 60%), linear-gradient(0deg, transparent 40%, rgba(255,255,255,0.1) 50%, transparent 60%)"
                        }}
                      />
                    </div>

                    {/* Glossy Overlay */}
                    <div className="absolute inset-0 bg-gradient-to-br from-white/10 via-transparent to-transparent pointer-events-none" />
                  </div>
                </motion.div>
              </motion.div>

              {/* Features Icons */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8 }}
                className="flex justify-center gap-8 mt-12"
              >
                <div className="flex items-center gap-2 text-sm text-gray-600">
                  <QrCode className="h-5 w-5 text-greencard-primary" />
                  <span>QR Code Verificável</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-600">
                  <Shield className="h-5 w-5 text-greencard-primary" />
                  <span>Carteirinha Digital</span>
                </div>
              </motion.div>
            </div>
          </div>
        </section>

        {/* Features Section Simplificada */}
        <section className="py-20 bg-white/50">
          <div className="container mx-auto px-6">
            <div className="max-w-4xl mx-auto">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="text-center mb-12"
              >
                <h2 className="text-4xl font-black mb-4">
                  Por que escolher o Greencard<span className="text-greencard-secondary italic">club</span>?
                </h2>
              </motion.div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {features.map((item, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.1 }}
                    viewport={{ once: true }}
                    className="text-center"
                  >
                    <div className="w-16 h-16 mx-auto mb-4 bg-greencard-secondary/20 rounded-2xl flex items-center justify-center">
                      <item.icon className="h-8 w-8 text-greencard-primary" />
                    </div>
                    <h3 className="font-bold text-lg mb-2">{item.title}</h3>
                    <p className="text-gray-600 text-sm">{item.desc}</p>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Simple CTA */}
        <section className="py-20 bg-gradient-to-br from-greencard-primary to-greencard-primary/90">
          <div className="container mx-auto px-6 text-center">
            <h2 className="text-4xl font-black text-white mb-4">
              Pronto para começar?
            </h2>
            <p className="text-white/80 mb-8">
              Junte-se a milhares de pessoas com acesso legal ao tratamento
            </p>
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} className="inline-block">
              <Link 
                href="/carteirinha" 
                className="bg-white hover:bg-gray-100 text-greencard-primary font-black px-12 py-4 rounded-full transition-all shadow-2xl inline-flex items-center gap-2"
              >
                Criar Minha Carteirinha
                <ArrowRight className="h-5 w-5" />
              </Link>
            </motion.div>
          </div>
        </section>

        {/* Footer */}
        <footer className="bg-greencard-primary/95 text-white py-12">
          <div className="container mx-auto px-6 text-center">
            <div className="flex items-center justify-center gap-2 mb-4">
              <Leaf className="h-6 w-6" />
              <span className="text-xl font-bold">GREENCARDCLUB</span>
            </div>
            <p className="text-sm opacity-80 mb-4">
              Tornando simples o que deveria ser.
            </p>
            <p className="text-xs opacity-60">
              © 2024 Greencardclub. Todos os direitos reservados.
            </p>
          </div>
        </footer>
      </main>
    </div>
  );
}