"use client";

import { motion } from "framer-motion";
import { useState, useEffect } from "react";

interface Card3DProps {
  name?: string;
  cpf?: string;
  validity?: string;
  photoUrl?: string;
  size?: "sm" | "md" | "lg";
  interactive?: boolean;
}

export default function Card3D({ 
  name = "NOME DO USUÁRIO",
  cpf = "000.000.000-00",
  validity = "12/2025",
  photoUrl,
  size = "md",
  interactive = true
}: Card3DProps) {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  
  const sizes = {
    sm: { width: "280px", height: "440px", scale: 0.8 },
    md: { width: "380px", height: "600px", scale: 1 },
    lg: { width: "480px", height: "760px", scale: 1.2 }
  };

  const currentSize = sizes[size];

  useEffect(() => {
    if (!interactive) return;
    
    const handleMouseMove = (e: MouseEvent) => {
      const x = (e.clientX / window.innerWidth - 0.5) * 20;
      const y = (e.clientY / window.innerHeight - 0.5) * 20;
      setMousePosition({ x, y });
    };
    
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, [interactive]);

  return (
    <motion.div
      animate={interactive ? {
        rotateY: mousePosition.x,
        rotateX: -mousePosition.y,
      } : {}}
      transition={{ type: "spring", stiffness: 75 }}
      style={{
        transformStyle: "preserve-3d",
        perspective: 1000,
        width: currentSize.width,
        height: currentSize.height,
      }}
      className="relative mx-auto"
    >
      {/* Glow Effect */}
      <div className="absolute inset-0 bg-gradient-to-r from-greencard-secondary to-greencard-tertiary rounded-3xl blur-3xl opacity-50 scale-110 animate-pulse" />
      
      {/* Main Card */}
      <motion.div
        whileHover={interactive ? { scale: 1.05, rotateY: 10 } : {}}
        className="relative w-full h-full bg-gradient-to-br from-greencard-primary to-greencard-primary/90 rounded-3xl shadow-2xl overflow-hidden"
        style={{ transformStyle: "preserve-3d", transform: `scale(${currentSize.scale})` }}
      >
        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%23ffffff" fill-opacity="0.1"%3E%3Cpath d="M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')] bg-repeat" />
        </div>
        
        {/* Holographic Overlay */}
        <motion.div
          animate={{
            background: [
              "linear-gradient(45deg, transparent 30%, rgba(206, 241, 123, 0.3) 50%, transparent 70%)",
              "linear-gradient(45deg, transparent 30%, rgba(206, 241, 123, 0.3) 50%, transparent 70%)",
            ],
            backgroundPosition: ["-200% 0%", "200% 0%"],
          }}
          transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
          className="absolute inset-0 pointer-events-none z-20"
        />
        
        {/* Card Content */}
        <div className="relative z-10 p-8 h-full flex flex-col text-white">
          {/* Header */}
          <div className="flex justify-between items-start mb-8">
            <div>
              <p className="text-xs uppercase tracking-wider opacity-80 mb-1">Carteira Digital</p>
              <h3 className="text-2xl font-black">GREENCARD</h3>
              <span className="text-greencard-secondary text-xl italic">club</span>
            </div>
            <motion.div 
              whileHover={{ scale: 1.1, rotate: 360 }}
              transition={{ duration: 0.5 }}
              className="bg-white/20 backdrop-blur-xl px-3 py-1 rounded-full"
            >
              <span className="text-xs font-bold">GC</span>
            </motion.div>
          </div>
          
          {/* Main Content */}
          <div className="flex gap-6 flex-1">
            {/* Photo */}
            <div className="relative">
              {photoUrl ? (
                <img 
                  src={photoUrl} 
                  alt="Foto do usuário" 
                  className="w-24 h-32 object-cover rounded-lg"
                />
              ) : (
                <motion.div 
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ repeat: Infinity, duration: 2 }}
                  className="w-24 h-32 bg-white/20 backdrop-blur-xl rounded-lg flex items-center justify-center"
                >
                  <svg className="w-12 h-12 text-white/50" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
                  </svg>
                </motion.div>
              )}
            </div>
            
            {/* User Info */}
            <div className="flex-1 space-y-4">
              <div>
                <p className="text-xs opacity-60 uppercase">Nome</p>
                <p className="font-bold text-sm truncate">{name}</p>
              </div>
              <div>
                <p className="text-xs opacity-60 uppercase">CPF</p>
                <p className="font-mono text-sm">{cpf}</p>
              </div>
              <div>
                <p className="text-xs opacity-60 uppercase">Registro</p>
                <p className="font-mono text-sm">GC{Math.random().toString(36).substr(2, 9).toUpperCase()}</p>
              </div>
            </div>
          </div>
          
          {/* Authorization Badge */}
          <div className="my-6 p-3 bg-white/10 backdrop-blur-xl rounded-xl">
            <p className="text-xs text-center opacity-80">AUTORIZAÇÃO VÁLIDA</p>
            <p className="text-xs text-center font-bold">Cannabis Medicinal</p>
          </div>
          
          {/* Bottom Section */}
          <div className="flex justify-between items-end mt-auto">
            {/* QR Code */}
            <motion.div 
              animate={{ opacity: [0.3, 1, 0.3] }}
              transition={{ repeat: Infinity, duration: 3 }}
              className="w-20 h-20 bg-white/20 backdrop-blur-xl rounded-lg p-2"
            >
              <div className="w-full h-full bg-white/30 rounded grid grid-cols-3 gap-[2px] p-1">
                {[...Array(9)].map((_, i) => (
                  <div key={i} className="bg-greencard-primary rounded-sm" />
                ))}
              </div>
            </motion.div>
            
            {/* Validity */}
            <div className="text-right">
              <p className="text-xs opacity-80">VÁLIDA ATÉ</p>
              <p className="text-lg font-bold font-mono">{validity}</p>
              <div className="flex items-center gap-2 mt-2 justify-end">
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ repeat: Infinity, duration: 2 }}
                  className="w-2 h-2 bg-greencard-secondary rounded-full"
                />
                <span className="text-xs uppercase font-bold">Ativa</span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}