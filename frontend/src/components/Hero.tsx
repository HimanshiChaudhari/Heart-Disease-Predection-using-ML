import React from 'react';
import { Activity, ArrowRight, Sparkles, Heart, Shield, Cpu, Binary } from 'lucide-react';

interface HeroProps {
  onStartScreening: () => void;
  onHowItWorks: () => void;
}

export const Hero: React.FC<HeroProps> = ({ onStartScreening, onHowItWorks }) => {
  return (
    <section className="relative overflow-hidden pt-8 pb-16 lg:pt-12 lg:pb-20 border-b border-slate-800/80">
      
      {/* Background ambient lighting */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-rose-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute top-1/3 right-1/4 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          
          {/* Left Column: Heading, Subtitle, CTAs */}
          <div className="lg:col-span-7 text-left space-y-6">
            
            {/* Tagline Badge */}
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900/90 border border-slate-700/80 text-xs font-semibold text-slate-300 shadow-sm">
              <span className="flex h-2 w-2 rounded-full bg-rose-500 animate-pulse" />
              <span>Machine Learning Heart Disease Risk Screening</span>
              <span className="text-slate-600">&bull;</span>
              <span className="text-indigo-400 font-mono">UCI Dataset Engine</span>
            </div>

            {/* Exact Required Heading */}
            <h1 className="text-4xl sm:text-5xl xl:text-6xl font-extrabold tracking-tight text-white leading-[1.12]">
              Understand Your Heart Health Risk with Machine Learning
            </h1>

            {/* Exact Required Subtitle */}
            <p className="text-base sm:text-lg text-slate-300 leading-relaxed max-w-xl font-normal">
              An ML-powered screening tool that analyzes selected cardiovascular health indicators and provides a model-based prediction.
            </p>

            {/* Exact Required Buttons: "Check Your Risk" and "How It Works" */}
            <div className="pt-2 flex flex-col sm:flex-row items-center gap-3.5">
              <button
                onClick={onStartScreening}
                className="w-full sm:w-auto px-7 py-3.5 rounded-xl font-bold text-sm bg-gradient-to-r from-rose-500 to-indigo-600 hover:from-rose-600 hover:to-indigo-700 text-white shadow-xl shadow-rose-500/25 flex items-center justify-center gap-2 transition-all hover:scale-[1.02] active:scale-[0.98]"
              >
                <Heart className="w-4 h-4 fill-current" />
                <span>Check Your Risk</span>
                <ArrowRight className="w-4 h-4 ml-0.5" />
              </button>

              <button
                onClick={onHowItWorks}
                className="w-full sm:w-auto px-6 py-3.5 rounded-xl font-semibold text-sm bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-800 flex items-center justify-center gap-2 transition-all hover:border-slate-700"
              >
                <Activity className="w-4 h-4 text-indigo-400" />
                <span>How It Works</span>
              </button>
            </div>

            {/* Trust and Clinical Ethics Mini Pills */}
            <div className="pt-4 flex flex-wrap items-center gap-4 text-xs text-slate-400 border-t border-slate-800/80">
              <div className="flex items-center gap-1.5">
                <Shield className="w-4 h-4 text-emerald-400" />
                <span>Educational Screening</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Cpu className="w-4 h-4 text-indigo-400" />
                <span>5 Evaluated ML Models</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Binary className="w-4 h-4 text-rose-400" />
                <span>Calibrated Probabilities</span>
              </div>
            </div>

          </div>

          {/* Right Column: Custom Professional SVG/CSS AI + Healthcare Visual */}
          <div className="lg:col-span-5 relative flex items-center justify-center">
            
            <div className="relative w-full max-w-md aspect-square rounded-3xl glow-card p-6 overflow-hidden flex flex-col justify-between border border-slate-700/60 shadow-2xl">
              
              {/* Top Bar of the Visual Interface */}
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <div className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-ping" />
                  <span className="text-xs font-mono font-bold text-slate-200 uppercase tracking-wider">
                    Neural Cardio Engine
                  </span>
                </div>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  Pipeline Online
                </span>
              </div>

              {/* Center SVG: Anatomical AI Heart Mesh & Live Electrocardiogram (ECG) */}
              <div className="relative flex-1 flex items-center justify-center my-4">
                
                {/* Glowing Circular Aura */}
                <div className="absolute w-44 h-44 rounded-full bg-gradient-to-tr from-rose-500/15 via-indigo-500/15 to-transparent blur-xl pointer-events-none" />

                {/* SVG Graphics Canvas */}
                <svg className="w-full h-48" viewBox="0 0 400 200" fill="none" xmlns="http://www.w3.org/2000/svg">
                  
                  {/* Background Grid Pattern */}
                  <defs>
                    <pattern id="ecg-grid" width="20" height="20" patternUnits="userSpaceOnUse">
                      <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(51, 65, 85, 0.25)" strokeWidth="0.5" />
                    </pattern>
                    <linearGradient id="ecg-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#818cf8" stopOpacity="0.2" />
                      <stop offset="30%" stopColor="#6366f1" stopOpacity="0.8" />
                      <stop offset="60%" stopColor="#f43f5e" stopOpacity="1" />
                      <stop offset="85%" stopColor="#f43f5e" stopOpacity="0.8" />
                      <stop offset="100%" stopColor="#818cf8" stopOpacity="0.2" />
                    </linearGradient>
                    <linearGradient id="heart-glow" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#f43f5e" />
                      <stop offset="100%" stopColor="#4f46e5" />
                    </linearGradient>
                  </defs>

                  {/* Grid Background */}
                  <rect width="400" height="200" fill="url(#ecg-grid)" />

                  {/* Center Heart Line Art Mesh */}
                  <path
                    d="M 200 60 C 185 35 150 35 135 60 C 115 95 165 145 200 170 C 235 145 285 95 265 60 C 250 35 215 35 200 60 Z"
                    stroke="url(#heart-glow)"
                    strokeWidth="1.5"
                    strokeDasharray="4 3"
                    fill="rgba(244, 63, 94, 0.04)"
                  />

                  {/* AI Neural Network Node Connectors */}
                  <line x1="135" y1="60" x2="170" y2="100" stroke="#6366f1" strokeWidth="1" strokeOpacity="0.4" />
                  <line x1="265" y1="60" x2="230" y2="100" stroke="#6366f1" strokeWidth="1" strokeOpacity="0.4" />
                  <line x1="170" y1="100" x2="200" y2="135" stroke="#f43f5e" strokeWidth="1" strokeOpacity="0.5" />
                  <line x1="230" y1="100" x2="200" y2="135" stroke="#f43f5e" strokeWidth="1" strokeOpacity="0.5" />
                  <line x1="170" y1="100" x2="230" y2="100" stroke="#818cf8" strokeWidth="1" strokeOpacity="0.3" />

                  {/* Neural Nodes */}
                  <circle cx="135" cy="60" r="3.5" fill="#818cf8" />
                  <circle cx="265" cy="60" r="3.5" fill="#818cf8" />
                  <circle cx="170" cy="100" r="4" fill="#f43f5e" />
                  <circle cx="230" cy="100" r="4" fill="#f43f5e" />
                  <circle cx="200" cy="135" r="4.5" fill="#22c55e" />

                  {/* Animated Continuous ECG Heart Rhythm Trace */}
                  <path
                    d="M 10 100 L 70 100 L 85 90 L 95 110 L 105 100 L 130 100 L 145 40 L 160 160 L 175 80 L 185 115 L 195 100 L 230 100 L 245 40 L 260 160 L 275 80 L 285 115 L 295 100 L 320 100 L 335 90 L 345 110 L 355 100 L 390 100"
                    stroke="url(#ecg-gradient)"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />

                  {/* Sweeping Pulse Glow Dot */}
                  <circle cx="260" cy="160" r="5" fill="#ffffff" filter="drop-shadow(0 0 6px #f43f5e)" />
                </svg>

                {/* Floating Micro Diagnostic Tags */}
                <div className="absolute -top-1 -left-2 bg-slate-900/90 border border-slate-700/80 px-2.5 py-1 rounded-lg text-[10px] font-mono text-slate-300 shadow-md">
                  ST-Depression (Oldpeak): <span className="text-rose-400 font-bold">1.2mm</span>
                </div>
                <div className="absolute -bottom-1 -right-2 bg-slate-900/90 border border-slate-700/80 px-2.5 py-1 rounded-lg text-[10px] font-mono text-slate-300 shadow-md">
                  Perfusion Scan: <span className="text-emerald-400 font-bold">Normal</span>
                </div>
              </div>

              {/* Bottom Real-time Model Inference Box */}
              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3 flex items-center justify-between text-xs">
                <div>
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                    Active Architecture
                  </span>
                  <span className="font-semibold text-white">Logistic Regression Pipeline</span>
                </div>
                <div className="text-right font-mono">
                  <span className="text-[10px] text-slate-400 block">Screening Benchmark</span>
                  <span className="text-emerald-400 font-bold font-mono">80.07% Accuracy</span>
                </div>
              </div>

            </div>

          </div>

        </div>

      </div>
    </section>
  );
};
