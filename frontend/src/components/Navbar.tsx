import React from 'react';
import { Activity, Cpu, Sparkles, Heart } from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  apiStatus: string;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, apiStatus }) => {
  return (
    <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Brand */}
        <div 
          className="flex items-center gap-3 cursor-pointer group"
          onClick={() => setActiveTab('home')}
        >
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 to-indigo-600 p-0.5 shadow-lg shadow-rose-500/20 group-hover:shadow-rose-500/30 transition-all">
            <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <Activity className="w-5 h-5 text-rose-400 group-hover:scale-110 transition-transform" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-lg font-extrabold tracking-tight text-white font-sans">
                Heart<span className="text-rose-500">AI</span>
              </span>
              <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                <Cpu className="w-2.5 h-2.5 mr-1" />
                ML Screening
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-medium hidden sm:block">
              Machine Learning Heart Disease Risk Screening
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 sm:gap-2">
          <button
            onClick={() => setActiveTab('home')}
            className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
              activeTab === 'home'
                ? 'bg-slate-800 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            Overview
          </button>
          <button
            onClick={() => setActiveTab('prediction')}
            className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all flex items-center gap-1.5 ${
              activeTab === 'prediction' || activeTab === 'results'
                ? 'bg-rose-500 text-white shadow-md shadow-rose-500/25'
                : 'text-slate-300 hover:text-white bg-slate-900 border border-slate-800'
            }`}
          >
            <Heart className="w-3.5 h-3.5 fill-current" />
            Risk Screening
          </button>
          <button
            onClick={() => setActiveTab('models')}
            className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
              activeTab === 'models'
                ? 'bg-slate-800 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            Model Benchmark
          </button>
          <button
            onClick={() => setActiveTab('about')}
            className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all hidden md:block ${
              activeTab === 'about'
                ? 'bg-slate-800 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            UCI Dataset
          </button>
        </nav>

        {/* Health Status Indicator */}
        <div className="hidden lg:flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs font-mono text-slate-300">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <span>{apiStatus}</span>
        </div>

      </div>
    </header>
  );
};
