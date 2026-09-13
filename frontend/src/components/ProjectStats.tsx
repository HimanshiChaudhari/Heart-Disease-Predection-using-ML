import React from 'react';
import { Database, Binary, Cpu, Award, Target, CheckCircle } from 'lucide-react';

export const ProjectStats: React.FC = () => {
  const stats = [
    {
      label: 'Dataset Size',
      value: '920',
      unit: 'Patients',
      desc: 'Observed patient cases in heart_disease_uci.csv',
      icon: Database,
      color: 'text-indigo-400',
    },
    {
      label: 'Clinical Features',
      value: '13',
      unit: 'Variables',
      desc: 'Diagnostic inputs evaluated (id dropped)',
      icon: Binary,
      color: 'text-rose-400',
    },
    {
      label: 'Models Evaluated',
      value: '5',
      unit: 'Architectures',
      desc: 'Logistic Regression, Tree, KNN, Linear & RBF SVM',
      icon: Cpu,
      color: 'text-amber-400',
    },
    {
      label: 'Best Model',
      value: 'Logistic',
      unit: 'Regression',
      desc: 'Selected for top calibrated screening sensitivity',
      icon: Award,
      color: 'text-emerald-400',
    },
    {
      label: 'Best Accuracy',
      value: '80.07%',
      unit: 'Test Split',
      desc: 'Evaluated on 30% held-out test split (276 samples)',
      icon: Target,
      color: 'text-emerald-400',
    },
  ];

  return (
    <section className="py-12 border-b border-slate-800/80 bg-slate-950/40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        <div className="flex flex-col sm:flex-row sm:items-end justify-between mb-8 gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-xs font-semibold text-emerald-400 mb-2">
              <CheckCircle className="w-3.5 h-3.5" />
              <span>Verified Project Benchmarks</span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Project Performance & Dataset Statistics
            </h2>
          </div>
          <p className="text-xs sm:text-sm text-slate-400 max-w-md">
            Values extracted directly from the actual repository code (<code>Heart disease ml.py</code>) and dataset.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {stats.map((item, idx) => {
            const Icon = item.icon;
            const isHighlight = item.label === 'Best Accuracy' || item.label === 'Best Model';

            return (
              <div
                key={idx}
                className={`glow-card rounded-2xl p-5 flex flex-col justify-between border ${
                  isHighlight
                    ? 'border-emerald-500/30 bg-emerald-950/10'
                    : 'border-slate-800/80'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                      {item.label}
                    </span>
                    <Icon className={`w-4 h-4 ${item.color}`} />
                  </div>
                  <div className="flex items-baseline gap-1.5 mb-1">
                    <span className="text-2xl lg:text-3xl font-black font-mono tracking-tight text-white">
                      {item.value}
                    </span>
                    <span className="text-xs font-semibold text-slate-400 font-mono">
                      {item.unit}
                    </span>
                  </div>
                </div>

                <p className="text-[11px] text-slate-400 leading-relaxed mt-2 pt-2 border-t border-slate-800/60">
                  {item.desc}
                </p>
              </div>
            );
          })}
        </div>

      </div>
    </section>
  );
};
