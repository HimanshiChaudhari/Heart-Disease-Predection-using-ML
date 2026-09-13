import React from 'react';
import { ModelMetadata } from '../types/prediction';
import { Cpu, CheckCircle2, TrendingUp, ShieldAlert, Award } from 'lucide-react';

interface ModelPerformanceProps {
  metadata: ModelMetadata | null;
}

export const ModelPerformance: React.FC<ModelPerformanceProps> = ({ metadata }) => {
  const models = metadata?.models || {};

  return (
    <div className="space-y-8">
      
      {/* Benchmark Summary Header */}
      <div className="glow-card rounded-2xl p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5 mb-6">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
              <h2 className="text-xl font-bold text-white">
                Machine Learning Model Benchmark
              </h2>
            </div>
            <p className="text-xs text-slate-400">
              Evaluated on 30% held-out test split (276 patient samples) with random_state=42
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-3 py-1 rounded-lg text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1.5">
              <Award className="w-4 h-4" />
              Flagship: Logistic Regression (~80.07% Acc)
            </span>
          </div>
        </div>

        {/* Table of Models */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs sm:text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase text-[11px] font-bold tracking-wider">
                <th className="pb-3 pr-4">Model Architecture</th>
                <th className="pb-3 px-3">Accuracy</th>
                <th className="pb-3 px-3">Precision</th>
                <th className="pb-3 px-3">Recall (Sensitivity)</th>
                <th className="pb-3 px-3">F1-Score</th>
                <th className="pb-3 px-3">ROC-AUC</th>
                <th className="pb-3 pl-3">Pipeline File</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {Object.keys(models).map((name) => {
                const m = models[name];
                const isPrimary = name === 'Logistic Regression';

                return (
                  <tr
                    key={name}
                    className={`hover:bg-slate-900/50 transition-colors ${
                      isPrimary ? 'bg-indigo-950/20 font-medium' : ''
                    }`}
                  >
                    <td className="py-3.5 pr-4">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-white">{name}</span>
                        {isPrimary && (
                          <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30">
                            Production
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-slate-400 mt-0.5">{m.description}</p>
                    </td>
                    <td className="py-3.5 px-3 font-mono font-bold text-slate-200">
                      {(m.accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="py-3.5 px-3 font-mono text-slate-300">
                      {(m.precision * 100).toFixed(2)}%
                    </td>
                    <td className="py-3.5 px-3 font-mono text-emerald-400 font-semibold">
                      {(m.recall * 100).toFixed(2)}%
                    </td>
                    <td className="py-3.5 px-3 font-mono text-slate-300">
                      {m.f1_score.toFixed(4)}
                    </td>
                    <td className="py-3.5 px-3 font-mono text-indigo-400">
                      {m.roc_auc ? `${(m.roc_auc * 100).toFixed(2)}%` : 'N/A'}
                    </td>
                    <td className="py-3.5 pl-3 font-mono text-[11px] text-slate-400">
                      <code>{m.filename}</code>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

      </div>

      {/* Clinical Screening Metric Rationale */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        
        <div className="glow-card rounded-2xl p-5 border-l-4 border-l-emerald-500">
          <div className="flex items-center gap-2 text-emerald-400 mb-2">
            <TrendingUp className="w-4 h-4" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              Recall / Sensitivity Priority
            </h3>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            In cardiovascular screening demonstrations, <strong>Recall</strong> measures the proportion of actual cardiac disease cases correctly identified. 
            Logistic Regression achieves <strong>83.56%</strong> recall, minimizing critical False Negatives.
          </p>
        </div>

        <div className="glow-card rounded-2xl p-5 border-l-4 border-l-indigo-500">
          <div className="flex items-center gap-2 text-indigo-400 mb-2">
            <Cpu className="w-4 h-4" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              Calibrated Probabilities
            </h3>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Logistic Regression leverages the natural sigmoid function to yield continuous, calibrated risk probabilities (0.0 to 1.0), 
            allowing the interface to display exact percentage gauges rather than rigid discrete labels.
          </p>
        </div>

        <div className="glow-card rounded-2xl p-5 border-l-4 border-l-rose-500">
          <div className="flex items-center gap-2 text-rose-400 mb-2">
            <ShieldAlert className="w-4 h-4" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              Decision Tree Alternative
            </h3>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            The Decision Tree was configured with <code>class_weight=&quot;balanced&quot;</code> in the original code, 
            actively penalizing misclassified disease cases and offering rule-based interpretability.
          </p>
        </div>

      </div>

    </div>
  );
};
