import React from 'react';
import { PredictionResult as ResultType } from '../types/prediction';
import { RiskGauge } from './RiskGauge';
import { Disclaimer } from './Disclaimer';
import { Printer, RefreshCw, AlertCircle, CheckCircle2, ChevronRight, Activity } from 'lucide-react';

interface PredictionResultProps {
  result: ResultType;
  onReset: () => void;
}

export const PredictionResult: React.FC<PredictionResultProps> = ({ result, onReset }) => {
  return (
    <div className="glow-card rounded-2xl p-6 sm:p-8 space-y-6">
      
      {/* Header with Active Model */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-rose-500" />
            Screening Assessment Results
          </h2>
          <p className="text-xs text-slate-400">
            Computed by Scikit-Learn reproducible inference pipeline
          </p>
        </div>
        <span className="px-2.5 py-1 rounded-md text-xs font-mono font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
          Model: {result.model}
        </span>
      </div>

      {/* Probability Gauge & High/Low Risk Classification */}
      <RiskGauge
        probability={result.probability}
        riskLevel={result.risk_level}
        predictionLabel={result.prediction_label}
      />

      {/* Mandatory Prominent Health Disclaimer */}
      <Disclaimer variant="card" />

      {/* Key Clinical Risk Factors Flagged */}
      <div className="space-y-3 pt-2">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
          <span>Clinical Biomarkers Identified</span>
          <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
            {result.risk_factors.length} flagged
          </span>
        </h3>

        {result.risk_factors.length === 0 ? (
          <div className="rounded-xl bg-slate-900/60 border border-slate-800 p-4 flex items-start gap-3">
            <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="text-sm font-semibold text-slate-200">No Elevated Clinical Risk Indicators</h4>
              <p className="text-xs text-slate-400 mt-0.5">
                The submitted patient vitals, ECG, and stress markers are within baseline non-ischemic tolerances according to the model parameters.
              </p>
            </div>
          </div>
        ) : (
          <div className="grid gap-2.5">
            {result.risk_factors.map((item, idx) => {
              const isHigh = item.level.toLowerCase().includes('high');
              const isMod = item.level.toLowerCase().includes('moderate') || item.level.toLowerCase().includes('cardio');

              return (
                <div
                  key={idx}
                  className={`rounded-xl p-3.5 border transition-all ${
                    isHigh
                      ? 'bg-rose-950/20 border-rose-800/30 text-rose-200'
                      : isMod
                      ? 'bg-amber-950/20 border-amber-800/30 text-amber-200'
                      : 'bg-slate-900/60 border-slate-800 text-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-bold text-white flex items-center gap-1.5">
                      <AlertCircle className={`w-3.5 h-3.5 ${isHigh ? 'text-rose-400' : isMod ? 'text-amber-400' : 'text-slate-400'}`} />
                      {item.factor}
                    </span>
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider ${
                        isHigh
                          ? 'bg-rose-500/20 text-rose-300'
                          : isMod
                          ? 'bg-amber-500/20 text-amber-300'
                          : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {item.level}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed pl-5">
                    {item.description}
                  </p>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row items-center gap-3 pt-4 border-t border-slate-800">
        <button
          onClick={() => window.print()}
          className="w-full sm:w-auto flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-xs sm:text-sm font-semibold bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-700 transition-colors"
        >
          <Printer className="w-4 h-4 text-slate-400" />
          Print / Save Assessment Summary
        </button>
        <button
          onClick={onReset}
          className="w-full sm:w-auto flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-xs sm:text-sm font-semibold bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-600/25 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          Screen Another Profile
        </button>
      </div>

    </div>
  );
};
