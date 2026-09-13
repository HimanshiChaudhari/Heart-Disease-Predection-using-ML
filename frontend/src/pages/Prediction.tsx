import React, { useState } from 'react';
import { PatientInput, PredictionResult as ResultType, PatientPreset } from '../types/prediction';
import { PredictionForm } from '../components/PredictionForm';
import { PredictionResult } from '../components/PredictionResult';
import { LoadingState } from '../components/LoadingState';
import { Disclaimer } from '../components/Disclaimer';
import { Activity, ShieldAlert, Sparkles } from 'lucide-react';

interface PredictionProps {
  onPredict: (data: PatientInput) => Promise<ResultType>;
  presets: PatientPreset[];
}

export const Prediction: React.FC<PredictionProps> = ({ onPredict, presets }) => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ResultType | null>(null);

  const handleFormSubmit = async (formData: PatientInput) => {
    setIsLoading(true);
    try {
      const res = await onPredict(formData);
      setResult(res);
      // Scroll smoothly to results on mobile
      window.scrollTo({ top: 180, behavior: 'smooth' });
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      
      {/* Page Header */}
      <div className="border-b border-slate-800 pb-5">
        <div className="flex items-center gap-2 mb-1.5">
          <Activity className="w-5 h-5 text-rose-500" />
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            Heart Disease Risk Screening
          </h1>
        </div>
        <p className="text-xs sm:text-sm text-slate-300 max-w-3xl">
          Enter the available health information below to receive an ML-based prediction.
        </p>
      </div>

      {/* Mandatory Disclaimer Box */}
      <Disclaimer variant="card" />

      {/* Main Grid: Form + Result Display */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* Left Column: Form (7 cols on lg) */}
        <div className="lg:col-span-7">
          <PredictionForm
            onSubmit={handleFormSubmit}
            isLoading={isLoading}
            presets={presets}
          />
        </div>

        {/* Right Column: Loading or Result Output (5 cols on lg) */}
        <div className="lg:col-span-5 space-y-6">
          {isLoading ? (
            <LoadingState message="Pre-processing clinical features and querying trained Scikit-Learn model..." />
          ) : result ? (
            <PredictionResult result={result} onReset={handleReset} />
          ) : (
            <div className="glow-card rounded-2xl p-8 text-center space-y-4">
              <div className="w-16 h-16 rounded-2xl bg-slate-900 border border-slate-800 mx-auto flex items-center justify-center text-slate-500">
                <Activity className="w-8 h-8 text-slate-600" />
              </div>
              <div>
                <h3 className="text-base font-bold text-white mb-1">Awaiting Patient Parameters</h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Complete the 13 clinical input fields or click one of the quick test presets above, then click <strong>Run Machine Learning Screening</strong>.
                </p>
              </div>
              <div className="pt-2 text-left bg-slate-900/50 rounded-xl p-3.5 border border-slate-800/80 text-[11px] text-slate-400 space-y-1">
                <div className="font-semibold text-slate-300 mb-1">Estimated Outputs Include:</div>
                <div>&bull; Model-estimated risk probability (0% – 100%)</div>
                <div>&bull; Risk classification label (&ldquo;Higher predicted risk&rdquo; or &ldquo;Lower predicted risk according to this model&rdquo;)</div>
                <div>&bull; Identified clinical risk factor contributions</div>
              </div>
            </div>
          )}
        </div>

      </div>

    </div>
  );
};
