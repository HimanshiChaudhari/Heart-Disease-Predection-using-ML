import React from 'react';
import { Database, ShieldCheck, FileText, Code2, HeartPulse, Sparkles, ExternalLink } from 'lucide-react';
import { Disclaimer } from '../components/Disclaimer';

export const About: React.FC = () => {
  const features = [
    { name: 'age', type: 'Numerical', desc: 'Patient age in years (range: 28 to 77)' },
    { name: 'gender', type: 'Categorical', desc: 'Biological sex: Female (0) or Male (1)' },
    { name: 'chest pain type', type: 'Categorical', desc: 'Typical Angina, Atypical Angina, Non-Anginal, or Asymptomatic' },
    { name: 'resting bps', type: 'Numerical', desc: 'Resting systolic blood pressure in mm Hg on hospital admission' },
    { name: 'cholestrol', type: 'Numerical', desc: 'Serum cholesterol in mg/dL' },
    { name: 'fasting blood sugar', type: 'Categorical', desc: 'Fasting blood sugar > 120 mg/dL: FALSE (0) or TRUE (1)' },
    { name: 'restecg result', type: 'Categorical', desc: 'Normal (0), ST-T Wave Abnormality (1), or LV Hypertrophy (2)' },
    { name: 'mx heart rate achieved', type: 'Numerical', desc: 'Maximum heart rate achieved during exercise stress test (thalach)' },
    { name: 'exang', type: 'Categorical', desc: 'Exercise-induced angina: FALSE (0) or TRUE (1)' },
    { name: 'oldpeak', type: 'Numerical', desc: 'ST depression induced by exercise relative to rest (in mm)' },
    { name: 'slope', type: 'Categorical', desc: 'Slope of peak exercise ST segment: Upsloping, Flat, or Downsloping' },
    { name: 'major vessels visible(ca)', type: 'Numerical', desc: 'Number of major vessels (0 to 3) colored by fluoroscopy' },
    { name: 'thal', type: 'Categorical', desc: 'Thallium cardiac perfusion scan: Normal, Fixed Defect, or Reversible Defect' },
    { name: 'target', type: 'Target Variable', desc: 'Converted to binary (0 = Absence, 1 = Disease detected across any vessel)' },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      
      {/* Header */}
      <div className="border-b border-slate-800 pb-5">
        <div className="flex items-center gap-2 mb-1.5">
          <Database className="w-5 h-5 text-indigo-400" />
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            Dataset & ML Pipeline Transparency
          </h1>
        </div>
        <p className="text-xs sm:text-sm text-slate-400 max-w-3xl">
          Comprehensive documentation of the UCI Heart Disease dataset, target definition, preprocessing, and model architecture.
        </p>
      </div>

      <Disclaimer variant="card" />

      {/* Dataset Overview Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glow-card rounded-2xl p-5">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
            Total Patient Records
          </span>
          <span className="text-2xl font-black font-mono text-white">920</span>
          <p className="text-[11px] text-slate-500 mt-1">From Cleveland, Hungary, Switzerland, & Long Beach VA</p>
        </div>

        <div className="glow-card rounded-2xl p-5">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
            Training Cohort
          </span>
          <span className="text-2xl font-black font-mono text-emerald-400">644 (70%)</span>
          <p className="text-[11px] text-slate-500 mt-1">Random seed = 42 (non-stratified split)</p>
        </div>

        <div className="glow-card rounded-2xl p-5">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
            Test Split
          </span>
          <span className="text-2xl font-black font-mono text-indigo-400">276 (30%)</span>
          <p className="text-[11px] text-slate-500 mt-1">Unseen benchmark evaluation split</p>
        </div>

        <div className="glow-card rounded-2xl p-5">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
            Target Formulation
          </span>
          <span className="text-xl font-black text-rose-400">Binary (0 / 1)</span>
          <p className="text-[11px] text-slate-500 mt-1">Converted via target &gt; 0 from 5 classes</p>
        </div>
      </div>

      {/* Target Definition Detailed Card */}
      <div className="glow-card rounded-2xl p-6 sm:p-8 space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <HeartPulse className="w-5 h-5 text-rose-400" />
          Target Definition & Binarization Logic
        </h2>
        <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
          In the original UCI Heart Disease dataset, the target column (<code className="text-rose-400 font-mono">target</code>) 
          recorded angiographic disease status with values from <code className="text-indigo-300">0</code> to <code className="text-indigo-300">4</code>:
        </p>
        <ul className="list-disc list-inside text-xs text-slate-400 space-y-1 pl-2">
          <li><strong>0</strong>: Absence of significant coronary artery disease (&lt; 50% stenosis)</li>
          <li><strong>1, 2, 3, 4</strong>: Presence of significant coronary stenosis (&gt; 50% diameter narrowing across 1, 2, 3, or 4 major coronary arteries)</li>
        </ul>
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <span className="text-xs font-mono font-semibold text-slate-400 block mb-1">Original Project Transformation (Line 20 of Heart disease ml.py):</span>
          <code className="text-xs font-mono text-emerald-400">data[&apos;target&apos;] = data[&apos;target&apos;].apply(lambda x: 1 if x &gt; 0 else 0)</code>
        </div>
      </div>

      {/* 13 Feature Definitions Table */}
      <div className="glow-card rounded-2xl p-6 sm:p-8">
        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <FileText className="w-5 h-5 text-indigo-400" />
          13 Diagnostic Clinical Features Dictionary
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs sm:text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase text-[11px] font-bold tracking-wider">
                <th className="pb-3 pr-4">Feature Name</th>
                <th className="pb-3 px-3">Type</th>
                <th className="pb-3 pl-3">Clinical Description & Observed Values</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {features.map((f) => (
                <tr key={f.name} className="hover:bg-slate-900/50">
                  <td className="py-3 pr-4 font-mono font-bold text-slate-200">{f.name}</td>
                  <td className="py-3 px-3 text-xs">
                    <span className={`px-2 py-0.5 rounded font-mono text-[10px] font-semibold ${
                      f.type === 'Numerical' ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20' :
                      f.type === 'Categorical' ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' :
                      'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                    }`}>
                      {f.type}
                    </span>
                  </td>
                  <td className="py-3 pl-3 text-xs text-slate-400">{f.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Origin & Credit */}
      <div className="glow-card rounded-2xl p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h3 className="text-sm font-bold text-white mb-1">Source of Truth & Original Codebase</h3>
          <p className="text-xs text-slate-400">
            Repository: <span className="font-mono text-slate-300">HimanshiChaudhari/Heart-Disease-Predection-using-ML</span>
          </p>
        </div>
        <div className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono text-slate-300">
          Scikit-Learn 1.2+ &bull; FastAPI 0.100+ &bull; React 18
        </div>
      </div>

    </div>
  );
};
