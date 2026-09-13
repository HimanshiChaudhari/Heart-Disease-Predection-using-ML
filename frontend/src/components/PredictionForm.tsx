import React, { useState } from 'react';
import { PatientInput, PatientPreset } from '../types/prediction';
import { validatePatientInput, ValidationError } from '../utils/validation';
import { Activity, Sparkles, Check, AlertCircle, Play } from 'lucide-react';

interface PredictionFormProps {
  onSubmit: (data: PatientInput) => void;
  isLoading: boolean;
  presets: PatientPreset[];
}

export const PredictionForm: React.FC<PredictionFormProps> = ({ onSubmit, isLoading, presets }) => {
  const [formData, setFormData] = useState<PatientInput>({
    age: 55,
    sex: 'Male',
    chest_pain: 'asymptomatic',
    resting_bp: 135,
    cholesterol: 240,
    fasting_blood_sugar: 'FALSE',
    resting_ecg: 'normal',
    max_heart_rate: 148,
    exercise_angina: 'FALSE',
    oldpeak: 1.2,
    st_slope: 'flat',
    major_vessels: 0,
    thalassemia: 'normal',
    selected_model: 'Logistic Regression',
  });

  const [errors, setErrors] = useState<ValidationError[]>([]);

  const handleInputChange = (field: keyof PatientInput, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    // Clear error for that field
    setErrors((prev) => prev.filter((e) => e.field !== field));
  };

  const handlePresetSelect = (preset: PatientPreset) => {
    setFormData({
      ...preset.data,
      selected_model: formData.selected_model || 'Logistic Regression',
    });
    setErrors([]);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const validationErrors = validatePatientInput(formData);
    if (validationErrors.length > 0) {
      setErrors(validationErrors);
      return;
    }
    onSubmit(formData);
  };

  const getFieldError = (field: keyof PatientInput) => {
    return errors.find((e) => e.field === field)?.message;
  };

  return (
    <div className="glow-card rounded-2xl p-6 sm:p-8 space-y-6">
      
      {/* Preset Quick Fill Bar */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
        <div className="flex items-center justify-between mb-2.5">
          <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            Quick Test Case Presets
          </span>
          <span className="text-[11px] text-slate-400">Click to populate all 13 parameters</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {presets.map((p, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => handlePresetSelect(p)}
              className="text-left p-2.5 rounded-lg border border-slate-800 bg-slate-950/60 hover:bg-slate-800/60 hover:border-slate-700 transition-all group"
            >
              <div className="flex items-center gap-1.5 mb-1">
                <span className={`w-2 h-2 rounded-full ${
                  idx === 0 ? 'bg-emerald-400' : idx === 1 ? 'bg-amber-400' : 'bg-rose-400'
                }`} />
                <span className="text-xs font-bold text-slate-200 group-hover:text-white">
                  {p.profile_name}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                {p.description}
              </p>
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">

        {/* Section 1: Demographics */}
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-indigo-400 mb-3 flex items-center gap-1.5">
            <span>1. Patient Demographics</span>
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Age (Years) <span className="text-rose-400">*</span>
              </label>
              <input
                type="number"
                value={formData.age}
                onChange={(e) => handleInputChange('age', parseFloat(e.target.value) || 0)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                min="18"
                max="120"
                required
              />
              {getFieldError('age') && (
                <p className="text-xs text-rose-400 mt-1 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" /> {getFieldError('age')}
                </p>
              )}
              <span className="text-[10px] text-slate-400">UCI Dataset Range: 28 – 77 yrs</span>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Biological Sex <span className="text-rose-400">*</span>
              </label>
              <select
                value={formData.sex}
                onChange={(e) => handleInputChange('sex', e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              >
                <option value="Male">Male (1)</option>
                <option value="Female">Female (0)</option>
              </select>
            </div>

          </div>
        </div>

        {/* Section 2: Vitals & Blood Tests */}
        <div className="pt-2 border-t border-slate-800/80">
          <h3 className="text-xs font-bold uppercase tracking-wider text-indigo-400 mb-3">
            2. Hemodynamics & Blood Chemistry
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Resting Blood Pressure <span className="text-rose-400">*</span>
              </label>
              <div className="relative">
                <input
                  type="number"
                  value={formData.resting_bp}
                  onChange={(e) => handleInputChange('resting_bp', parseFloat(e.target.value) || 0)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 pr-12 text-sm text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                  min="70"
                  max="250"
                  required
                />
                <span className="absolute right-3 top-2.5 text-xs text-slate-400 pointer-events-none">mmHg</span>
              </div>
              {getFieldError('resting_bp') && (
                <p className="text-xs text-rose-400 mt-1 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" /> {getFieldError('resting_bp')}
                </p>
              )}
              <span className="text-[10px] text-slate-400">Normal: &lt; 120 mmHg</span>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Serum Cholesterol <span className="text-rose-400">*</span>
              </label>
              <div className="relative">
                <input
                  type="number"
                  value={formData.cholesterol}
                  onChange={(e) => handleInputChange('cholesterol', parseFloat(e.target.value) || 0)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 pr-12 text-sm text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                  min="80"
                  max="700"
                  required
                />
                <span className="absolute right-3 top-2.5 text-xs text-slate-400 pointer-events-none">mg/dL</span>
              </div>
              {getFieldError('cholesterol') && (
                <p className="text-xs text-rose-400 mt-1 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" /> {getFieldError('cholesterol')}
                </p>
              )}
              <span className="text-[10px] text-slate-400">Desirable: &lt; 200 mg/dL</span>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Fasting Blood Sugar &gt; 120
              </label>
              <select
                value={String(formData.fasting_blood_sugar)}
                onChange={(e) => handleInputChange('fasting_blood_sugar', e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              >
                <option value="FALSE">FALSE (&le; 120 mg/dL)</option>
                <option value="TRUE">TRUE (&gt; 120 mg/dL)</option>
              </select>
              <span className="text-[10px] text-slate-400">Marker for diabetes</span>
            </div>

          </div>
        </div>

        {/* Section 3: Symptoms & Resting ECG */}
        <div className="pt-2 border-t border-slate-800/80">
          <h3 className="text-xs font-bold uppercase tracking-wider text-indigo-400 mb-3">
            3. Chest Pain & Resting Electrocardiogram
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Chest Pain Classification <span className="text-rose-400">*</span>
              </label>
              <select
                value={formData.chest_pain}
                onChange={(e) => handleInputChange('chest_pain', e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              >
                <option value="typical angina">Typical Angina (Substernal chest pressure on exertion)</option>
                <option value="atypical angina">Atypical Angina (Non-classic presentation)</option>
                <option value="non-anginal">Non-Anginal Discomfort</option>
                <option value="asymptomatic">Asymptomatic (Silent ischemia risk)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Resting ECG Results <span className="text-rose-400">*</span>
              </label>
              <select
                value={formData.resting_ecg}
                onChange={(e) => handleInputChange('resting_ecg', e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              >
                <option value="normal">Normal (0)</option>
                <option value="st-t abnormality">ST-T Wave Abnormality (T-inversion/elevation) (1)</option>
                <option value="lv hypertrophy">Left Ventricular Hypertrophy (Estes' criteria) (2)</option>
              </select>
            </div>

          </div>
        </div>

        {/* Section 4: Exercise Stress Test & Imaging */}
        <div className="pt-2 border-t border-slate-800/80">
          <h3 className="text-xs font-bold uppercase tracking-wider text-indigo-400 mb-3">
            4. Exercise Stress & Myocardial Imaging
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
            
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Max Heart Rate Achieved
              </label>
              <div className="relative">
                <input
                  type="number"
                  value={formData.max_heart_rate}
                  onChange={(e) => handleInputChange('max_heart_rate', parseFloat(e.target.value) || 0)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 pr-12 text-sm text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                  min="60"
                  max="230"
                  required
                />
                <span className="absolute right-3 top-2.5 text-xs text-slate-400 pointer-events-none">bpm</span>
              </div>
              <span className="text-[10px] text-slate-400">Peak stress thalach</span>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Exercise-Induced Angina
              </label>
              <select
                value={String(formData.exercise_angina)}
                onChange={(e) => handleInputChange('exercise_angina', e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              >
                <option value="FALSE">FALSE (No angina provoked)</option>
                <option value="TRUE">TRUE (Angina induced by exertion)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                ST Depression (Oldpeak)
              </label>
              <div className="relative">
                <input
                  type="number"
                  step="0.1"
                  value={formData.oldpeak}
                  onChange={(e) => handleInputChange('oldpeak', parseFloat(e.target.value) || 0)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 pr-10 text-sm text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                  min="-2.0"
                  max="8.0"
                  required
                />
                <span className="absolute right-3 top-2.5 text-xs text-slate-400 pointer-events-none">mm</span>
              </div>
              <span className="text-[10px] text-slate-400">Depression relative to rest</span>
            </div>

          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Peak ST Segment Slope
              </label>
              <select
                value={formData.st_slope}
                onChange={(e) => handleInputChange('st_slope', e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              >
                <option value="upsloping">Upsloping (Typically benign)</option>
                <option value="flat">Flat (Equivocal / Ischemic)</option>
                <option value="downsloping">Downsloping (High Ischemia Indicator)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Major Vessels Visible (ca)
              </label>
              <select
                value={formData.major_vessels}
                onChange={(e) => handleInputChange('major_vessels', parseFloat(e.target.value) || 0)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              >
                <option value={0}>0 Vessels (Uncompromised)</option>
                <option value={1}>1 Vessel Narrowed</option>
                <option value={2}>2 Vessels Narrowed</option>
                <option value={3}>3 Vessels Narrowed</option>
              </select>
              <span className="text-[10px] text-slate-400">Fluoroscopy colored count</span>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Thallium Perfusion Defect
              </label>
              <select
                value={formData.thalassemia}
                onChange={(e) => handleInputChange('thalassemia', e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              >
                <option value="normal">Normal Perfusion</option>
                <option value="fixed defect">Fixed Defect (Old scar / Infarction)</option>
                <option value="reversable defect">Reversible Defect (Active ischemia)</option>
              </select>
            </div>

          </div>
        </div>

        {/* Model Pipeline Selector */}
        <div className="pt-3 border-t border-slate-800/80">
          <div className="bg-slate-900/80 border border-slate-700/60 rounded-xl p-3 sm:p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <label className="block text-xs font-bold text-white uppercase tracking-wider">
                Machine Learning Evaluation Model
              </label>
              <p className="text-[11px] text-slate-400">
                Choose the model pipeline for computing probability
              </p>
            </div>
            <select
              value={formData.selected_model}
              onChange={(e) => handleInputChange('selected_model', e.target.value)}
              className="bg-slate-950 border border-slate-700 rounded-lg px-3 py-1.5 text-xs sm:text-sm font-semibold text-indigo-300 focus:outline-none focus:border-indigo-500"
            >
              <option value="Logistic Regression">Logistic Regression (Flagship &bull; ~80% Acc)</option>
              <option value="Decision Tree">Decision Tree (Balanced Weights &bull; High Recall)</option>
              <option value="KNN (k=21)">K-Nearest Neighbors (k=21)</option>
              <option value="Linear SVM">Support Vector Classifier (Linear Kernel)</option>
              <option value="RBF SVM">Support Vector Classifier (RBF Kernel)</option>
            </select>
          </div>
        </div>

        {/* Submit Actions */}
        <div className="pt-2 flex flex-col sm:flex-row items-center gap-3">
          <button
            type="submit"
            disabled={isLoading}
            className="w-full sm:flex-1 py-3 px-6 rounded-xl font-bold text-sm bg-gradient-to-r from-rose-500 to-indigo-600 hover:from-rose-600 hover:to-indigo-700 text-white shadow-lg shadow-rose-500/25 flex items-center justify-center gap-2 transition-all disabled:opacity-50"
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                <span>Computing Risk Probability...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Run Machine Learning Screening</span>
              </>
            )}
          </button>
        </div>

      </form>
    </div>
  );
};
