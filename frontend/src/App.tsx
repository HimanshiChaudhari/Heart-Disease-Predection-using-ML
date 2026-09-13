import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { Disclaimer } from './components/Disclaimer';
import { Home } from './pages/Home';
import { Prediction } from './pages/Prediction';
import { Results } from './pages/Results';
import { About } from './pages/About';
import { ModelPerformance } from './components/ModelPerformance';
import { PatientInput, PredictionResult as ResultType, ModelMetadata, PatientPreset } from './types/prediction';
import { predictHeartDisease, fetchHealth, fetchModelInfo, fetchSamplePatients } from './services/predictionApi';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('home');
  const [apiStatus, setApiStatus] = useState<string>('Connecting...');
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null);
  const [presets, setPresets] = useState<PatientPreset[]>([]);
  const [lastResult, setLastResult] = useState<ResultType | null>(null);

  useEffect(() => {
    // Initial fetch of API health, benchmarks, and test presets
    fetchHealth().then((h) => {
      setApiStatus(h.status === 'online' ? 'API Online & Ready' : 'Local ML Engine');
    });

    fetchModelInfo().then((m) => {
      setMetadata(m);
    });

    fetchSamplePatients().then((p) => {
      setPresets(p);
    });
  }, []);

  const handlePredict = async (data: PatientInput): Promise<ResultType> => {
    const res = await predictHeartDisease(data);
    setLastResult(res);
    return res;
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 selection:bg-rose-500 selection:text-white">
      
      {/* Top Health & Screening Disclaimer */}
      <Disclaimer variant="banner" />

      {/* Navigation Header */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        apiStatus={apiStatus}
      />

      {/* Main Content Pages */}
      <main className="flex-1">
        {activeTab === 'home' && (
          <Home
            onStartScreening={() => setActiveTab('prediction')}
            onViewModelBenchmark={() => setActiveTab('models')}
          />
        )}

        {activeTab === 'prediction' && (
          <Prediction
            onPredict={handlePredict}
            presets={presets}
          />
        )}

        {activeTab === 'results' && (
          <Results
            result={lastResult}
            onBackToForm={() => setActiveTab('prediction')}
          />
        )}

        {activeTab === 'models' && (
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
            <div className="border-b border-slate-800 pb-4">
              <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
                Model Evaluation & Clinical Transparency
              </h1>
              <p className="text-xs sm:text-sm text-slate-400 mt-1">
                Direct comparative performance metrics across all 5 models trained on the original project split.
              </p>
            </div>
            <ModelPerformance metadata={metadata} />
          </div>
        )}

        {activeTab === 'about' && (
          <About />
        )}
      </main>

      {/* Footer */}
      <Footer />

    </div>
  );
};

export default App;
