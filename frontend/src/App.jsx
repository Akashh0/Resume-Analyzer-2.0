import React, { useState, useRef } from 'react';
import Header from './components/Header';
import InputSection from './components/InputSection';
import MetricsRow from './components/MetricsRow';
import AuditReport from './components/AuditReport';

function App() {
  const [jobDescription, setJobDescription] = useState('');
  const [resumeFile, setResumeFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [apiResult, setApiResult] = useState(null);
  
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type === "application/pdf") {
      setResumeFile(file);
    } else alert("Please upload a valid PDF file.");
  };

  const triggerFileUpload = () => fileInputRef.current.click();

  const handleAnalyze = async () => {
    if (!jobDescription || !resumeFile) return;
    setIsAnalyzing(true);

    const formData = new FormData();
    formData.append("job_description", jobDescription);
    formData.append("resume_file", resumeFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/analyze", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setApiResult(data);
    } catch (error) {
      console.error("Connection Failed:", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    // Pure black background, minimalist font
    <div className="min-h-screen bg-[#000000] text-zinc-300 font-sans selection:bg-zinc-800">
      
      <div className="max-w-5xl mx-auto px-6 pb-24">
        <Header />

        <InputSection 
          jobDescription={jobDescription}
          setJobDescription={setJobDescription}
          resumeFile={resumeFile}
          fileInputRef={fileInputRef}
          handleFileChange={handleFileChange}
          triggerFileUpload={triggerFileUpload}
        />

        {/* Minimalist Action Button */}
        <div className="py-8 border-t border-zinc-900 flex justify-end">
          <button 
            onClick={handleAnalyze}
            disabled={!jobDescription || !resumeFile || isAnalyzing}
            className={`text-sm tracking-widest uppercase transition-all duration-300 ${
              !jobDescription || !resumeFile 
                ? 'text-zinc-800 cursor-not-allowed' 
                : isAnalyzing 
                  ? 'text-cyan-500 animate-pulse'
                  : 'text-white hover:text-cyan-400'
            }`}
          >
            {isAnalyzing ? "Processing..." : "Run Analysis →"}
          </button>
        </div>

        <MetricsRow jobDescription={jobDescription} resumeFile={resumeFile} apiResult={apiResult} />
        <AuditReport apiResult={apiResult} />

      </div>
    </div>
  );
}

export default App;