import React from 'react';

// Pre-defined job descriptions
const JOB_TEMPLATES = {
  ai: "We are seeking an AI Application Engineer to join our core product team. You will be responsible for building scalable web applications that integrate advanced machine learning models.\n\nKey Responsibilities:\n- Architect robust backend APIs using Python (Django or FastAPI).\n- Build responsive user interfaces using React.js.\n- Integrate LLMs and NLP models into production environments.\n\nRequirements:\n- 3+ years of software engineering experience.\n- Proficiency in Python, React, and Django.\n- Familiarity with AI/ML tools (TensorFlow, PyTorch, Hugging Face).",
  
  frontend: "Looking for a Senior Frontend Developer to lead the complete overhaul of our customer-facing web platform. You will work closely with design to translate Figma prototypes into pixel-perfect apps.\n\nKey Responsibilities:\n- Lead development using React.js and Next.js.\n- Implement complex global state management (Redux/Zustand).\n- Ensure WCAG accessibility standards.\n\nRequirements:\n- 5+ years dedicated frontend experience.\n- Expert in React, HTML5, CSS3, Tailwind CSS.\n- Strong UI/UX principles. (Backend experience not required).",
  
  data: "Hiring a Data Engineer to build and scale massive data pipelines. You will extract, transform, and load terabytes of data daily.\n\nKey Responsibilities:\n- Design scalable ETL/ELT pipelines.\n- Manage cloud data warehouses (Snowflake, Redshift).\n- Orchestrate workflows using Apache Airflow.\n\nRequirements:\n- Advanced SQL optimization skills.\n- Strong Python programming (specifically Pandas).\n- Experience with Apache Spark and Hadoop."
};

function InputSection({ 
  jobDescription, 
  setJobDescription, 
  resumeFile, 
  fileInputRef, 
  handleFileChange, 
  triggerFileUpload 
}) {
  return (
    <section className="grid grid-cols-1 md:grid-cols-2 gap-12 py-8 border-t border-zinc-900">
      
      {/* Target Role Input */}
      <div className="flex flex-col group">
        
        {/* Header with Preset Buttons */}
        <div className="flex flex-wrap items-center justify-between mb-4 gap-y-2">
          <label className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase">
            Target Role Vector
          </label>
          
          <div className="flex flex-wrap items-center gap-2 sm:gap-3 text-[10px] font-mono uppercase tracking-widest">
            <span className="text-zinc-700 whitespace-nowrap">Load Preset:</span>
            <button 
              onClick={() => setJobDescription(JOB_TEMPLATES.ai)} 
              className="text-cyan-500/50 hover:text-cyan-400 transition-colors whitespace-nowrap"
            >
              [ AI_Eng ]
            </button>
            <button 
              onClick={() => setJobDescription(JOB_TEMPLATES.frontend)} 
              className="text-fuchsia-500/50 hover:text-fuchsia-400 transition-colors whitespace-nowrap"
            >
              [ UI_Dev ]
            </button>
            <button 
              onClick={() => setJobDescription(JOB_TEMPLATES.data)} 
              className="text-amber-500/50 hover:text-amber-400 transition-colors whitespace-nowrap"
            >
              [ Data ]
            </button>
            {/* Clear Button */}
            <button 
              onClick={() => setJobDescription('')} 
              className="text-zinc-600 hover:text-red-400 transition-colors sm:ml-2 whitespace-nowrap"
              title="Clear text"
            >
              [ x ]
            </button>
          </div>
        </div>

        <textarea
          className="w-full h-32 bg-transparent resize-none outline-none text-zinc-300 placeholder:text-zinc-800 font-light leading-relaxed border-b border-zinc-900 focus:border-cyan-500 transition-colors"
          placeholder="Paste the complete job description here, or load a preset above..."
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
        ></textarea>
      </div>

      {/* Upload Matrix */}
      <div className="flex flex-col">
        <label className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase mb-4">
          Candidate Matrix
        </label>
        
        <div className="flex-1 flex flex-col justify-center">
          <input 
            type="file" 
            accept=".pdf" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            className="hidden" 
          />

          <div className="flex items-center justify-between border-b border-zinc-900 pb-4">
            <div>
              {resumeFile ? (
                <p className="text-fuchsia-400 font-light truncate max-w-[200px] sm:max-w-[300px]">{resumeFile.name}</p>
              ) : (
                <p className="text-zinc-800 font-light italic">No PDF linked.</p>
              )}
            </div>

            <button 
              onClick={triggerFileUpload}
              className="text-xs font-mono uppercase tracking-widest text-zinc-400 hover:text-white transition-colors whitespace-nowrap"
            >
              {resumeFile ? "[ Change ]" : "[ Upload ]"}
            </button>
          </div>
        </div>
      </div>
      
    </section>
  );
}

export default InputSection;