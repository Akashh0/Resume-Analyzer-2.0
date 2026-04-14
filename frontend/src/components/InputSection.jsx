import React from 'react';

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
        <label className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase mb-4">Target Role Description</label>
        <textarea
          className="w-full h-32 bg-transparent resize-none outline-none text-zinc-300 placeholder:text-zinc-800 font-light leading-relaxed border-b border-zinc-900 focus:border-cyan-500 transition-colors"
          placeholder="We are seeking a Full-Stack Software Engineer with <number> years of experience in React, Node.js, and PostgreSQL to architect scalable APIs and build responsive web interfaces."
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
        ></textarea>
      </div>

      {/* Upload Matrix */}
      <div className="flex flex-col">
        <label className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase mb-4">Candidate Resume (One File only allowed)</label>
        
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
                <p className="text-fuchsia-400 font-light">{resumeFile.name}</p>
              ) : (
                <p className="text-zinc-800 font-light italic">No PDF linked.</p>
              )}
            </div>

            <button 
              onClick={triggerFileUpload}
              className="text-xs font-mono uppercase tracking-widest text-zinc-400 hover:text-white transition-colors"
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