import React from 'react';

function MetricsRow({ jobDescription, resumeFile, apiResult }) {
  const isReady = jobDescription && resumeFile;
  const matchScore = apiResult ? apiResult.match_score : "--";
  const tokens = apiResult ? apiResult.tokens : "--";

  return (
    <section className="grid grid-cols-3 gap-8 py-12 border-t border-zinc-900">
      
      <div className="flex flex-col">
        <p className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase mb-2 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-500"></span> Similarity to Role %
        </p>
        <p className={`text-6xl font-light tracking-tighter ${apiResult ? 'text-white' : 'text-zinc-800'}`}>
          {matchScore}<span className="text-2xl text-zinc-600">%</span>
        </p>
      </div>

      <div className="flex flex-col">
        <p className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase mb-2 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-500"></span> Status
        </p>
        <p className={`text-2xl font-light mt-2 ${apiResult ? 'text-white' : 'text-zinc-800'}`}>
          {apiResult ? "Overview Complete" : isReady ? "Ready to Run" : "Awaiting Data"}
        </p>
      </div>

      <div className="flex flex-col">
        <p className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase mb-2 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-fuchsia-500"></span> Words
        </p>
        <p className={`text-2xl font-light mt-2 ${apiResult ? 'text-white' : 'text-zinc-800'}`}>
          {tokens}
        </p>
      </div>

    </section>
  );
}

export default MetricsRow;