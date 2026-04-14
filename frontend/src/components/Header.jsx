import React from 'react';

function Header() {
  return (
    <header className="pt-12 pb-8">
      <h1 className="text-4xl font-light tracking-tighter text-white flex items-center gap-3">
        <span className="w-3 h-3 rounded-full bg-gradient-to-r from-cyan-400 to-fuchsia-500"></span>
        Resume Analyser
      </h1>
      <p className="text-zinc-500 text-sm mt-4 font-mono tracking-wide">
        A semantic engine that checks the candidate profiles to targeted job description.
      </p>
    </header>
  );
}

export default Header;