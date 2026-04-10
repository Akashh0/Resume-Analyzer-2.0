import React from 'react';

function Sidebar() {
  return (
    <div className="w-80 bg-white shadow-xl p-6 flex flex-col border-r border-gray-200">
      {/* Logo / Title */}
      <div className="flex items-center gap-3 mb-8">
        <span className="text-3xl">🚀</span>
        <h1 className="text-2xl font-bold text-gray-800">Resume Architect</h1>
      </div>
      
      {/* Input Areas */}
      <div className="flex-1 space-y-6">
        <div>
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
            1. Profile Settings
          </h3>
          <div className="bg-gray-50 p-4 rounded-lg border border-gray-100">
             <p className="text-sm text-gray-500 italic">Settings controls will go here...</p>
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
            2. Job Description
          </h3>
          <textarea 
            className="w-full h-32 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none resize-none text-sm"
            placeholder="Paste JD here..."
          ></textarea>
        </div>

        {/* Upload Button */}
        <div>
          <button className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition duration-200 shadow-md">
            Upload PDF Resume
          </button>
        </div>
      </div>
    </div>
  );
}

export default Sidebar;