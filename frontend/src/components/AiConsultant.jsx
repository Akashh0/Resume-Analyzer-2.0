import React from 'react';

function AiConsultant() {
  return (
    <div className="bg-white rounded-xl shadow-sm p-8 border border-gray-100 mt-8">
      <h3 className="text-xl font-bold text-gray-800 mb-4">💡 AI Consultant</h3>
      
      {/* Empty State Placeholder */}
      <div className="text-gray-500 text-center py-10 border-2 border-dashed border-gray-200 rounded-lg bg-gray-50">
        Upload a resume and job description to generate your targeted analysis.
      </div>
    </div>
  );
}

export default AiConsultant;