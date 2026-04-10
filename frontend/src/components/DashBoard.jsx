import React from 'react';
import MetricCard from './MetricCard';
import AiConsultant from './AiConsultant';

function Dashboard() {
  return (
    <div className="flex-1 p-10 overflow-y-auto">
      <h2 className="text-3xl font-bold text-gray-800 mb-8">📊 Match Report</h2>
      
      {/* Row of Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <MetricCard 
          title="Match Confidence" 
          value="--%" 
          borderColor="border-blue-500" 
        />
        <MetricCard 
          title="Status" 
          value="Awaiting Data" 
          borderColor="border-indigo-500" 
        />
        <MetricCard 
          title="Resume Length" 
          value="-- Words" 
          borderColor="border-gray-300" 
        />
      </div>

      {/* Detailed Analysis Section */}
      <AiConsultant />
      
    </div>
  );
}

export default Dashboard;