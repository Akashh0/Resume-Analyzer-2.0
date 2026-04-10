import React from 'react';
import ReactMarkdown from 'react-markdown';

function AuditReport({ apiResult }) {
  if (!apiResult) return null; // In true minimalism, we don't show empty placeholders. We show nothing.

  const { role_info, audit_report } = apiResult;

  return (
    <section className="py-12 border-t border-zinc-900 animate-fade-in">
      
      {/* Refined Persona Box */}
      <div className="mb-16">
        <p className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase mb-4">Targeted Role</p>
        <h3 className="text-3xl font-light text-white mb-2">{role_info?.title || "Unknown Role"}</h3>
        <p className="text-zinc-500 font-light italic">"{role_info?.focus || "No focus detected"}"</p>
      </div>

      {/* Deep Dive Output */}
      <div>
        <p className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase mb-8">Overview</p>
        
        <div className="text-zinc-400 font-light max-w-4xl">
          <ReactMarkdown
            components={{
              h3: ({node, ...props}) => <h3 className="text-sm font-mono text-white mt-12 mb-6 tracking-widest uppercase border-b border-zinc-900 pb-2" {...props} />,
              strong: ({node, ...props}) => <strong className="font-medium text-zinc-200" {...props} />,
              p: ({node, ...props}) => <p className="mb-4 leading-relaxed" {...props} />,
              ul: ({node, ...props}) => <ul className="list-none space-y-4 mb-8" {...props} />,
              li: ({node, ...props}) => (
                <li className="flex gap-4" {...props}>
                  <span className="text-zinc-800 mt-1">▹</span>
                  <span>{props.children}</span>
                </li>
              )
            }}
          >
            {audit_report}
          </ReactMarkdown>
        </div>
      </div>

    </section>
  );
}

export default AuditReport;