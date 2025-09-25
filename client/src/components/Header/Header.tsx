import Image from "next/image";
import { MdSettings, MdPerson, MdMenu } from "react-icons/md";
import { useState } from "react";

interface HeaderProps {
  onSummon: (prompt: string) => void;
  isCustomizeOpen: boolean;
  onToggleCustomize: () => void;
  isGenerating?: boolean;
}

export default function Header({ 
  onSummon, 
  isCustomizeOpen, 
  onToggleCustomize,
  isGenerating = false
}: HeaderProps) {
  const [prompt, setPrompt] = useState("");

  const handleSubmit = () => {
    if (prompt.trim()) {
      onSummon(prompt.trim());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };
  return (
    <header className="p-4 border-b border-[var(--secondary-bg)] flex-shrink-0 bg-[var(--primary-bg)]">
      <div className="w-full flex items-center justify-between gap-4">
        {/* Logo */}
        <div className="flex-shrink-0">
          <Image
            src="/logo.svg"
            alt="Artmancer"
            width={180}
            height={48}
            className="h-12 w-auto"
            style={{ height: '48px', width: 'auto' }}
            priority
          />
        </div>

        {/* Input Field and Edit Button */}
        <div className="flex-1 max-w-2xl mx-4 flex gap-3 items-center">
          <input
            type="text"
            placeholder="Describe the edits you want to make..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isGenerating}
            className="flex-1 px-4 py-3 bg-transparent border-2 border-[var(--primary-accent)] rounded-lg text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:border-[var(--highlight-accent)] focus:outline-none transition-colors text-sm h-12 disabled:opacity-50"
          />
          <button
            onClick={handleSubmit}
            disabled={isGenerating || !prompt.trim()}
            className="px-6 py-3 bg-[var(--primary-accent)] hover:bg-[var(--highlight-accent)] text-white font-semibold rounded-lg transition-colors text-sm flex-shrink-0 h-12 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isGenerating ? 'Generating...' : 'Edit!'}
          </button>
        </div>

        {/* Header Actions */}
        <div className="flex items-center gap-4 flex-shrink-0">
          <button
            className="p-3 rounded-lg bg-[var(--secondary-bg)] hover:bg-[var(--primary-accent)] text-[var(--text-secondary)] hover:text-white transition-all duration-200 h-12 w-12 flex items-center justify-center"
            aria-label="Settings"
          >
            <MdSettings size={20} />
          </button>
          <button
            className="p-3 rounded-lg bg-[var(--secondary-bg)] hover:bg-[var(--primary-accent)] text-[var(--text-secondary)] hover:text-white transition-all duration-200 h-12 w-12 flex items-center justify-center"
            aria-label="Profile"
          >
            <MdPerson size={20} />
          </button>
          <button
            onClick={onToggleCustomize}
            className={`p-3 rounded-lg transition-all duration-200 h-12 w-12 flex items-center justify-center ${
              isCustomizeOpen 
                ? 'bg-[var(--primary-accent)] text-white' 
                : 'bg-[var(--secondary-bg)] hover:bg-[var(--primary-accent)] text-[var(--text-secondary)] hover:text-white'
            }`}
            aria-label="Toggle Sidebar"
          >
            <MdMenu size={20} />
          </button>
        </div>
      </div>
    </header>
  );
}
