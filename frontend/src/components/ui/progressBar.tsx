// frontend/src/components/ui/ProgressBar.tsx
'use client';

interface ProgressBarProps {
    score: number; // A value between 0 and 1
    label: string;
}

export default function ProgressBar({ score, label }: ProgressBarProps) {
    const percent = Math.round(score * 100);
    let bgColor = 'bg-red-500';
    if (percent >= 70) {
        bgColor = 'bg-green-500';
    } else if (percent >= 40) {
        bgColor = 'bg-yellow-500';
    }

    return (
        <div>
            <div className="flex justify-between mb-1">
                <span className="text-sm font-medium text-gray-700">{label}</span>
                <span className="text-sm font-medium text-gray-700">{percent}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div 
                    className={`${bgColor} h-2.5 rounded-full transition-all duration-500`} 
                    style={{ width: `${percent}%` }}
                ></div>
            </div>
        </div>
    );
}