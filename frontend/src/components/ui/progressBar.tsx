interface ProgressBarProps {
    score: number; //0 to 1 range
    label: string;
}

export default function ProgressBar({ score, label }: ProgressBarProps) {
    const percent = Math.round(score * 100);
    
    //color logic based on score
    let bgColor = 'bg-red-500';
    if (percent >= 70) {
        bgColor = 'bg-green-500'; //good score
    } else if (percent >= 40) {
        bgColor = 'bg-yellow-500'; //medium score
    }
    //below 40 stays red - poor score
    
    return (
        <div>
            {/* label and percentage display */}
            <div className="flex justify-between mb-1">
                <span className="text-sm font-medium text-gray-700">{label}</span>
                <span className="text-sm font-medium text-gray-700">{percent}%</span>
            </div>
            
            {/* progress bar track */}
            <div className="w-full bg-gray-200 rounded-full h-2.5">
                {/* filled portion with color */}
                <div
                    className={`${bgColor} h-2.5 rounded-full transition-all duration-500`}
                    style={{ width: `${percent}%` }}
                ></div>
            </div>
        </div>
    );
}