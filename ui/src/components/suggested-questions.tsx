"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const DEFAULT_QUESTIONS = [
  "与最高的建筑相比，埃菲尔铁塔有多高？",
  "GitHub上最热门的存储库有哪些？",
  "如果看待中国的外卖大战？",
  "超加工食品与健康有关吗？超加工食品的历史怎样？",
];

interface SuggestedQuestionsProps {
  className?: string;
  questions?: string[];
  onSelect?: (question: string) => void;
}

export function SuggestedQuestions({
  className,
  questions = DEFAULT_QUESTIONS,
  onSelect,
}: SuggestedQuestionsProps) {
  return (
    <div className={cn("flex flex-wrap gap-2", className)}>
      {questions.map((question) => (
        <Button
          key={question}
          variant="outline"
          className="cursor-pointer rounded-xl border-border bg-card text-muted-foreground transition-all hover:border-border-strong hover:-translate-y-0.5 hover:shadow-[var(--shadow-card)]"
          onClick={() => onSelect?.(question)}
        >
          {question}
        </Button>
      ))}
    </div>
  );
}
