import { useCallback, useEffect, useRef } from "react";
import { apiBase } from "../hooks/useApi";

interface Props {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

export default function PromptBar({ value, onChange, disabled = false }: Props) {
  const lastSubmittedValueRef = useRef(value);
  const timerRef = useRef<number | null>(null);

  const submit = useCallback(async (prompt: string) => {
    const response = await fetch(`${apiBase()}/api/inference/config`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  }, []);

  useEffect(() => {
    if (value === lastSubmittedValueRef.current) return;
    timerRef.current = window.setTimeout(() => {
      timerRef.current = null;
      lastSubmittedValueRef.current = value;
      void submit(value).catch(() => {});
    }, 400);
    return () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    };
  }, [submit, value]);

  const applyNow = () => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    lastSubmittedValueRef.current = value;
    void submit(value).catch(() => {});
  };

  return (
    <section className="prompt-bar" aria-label="탐지 클래스 프롬프트">
      <div className="prompt-bar-copy">
        <label htmlFor="engine-prompt">탐지 클래스</label>
        <p>콤마로 구분하세요. 비우면 모델 기본 어휘를 사용합니다.</p>
      </div>
      <div className="prompt-bar-control">
        <input
          id="engine-prompt"
          type="text"
          value={value}
          disabled={disabled}
          placeholder="person, car, bicycle"
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") applyNow();
          }}
        />
        <button type="button" disabled={disabled} onClick={applyNow}>
          적용
        </button>
      </div>
    </section>
  );
}
