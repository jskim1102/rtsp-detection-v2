import { useCallback, useEffect, useState } from "react";

const DEFAULT_ENGINE = "yolo26";
const ENGINE_IDS = new Set([DEFAULT_ENGINE, "rtdetr", "rfdetr", "yolo_world"]);

function engineFromHash(): string {
  const candidate = window.location.hash.slice(1);
  return ENGINE_IDS.has(candidate) ? candidate : DEFAULT_ENGINE;
}

export function useEngineTab() {
  const [engine, setCurrentEngine] = useState(engineFromHash);

  useEffect(() => {
    const handleHashChange = () => setCurrentEngine(engineFromHash());
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  const setEngine = useCallback((id: string) => {
    if (!ENGINE_IDS.has(id)) return;
    if (window.location.hash === `#${id}`) {
      setCurrentEngine(id);
      return;
    }
    window.location.hash = id;
  }, []);

  return { engine, setEngine };
}
