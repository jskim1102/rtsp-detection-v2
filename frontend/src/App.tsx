import { useEffect, useRef, useState } from "react";
import CamerasPage from "./pages/CamerasPage";
import EngineTabs, { type EngineSpec } from "./components/EngineTabs";
import { useEngineTab } from "./hooks/useEngineTab";
import { apiBase } from "./hooks/useApi";

export default function App() {
  const { engine, setEngine } = useEngineTab();
  const [engines, setEngines] = useState<EngineSpec[]>([]);
  const [error, setError] = useState("");
  const requestedEnginesRef = useRef(false);
  const syncedEngineRef = useRef<string | null>(null);

  useEffect(() => {
    if (requestedEnginesRef.current) return;
    requestedEnginesRef.current = true;
    fetch(`${apiBase()}/api/inference/engines`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return (await response.json()) as EngineSpec[];
      })
      .then(setEngines)
      .catch(() => setError("추론 엔진 목록을 불러오지 못했습니다."));
  }, []);

  const spec = engines.find((candidate) => candidate.id === engine);

  useEffect(() => {
    if (!spec || syncedEngineRef.current === engine) return;
    syncedEngineRef.current = engine;
    fetch(`${apiBase()}/api/inference/config`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ engine }),
    })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        setError("");
      })
      .catch(() => setError("엔진 전환을 저장하지 못했습니다."));
  }, [engine, spec]);

  return (
    <>
      <EngineTabs engines={engines} active={engine} onChange={setEngine} />
      {error && <p className="app app-error">{error}</p>}
      {spec && <CamerasPage key={engine} engine={engine} spec={spec} />}
    </>
  );
}
