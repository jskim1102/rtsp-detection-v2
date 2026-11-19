export interface EngineSpec {
  id: string;
  label: string;
  presets: string[];
  default: string;
  closed_set_coco: boolean;
  prompt: boolean;
}

interface Props {
  engines: EngineSpec[];
  active: string;
  onChange: (id: string) => void;
}

export default function EngineTabs({ engines, active, onChange }: Props) {
  return (
    <nav className="engine-tabs-band" aria-label="추론 엔진 선택">
      <div className="engine-tabs" role="tablist">
        {engines.map((engine) => {
          const selected = engine.id === active;
          return (
            <button
              key={engine.id}
              type="button"
              role="tab"
              aria-selected={selected}
              className={`engine-tabs-item${selected ? " engine-tabs-item-active" : ""}`}
              onClick={() => onChange(engine.id)}
            >
              {engine.label}
            </button>
          );
        })}
      </div>
    </nav>
  );
}
