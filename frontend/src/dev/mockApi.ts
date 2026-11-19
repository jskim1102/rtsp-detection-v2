import type { EngineSpec } from "../components/EngineTabs";

const MOCK_QUERY_KEY = "mock";
const originalFetch = window.fetch.bind(window);

const engines: EngineSpec[] = [
  {
    id: "yolo26",
    label: "YOLO26",
    presets: ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"],
    default: "yolo26n.pt",
    closed_set_coco: true,
    prompt: false,
  },
  {
    id: "rtdetr",
    label: "RT-DETR",
    presets: ["rtdetr-l.pt", "rtdetr-x.pt"],
    default: "rtdetr-l.pt",
    closed_set_coco: true,
    prompt: false,
  },
  {
    id: "rfdetr",
    label: "RF-DETR",
    presets: ["rfdetr-base"],
    default: "rfdetr-base",
    closed_set_coco: true,
    prompt: false,
  },
  {
    id: "yolo_world",
    label: "YOLO-World",
    presets: ["yolov8s-worldv2.pt", "yolov8m-worldv2.pt", "yolov8l-worldv2.pt", "yolov8x-worldv2.pt"],
    default: "yolov8s-worldv2.pt",
    closed_set_coco: false,
    prompt: true,
  },
];

const cameras = [
  {
    id: 1,
    name: "정문 카메라",
    rtsp_url: "rtsp://demo.local/front-door",
    stream_key: "rtsp_detection_v2__ipcam-demo-1",
    created_at: "2026-08-18T00:00:00Z",
  },
  {
    id: 2,
    name: "주차장 카메라",
    rtsp_url: "rtsp://demo.local/parking",
    stream_key: "rtsp_detection_v2__ipcam-demo-2",
    created_at: "2026-08-18T00:00:00Z",
  },
];

const config = {
  enabled: true,
  model: engines[0].default,
  conf_threshold: 0.5,
  device: "cuda:0",
  gpu_util_target: 0.85,
  gpu_util_duty: 0.42,
  engine: engines[0].id,
  prompt: "",
};

const sourceInference = new Map(
  cameras.map((camera) => [
    camera.stream_key,
    { enabled: true, conf_threshold: 0.5, models: [engines[0].default] },
  ])
);

let installed = false;

export function isMockApiEnabled(): boolean {
  return import.meta.env.DEV && new URLSearchParams(window.location.search).has(MOCK_QUERY_KEY);
}

function engineSpec(id: string): EngineSpec {
  return engines.find((candidate) => candidate.id === id) ?? engines[0];
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function requestUrl(input: RequestInfo | URL): URL {
  if (input instanceof Request) return new URL(input.url);
  return new URL(input.toString(), window.location.href);
}

function requestMethod(input: RequestInfo | URL, init?: RequestInit): string {
  if (init?.method) return init.method.toUpperCase();
  return input instanceof Request ? input.method.toUpperCase() : "GET";
}

function requestBody(init?: RequestInit): Record<string, unknown> {
  if (typeof init?.body !== "string") return {};
  try {
    return JSON.parse(init.body) as Record<string, unknown>;
  } catch {
    return {};
  }
}

async function mockFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const url = requestUrl(input);
  const method = requestMethod(input, init);
  const body = requestBody(init);
  const path = url.pathname;

  if (path === "/api/inference/engines" && method === "GET") return json(engines);

  if (path === "/api/inference/config") {
    if (method === "PUT") {
      if (typeof body.engine === "string") {
        const nextSpec = engineSpec(body.engine);
        config.engine = nextSpec.id;
        config.model = nextSpec.default;
        for (const state of sourceInference.values()) state.models = [nextSpec.default];
      }
      if (typeof body.prompt === "string") config.prompt = body.prompt;
      if (typeof body.gpu_util_target === "number") {
        config.gpu_util_target = body.gpu_util_target;
        config.gpu_util_duty = Math.min(body.gpu_util_target, 0.42);
      }
    }
    return json(config);
  }

  if (path === "/api/config" && method === "GET") return json({ max_ipcams: 16 });

  if (path === "/api/inference/models" && method === "GET") {
    const spec = engineSpec(url.searchParams.get("engine") ?? config.engine);
    return json(spec.presets.map((name) => ({ name, type: "preset", size_mb: null })));
  }

  if (/^\/api\/inference\/models\/[^/]+\/classes$/.test(path) && method === "GET") {
    return json([
      { id: 0, name: "person" },
      { id: 1, name: "bicycle" },
      { id: 2, name: "car" },
      { id: 7, name: "truck" },
    ]);
  }

  if (path === "/api/ipcams") {
    if (method === "GET") return json(cameras);
    if (method === "POST") return json(cameras[0], 201);
  }

  const statsMatch = path.match(/^\/api\/ipcams\/([^/]+)\/stats$/);
  if (statsMatch && method === "GET") return json({ active: true, readers: 1 });

  const inferenceMatch = path.match(/^\/api\/ipcams\/([^/]+)\/inference$/);
  if (inferenceMatch) {
    const streamKey = decodeURIComponent(inferenceMatch[1]);
    const state = sourceInference.get(streamKey) ?? {
      enabled: true,
      conf_threshold: 0.5,
      models: [engineSpec(config.engine).default],
    };
    if (method === "PUT") {
      if (typeof body.enabled === "boolean") state.enabled = body.enabled;
      if (typeof body.conf_threshold === "number") state.conf_threshold = body.conf_threshold;
      if (Array.isArray(body.models)) state.models = body.models.filter((name): name is string => typeof name === "string");
      sourceInference.set(streamKey, state);
    }
    return json(state);
  }

  if (/^\/api\/ipcams\/\d+$/.test(path) && (method === "PUT" || method === "DELETE")) {
    return method === "DELETE" ? new Response(null, { status: 204 }) : json(cameras[0]);
  }

  return originalFetch(input, init);
}

class MockWebSocket {
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  private timer: number | null;

  constructor(_url: string | URL) {
    this.timer = window.setInterval(() => {
      const promptNames = config.prompt
        .split(",")
        .map((name) => name.trim())
        .filter(Boolean);
      const names = promptNames.length > 0 ? promptNames : ["person", "car"];
      const model = engineSpec(config.engine).default;
      this.onmessage?.(
        new MessageEvent("message", {
          data: JSON.stringify({
            type: "detections",
            timestamp: Date.now() / 1000,
            frame: { w: 1280, h: 720 },
            items: [
              { class_id: 0, name: names[0], conf: 0.94, xyxy: [120, 90, 460, 650], model },
              { class_id: 2, name: names[1] ?? names[0], conf: 0.86, xyxy: [690, 310, 1140, 620], model },
            ],
          }),
        })
      );
    }, 1000);
  }

  close() {
    if (this.timer !== null) window.clearInterval(this.timer);
    this.timer = null;
    this.onclose?.();
  }
}

export function installMockApi(): void {
  if (!isMockApiEnabled() || installed) return;
  installed = true;
  window.fetch = mockFetch;
  window.WebSocket = MockWebSocket as unknown as typeof WebSocket;
}
