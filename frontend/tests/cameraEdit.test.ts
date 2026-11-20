import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";


// .tsx 는 node strip-types 로 렌더할 수 없어 소스를 직접 본다.
const source = (path: string) => readFile(new URL(path, import.meta.url), "utf8");


test("비밀번호만 바꾼 수정도 그 셀을 다시 붙인다 — 저장 성공 + 주소 변경 때 epoch 를 올려 셀 key 에 넣는다", async () => {
  // 응답 rtsp_url 은 비밀번호가 *** 로 가려져 와서 key 재료가 못 된다 — 부모 rtsp-streaming 과 같은 epoch 연결.
  const page = await source("../src/pages/CamerasPage.tsx");
  const grid = await source("../src/components/CameraGrid.tsx");
  assert.match(grid, /key=\{`\$\{cam\.id\}-\$\{epochs\?\.\[cam\.id\] \?\? 0\}`\}/);
  assert.match(page, /epochs=\{playerEpoch\}/);
  assert.match(page, /const urlChanged = rtspUrl !== editCam\.rtsp_url;/);
  const saved = page.indexOf('if (!resp.ok) return "카메라 수정에 실패했습니다.";');
  const bump = page.indexOf("if (urlChanged) {");
  assert.ok(saved !== -1 && bump > saved, "epoch 는 저장이 성공한 뒤에만 올린다");
});


test("수정 창 *** 안내는 주소를 바꿀 때도 실제 비밀번호를 다시 넣으라고 한다", async () => {
  // backend _guard_masked_credential_exfil 은 *** 인 채 주소를 바꾸면 400 으로 거부한다.
  const modal = await source("../src/components/CameraFormModal.tsx");
  assert.match(modal, /이름만 수정할 때만 유효합니다/);
  assert.match(modal, /주소·포트·경로 또는 비밀번호를 변경하려면/);
  assert.doesNotMatch(modal, /그대로 수정하면 반영됩니다/);
});
