# 自托管 ffmpeg.wasm（@ffmpeg 0.12.x，单线程）

浏览器内无损提取音轨用。全部自托管，不依赖 CDN。

## 文件

| 文件 | 来源 |
|---|---|
| `ffmpeg.js` | `@ffmpeg/ffmpeg@0.12.15` UMD（**已打补丁**，见下） |
| `ffmpeg.js.orig` | 补丁前原始文件（留存对比用） |
| `814.ffmpeg.js` | `@ffmpeg/ffmpeg@0.12.15` 的 worker（原样） |
| `ffmpeg-core.js` / `ffmpeg-core.wasm` | `@ffmpeg/core@0.12.10` **UMD** 构建 |

## 关键补丁（ffmpeg.js）

把 `classWorkerURL` 分支构造 worker 的 `{type:"module"}` 改为 `{}`（经典 worker）：

```
new Worker(new URL(s, <hardcoded base>), {type:"module"})   →   {}
```

**原因**：worker（`814.ffmpeg.js`）加载 core 的逻辑是先 `importScripts(coreURL)`，仅当
`importScripts` 不可用时才回退到 `r(454)(coreURL)`——而 `r(454)` 是 webpack 编译出的
死桩，**永远抛 `Cannot find module`**。模块 worker 里没有 `importScripts`，必然掉进死桩。
因此 worker 必须是**经典 worker**，`importScripts` 才可用，配合 **UMD** core 即可正常加载。

自托管 UMD 时又必须显式传 `classWorkerURL`（否则默认 worker 路径按构建机的 `file://`
基址解析而失败），但传了 `classWorkerURL` 官方就硬编码成模块 worker——这个组合在 UMD
自托管场景是坏的，故打此补丁。三个 URL（classWorkerURL/coreURL/wasmURL）均须传绝对
http 地址（见 `web/app.js`）。

## 重新获取时

若重新下载 `ffmpeg.js` 覆盖，请重新应用上述一处 `{type:"module"}`→`{}` 补丁。
