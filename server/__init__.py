"""在线视频生成字幕系统 — 服务端包。

模块划分见 docs/在线视频生成字幕系统-详细设计.md §3.1：
  config    可配置项（环境变量前缀 VD_）
  store     任务目录 / status.json 原子读写 / task_id / 启动恢复 / 删除墓碑
  probe     ffprobe 上传校验
  errors    ApiError 错误码目录
  scheduler 内存队列 + dispatcher 线程 + GPU Worker 生命周期
  worker    GPU Worker 子进程入口（常驻模型 + 任务循环）
  main      FastAPI 应用与路由
  pipeline  预处理 / ASR / 排版 / 序列化
"""
