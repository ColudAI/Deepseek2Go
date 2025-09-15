# DeepSeek Go Proxy (Reloaded-Version)

> 本产品由 ColudAi & AmethystCraft-DevTeam 联合开发

目标
- 用 Go 重写 Python 版代理，显著降低常驻内存与依赖体积
- 保持与现有 API 行为基本一致：OpenAI 风格 `/v1/chat/completions` 与 Claude 风格 `/anthropic/v1/messages`
- 集成 WASM PoW 计算（使用 Wazero），支持 DeepSeek 官方接口调用

当前进度
- 路由与服务骨架：已完成
- 账号池与配置读取：已完成
- DeepSeek 客户端（登录、创建会话、获取 PoW、调用 completion）：已实现
- WASM PoW 计算（Wazero）：已实现
- 消息预处理 messages_prepare：已实现
- OpenAI 路径 SSE 转换：已实现
- Claude 路径：已实现非流式，流式待补充
- TLS 指纹模拟（CycleTLS）：暂未接入（目前使用标准 http 客户端）

核心代码入口
- HTTP 服务启动与路由：[go.main()](cmd/server/main.go:1149)、[go.setupRouter()](cmd/server/main.go:1136)
- 配置加载：[go.loadConfig()](cmd/server/main.go:107)
- 账号池实现：[go.NewAccountManager()](cmd/server/main.go:60)、[go.AccountManager.ChooseNew()](cmd/server/main.go:69)、[go.AccountManager.Release()](cmd/server/main.go:83)
- 模式判定（配置 key 模式 vs 用户自带 Token）：[go.determineModeAndToken()](cmd/server/main.go:219)
- DeepSeek 登录：[go.loginDeepseekViaAccount()](cmd/server/main.go:274)
- 创建会话：[go.createSession()](cmd/server/main.go:327)
- PoW 挑战与计算：结构 [go.powChallenge](cmd/server/main.go:359)、WASM 初始化 [go.initPow()](cmd/server/main.go:395)、计算 [go.computePowAnswer()](cmd/server/main.go:441)、打包 header 值 [go.getPowResponseB64()](cmd/server/main.go:528)
- 调用完成接口：[go.callCompletionEndpoint()](cmd/server/main.go:571)
- 消息预处理与模板化：[go.messagesPrepare()](cmd/server/main.go:594)
- 模型映射（Claude → DeepSeek）：[go.mapClaudeModelToDeepseek()](cmd/server/main.go:665)
- OpenAI 路径处理（含 SSE）：[go.handleChatCompletions()](cmd/server/main.go:727)
- Claude 路径处理（非流式）：[go.handleClaudeMessages()](cmd/server/main.go:1102)

目录结构
- cmd/server/main.go：服务入口与全部实现（首版合并在一个文件，后续可内聚为 internal 包）
- config.json：运行时配置（keys、accounts、wasm 路径、模型映射）
- ../sha3_wasm_bg.7b9ca65ddd.wasm：WASM 文件（在仓库根目录）

构建与运行
1) 安装依赖（已在 go.mod 中声明）
   go env -w GOPROXY=https://goproxy.cn,direct
   go build ./cmd/server

2) 编辑配置 Reloaded-Version/config.json
   {
     "keys": ["YOUR_GATEWAY_KEY"],     // 自定义网关鉴权 key，可选
     "accounts": [
       {
         "email": "your_email@example.com",
         "password": "your_password",
         "token": ""                    // 首次留空，服务会尝试登录刷新
       }
     ],
     "claude_model_mapping": {
       "fast": "deepseek-chat",
       "slow": "deepseek-chat"
     },
     "wasm_path": "../sha3_wasm_bg.7b9ca65ddd.wasm",
     "template_dir": "templates"
   }

3) 启动
   # 端口默认 5001，如被占用可改用其他端口
   PORT=5051 ./server

4) 健康检查
   curl -s http://127.0.0.1:5051/

5) 列出模型
   curl -s http://127.0.0.1:5051/v1/models

调用示例
- OpenAI 风格（非流式）
  curl -sS -X POST "http://127.0.0.1:5051/v1/chat/completions" \
    -H "Authorization: Bearer &lt;DeepSeek Token 或 config.json keys 中的某个值&gt;" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "deepseek-chat",
      "messages": [
        {"role":"user","content":"Hello from Go proxy!"}
      ],
      "stream": false
    }'

- OpenAI 风格（流式）
  curl -N -sS -X POST "http://127.0.0.1:5051/v1/chat/completions" \
    -H "Authorization: Bearer &lt;DeepSeek Token 或 config.json keys 中的某个值&gt;" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "deepseek-chat",
      "messages": [
        {"role":"user","content":"Stream please"}
      ],
      "stream": true
    }'

- Claude 风格（非流式）
  curl -sS -X POST "http://127.0.0.1:5051/anthropic/v1/messages" \
    -H "Authorization: Bearer &lt;DeepSeek Token 或 config.json keys 中的某个值&gt;" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "claude-sonnet-4-20250514",
      "messages": [
        {"role":"user","content":"你好"}
      ],
      "stream": false
    }'

注意事项
- 账号登录与 TLS 指纹：Python 版使用 curl_cffi impersonate safari15_3；Go 首版使用标准 http 客户端。某些网络环境下 DeepSeek 登录可能较严格。如果遇到 4xx/5xx，可优先使用自带 DeepSeek Token 调用，或后续接入 CycleTLS 提升兼容性。
- WASM 路径：若出现 missing wasm exports / wasm_solve 不存在 等错误，请确认 config.json 中 wasm_path 指向的文件存在。
- 资源释放：配置模式下调用结束会自动将账号归还队列。[go.handleChatCompletions()](cmd/server/main.go:727) 和 [go.handleClaudeMessages()](cmd/server/main.go:1102) 已在 defer 中处理。
- Token 模式：如果 Authorization Bearer 未命中 config.json keys，将被视为直接使用 DeepSeek Token 调用。

后续规划
- 接入 CycleTLS，增强登录/调用的稳定性（减少被风控风险）
- 补齐 Claude 路径的流式 SSE 兼容
- 拆分 internal 包（config、deepseek、pow、sse、tokenizer），提升可维护性
- 增强错误处理与重试策略、完善日志与指标