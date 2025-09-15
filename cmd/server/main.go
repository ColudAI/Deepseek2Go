package main

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
)

/*
Go 重构版本骨架
- 目标：降低常驻内存、保持与 Python 版 API 兼容
- 当前进度：项目骨架、配置加载、CORS、中间件、基础路由
- 待办：DeepSeek 登录/会话/PoW/Completion、SSE 转换、Tokenizer 复刻
*/

// ========================== 配置与模型 ==========================

type Account struct {
	Email    string `json:"email"`
	Mobile   string `json:"mobile"`
	AreaCode string `json:"area_code,omitempty"`
	Password string `json:"password"`
	Token    string `json:"token"`
}

type AppConfig struct {
	Keys               []string          `json:"keys"`
	Accounts           []Account         `json:"accounts"`
	ClaudeModelMapping map[string]string `json:"claude_model_mapping"`
	WasmPath           string            `json:"wasm_path"`
	TemplateDir        string            `json:"template_dir"`
}

var (
	appConfig   AppConfig
	configMutex sync.RWMutex
)

// ========================== 账号池（并发安全队列） ==========================

type AccountManager struct {
	mu    sync.Mutex
	queue []*Account
}

func NewAccountManager(accounts []Account) *AccountManager {
	q := make([]*Account, 0, len(accounts))
	for i := range accounts {
		acc := accounts[i] // 拷贝
		q = append(q, &acc)
	}
	return &AccountManager{queue: q}
}

func (m *AccountManager) ChooseNew(exclude map[string]bool) *Account {
	m.mu.Lock()
	defer m.mu.Unlock()
	for i := 0; i < len(m.queue); i++ {
		acc := m.queue[i]
		id := getAccountIdentifier(acc)
		if id != "" && !exclude[id] {
			// 从队列移除
			m.queue = append(m.queue[:i], m.queue[i+1:]...)
			return acc
		}
	}
	return nil
}

func (m *AccountManager) Release(acc *Account) {
	if acc == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.queue = append(m.queue, acc)
}

func getAccountIdentifier(a *Account) string {
	if a == nil {
		return ""
	}
	if s := strings.TrimSpace(a.Email); s != "" {
		return s
	}
	return strings.TrimSpace(a.Mobile)
}

var accountMgr *AccountManager

// ========================== 配置加载/保存 ==========================

func loadConfig(path string) (AppConfig, error) {
	var cfg AppConfig
	f, err := os.Open(path)
	if err != nil {
		// 默认空配置
		cfg = AppConfig{
			Keys:        []string{},
			Accounts:    []Account{},
			WasmPath:    "sha3_wasm_bg.7b9ca65ddd.wasm",
			TemplateDir: "templates",
			ClaudeModelMapping: map[string]string{
				"fast": "deepseek-chat",
				"slow": "deepseek-chat",
			},
		}
		return cfg, nil
	}
	defer f.Close()
	if err := json.NewDecoder(f).Decode(&cfg); err != nil {
		return AppConfig{}, err
	}
	// 兜底
	if cfg.WasmPath == "" {
		cfg.WasmPath = "sha3_wasm_bg.7b9ca65ddd.wasm"
	}
	if cfg.TemplateDir == "" {
		cfg.TemplateDir = "templates"
	}
	if cfg.ClaudeModelMapping == nil {
		cfg.ClaudeModelMapping = map[string]string{
			"fast": "deepseek-chat",
			"slow": "deepseek-chat",
		}
	}
	return cfg, nil
}

func saveConfig(path string, cfg AppConfig) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(cfg); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	_ = f.Close()
	return os.Rename(tmp, path)
}

// ========================== 常量与HTTP工具 ==========================

const (
	deepseekHost             = "chat.deepseek.com"
	deepseekLoginURL         = "https://chat.deepseek.com/api/v0/users/login"
	deepseekCreateSessionURL = "https://chat.deepseek.com/api/v0/chat_session/create"
	deepseekCreatePowURL     = "https://chat.deepseek.com/api/v0/chat/create_pow_challenge"
	deepseekCompletionURL    = "https://chat.deepseek.com/api/v0/chat/completion"
)

var baseHeaders = map[string]string{
	"Host":              deepseekHost,
	"User-Agent":        "DeepSeek/1.0.13 Android/35",
	"Accept":            "application/json",
	"Accept-Encoding":   "gzip",
	"Content-Type":      "application/json",
	"x-client-platform": "android",
	"x-client-version":  "1.3.0-auto-resume",
	"x-client-locale":   "zh_CN",
	"accept-charset":    "UTF-8",
}

var httpClient = &http.Client{
	Timeout: 60 * time.Second,
}

func httpPostJSON(url string, headers map[string]string, body any) (*http.Response, error) {
	bs, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest(http.MethodPost, url, strings.NewReader(string(bs)))
	if err != nil {
		return nil, err
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	return httpClient.Do(req)
}

func getAuthHeaders(token string) map[string]string {
	h := make(map[string]string, len(baseHeaders)+2)
	for k, v := range baseHeaders {
		h[k] = v
	}
	// HTTP 头大小写不敏感，但为避免某些风控实现的大小写检查，双写一份
	h["authorization"] = "Bearer " + token
	h["Authorization"] = "Bearer " + token
	return h
}

// ========================== 模式判定与登录（占位） ==========================

type ModeContext struct {
	UseConfigToken bool
	DeepseekToken  string
	Account        *Account
}

func determineModeAndToken(authHeader string) (ModeContext, error) {
	if !strings.HasPrefix(authHeader, "Bearer ") {
		return ModeContext{}, errors.New("Unauthorized: missing Bearer token")
	}
	callerKey := strings.TrimSpace(strings.TrimPrefix(authHeader, "Bearer "))

	configMutex.RLock()
	defer configMutex.RUnlock()

	// 配置模式：callerKey 命中 keys 列表
	for _, k := range appConfig.Keys {
		if callerKey == k {
			exclude := map[string]bool{}
			acc := accountMgr.ChooseNew(exclude)
			log.Printf("[auth] config-mode: picked account=%s token_set=%t", getAccountIdentifier(acc), strings.TrimSpace(acc.Token) != "")
			if acc == nil {
				return ModeContext{}, errors.New("No accounts configured or all accounts are busy")
			}
			// 如账号无 token，需要登录（当前未实现，后续补齐）
			if strings.TrimSpace(acc.Token) == "" {
				_, err := loginDeepseekViaAccount(acc)
				if err != nil {
					// 释放并返回错误
					accountMgr.Release(acc)
					return ModeContext{}, fmt.Errorf("Account login failed: %v", err)
				}
				// 持久化写回
				for i := range appConfig.Accounts {
					if getAccountIdentifier(&appConfig.Accounts[i]) == getAccountIdentifier(acc) {
						appConfig.Accounts[i].Token = acc.Token
						_ = saveConfig("config.json", appConfig)
						log.Printf("[auth] persisted token for account=%s", getAccountIdentifier(acc))
						break
					}
				}
			}
			return ModeContext{
				UseConfigToken: true,
				DeepseekToken:  acc.Token,
				Account:        acc,
			}, nil
		}
	}

	// 用户自带 Token 模式
	log.Printf("[auth] direct-token mode; bearer_len=%d", len(callerKey))
	return ModeContext{
		UseConfigToken: false,
		DeepseekToken:  callerKey,
		Account:        nil,
	}, nil
}

/*
loginDeepseekViaAccount
- 使用邮箱或手机号+密码登录，返回 token 并写回到 acc.Token
- 兼容 Python 版的 payload 结构
*/
func loginDeepseekViaAccount(acc *Account) (string, error) {
	if acc == nil {
		return "", errors.New("nil account")
	}
	email := strings.TrimSpace(acc.Email)
	mobile := strings.TrimSpace(acc.Mobile)
	password := strings.TrimSpace(acc.Password)
	if password == "" || (email == "" && mobile == "") {
		return "", errors.New("account missing email/mobile or password")
	}

	payload := map[string]any{
		"device_id": "deepseek_to_api_go",
		"os":        "android",
	}
	if email != "" {
		payload["email"] = email
		payload["password"] = password
	} else {
		payload["mobile"] = mobile
		ac := strings.TrimSpace(acc.AreaCode)
		if ac != "" {
			payload["area_code"] = ac
		} else {
			payload["area_code"] = nil
		}
		payload["password"] = password
	}

	resp, err := httpPostJSON(deepseekLoginURL, baseHeaders, payload)
	if err != nil {
		return "", fmt.Errorf("login request failed: %w", err)
	}
	defer resp.Body.Close()
	status := resp.StatusCode

	var data struct {
		Data struct {
			BizData struct {
				User struct {
					Token string `json:"token"`
				} `json:"user"`
			} `json:"biz_data"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return "", fmt.Errorf("login decode failed: %w", err)
	}
	log.Printf("[login] status=%d token_present=%t", status, data.Data.BizData.User.Token != "")
	if data.Data.BizData.User.Token == "" {
		return "", errors.New("login missing token in response")
	}
	acc.Token = data.Data.BizData.User.Token
	return acc.Token, nil
}

/*
createSession
- 使用 deepseek token 创建会话，返回 session_id
*/
func createSession(token string) (string, error) {
	headers := getAuthHeaders(token)
	resp, err := httpPostJSON(deepseekCreateSessionURL, headers, map[string]any{"agent": "chat"})
	if err != nil {
		return "", fmt.Errorf("create_session request failed: %w", err)
	}
	defer resp.Body.Close()

	var data struct {
		Code int `json:"code"`
		Data struct {
			BizData struct {
				ID string `json:"id"`
			} `json:"biz_data"`
		} `json:"data"`
		Msg string `json:"msg"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return "", fmt.Errorf("create_session decode failed: %w", err)
	}
	if resp.StatusCode != http.StatusOK || data.Code != 0 {
		log.Printf("[create_session] status=%d code=%d msg=%s", resp.StatusCode, data.Code, data.Msg)
		return "", fmt.Errorf("create_session failed: code=%d msg=%s", data.Code, data.Msg)
	}
	log.Printf("[create_session] ok id=%s", data.Data.BizData.ID)
	return data.Data.BizData.ID, nil
}

/*
PoW 相关
- 先定义 challenge 结构与组装 x-ds-pow-response 的逻辑
- 计算 answer 的逻辑后续用 Wazero 调用 wasm_solve 实现
*/

type powChallenge struct {
	Algorithm  string `json:"algorithm"`
	Challenge  string `json:"challenge"`
	Salt       string `json:"salt"`
	Difficulty int    `json:"difficulty"`
	ExpireAt   int64  `json:"expire_at"`
	Signature  string `json:"signature"`
	TargetPath string `json:"target_path"`
}

/*
WASM PoW 引擎：使用 Wazero 加载 sha3_wasm_bg*.wasm，调用导出函数：
- memory
- __wbindgen_add_to_stack_pointer
- __wbindgen_export_0 (allocator)
- wasm_solve
流程与 Python 版本一致：分配栈 -16，编码 challenge 与 prefix，调用 wasm_solve，读取 retptr 处状态与结果，恢复栈 +16。
*/

type wasmPow struct {
	mu        sync.Mutex
	rt        wazero.Runtime
	mod       api.Module
	mem       api.Memory
	fAddStack api.Function
	fAlloc    api.Function
	fSolve    api.Function
	ctx       context.Context
}

var (
	powOnce sync.Once
	powInst *wasmPow
	powErr  error
)

func initPow(path string) (*wasmPow, error) {
	ctx := context.Background()
	rt := wazero.NewRuntime(ctx)

	wasmBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read wasm failed: %w", err)
	}

	mod, err := rt.Instantiate(ctx, wasmBytes)
	if err != nil {
		return nil, fmt.Errorf("instantiate wasm failed: %w", err)
	}
	mem := mod.Memory()
	if mem == nil {
		return nil, errors.New("wasm memory export missing")
	}

	fAdd := mod.ExportedFunction("__wbindgen_add_to_stack_pointer")
	fAlloc := mod.ExportedFunction("__wbindgen_export_0")
	fSolve := mod.ExportedFunction("wasm_solve")
	if fAdd == nil || fAlloc == nil || fSolve == nil {
		return nil, errors.New("missing wasm exports")
	}
	return &wasmPow{
		rt:        rt,
		mod:       mod,
		mem:       mem,
		fAddStack: fAdd,
		fAlloc:    fAlloc,
		fSolve:    fSolve,
		ctx:       ctx,
	}, nil
}

func ensurePow() (*wasmPow, error) {
	powOnce.Do(func() {
		configMutex.RLock()
		path := appConfig.WasmPath
		configMutex.RUnlock()
		powInst, powErr = initPow(path)
	})
	return powInst, powErr
}

// 计算 DeepSeekHashV1 的 PoW 答案
func computePowAnswer(alg, challenge, salt string, difficulty int, expireAt int64, signature, targetPath string) (int64, error) {
	if alg != "DeepSeekHashV1" {
		return 0, fmt.Errorf("unsupported algorithm: %s", alg)
	}
	w, err := ensurePow()
	if err != nil {
		return 0, err
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	prefix := fmt.Sprintf("%s_%d_", salt, expireAt)

	// 栈申请 -16，返回 retptr
	delta := int32(-16)
	res, err := w.fAddStack.Call(w.ctx, uint64(uint32(delta)))
	if err != nil || len(res) == 0 {
		return 0, fmt.Errorf("add_to_stack -16 failed: %v", err)
	}
	retptr := uint32(res[0])

	// 分配并写入字符串到 wasm 内存
	allocWrite := func(s string) (uint32, uint32, error) {
		b := []byte(s)
		ln := uint32(len(b))
		r, err := w.fAlloc.Call(w.ctx, uint64(ln), uint64(1))
		if err != nil || len(r) == 0 {
			return 0, 0, fmt.Errorf("alloc failed: %v", err)
		}
		ptr := uint32(r[0])
		if ok := w.mem.Write(ptr, b); !ok {
			return 0, 0, errors.New("memory write failed")
		}
		return ptr, ln, nil
	}

	ptrCh, lenCh, err := allocWrite(challenge)
	if err != nil {
		_, _ = w.fAddStack.Call(w.ctx, uint64(uint32(16)))
		return 0, err
	}
	ptrPrefix, lenPrefix, err := allocWrite(prefix)
	if err != nil {
		_, _ = w.fAddStack.Call(w.ctx, uint64(uint32(16)))
		return 0, err
	}

	// 调用 wasm_solve
	_, err = w.fSolve.Call(w.ctx,
		uint64(retptr),
		uint64(ptrCh),
		uint64(lenCh),
		uint64(ptrPrefix),
		uint64(lenPrefix),
		math.Float64bits(float64(difficulty)),
	)
	if err != nil {
		_, _ = w.fAddStack.Call(w.ctx, uint64(uint32(16)))
		return 0, fmt.Errorf("wasm_solve failed: %v", err)
	}

	// 从 retptr 读取状态(int32 LE) 和结果(f64 LE)
	stBytes, ok := w.mem.Read(retptr, 4)
	if !ok || len(stBytes) != 4 {
		_, _ = w.fAddStack.Call(w.ctx, uint64(uint32(16)))
		return 0, errors.New("read status failed")
	}
	status := int32(binary.LittleEndian.Uint32(stBytes))

	valBytes, ok := w.mem.Read(retptr+8, 8)
	if !ok || len(valBytes) != 8 {
		_, _ = w.fAddStack.Call(w.ctx, uint64(uint32(16)))
		return 0, errors.New("read value failed")
	}
	bits := binary.LittleEndian.Uint64(valBytes)
	value := math.Float64frombits(bits)

	// 恢复栈 +16
	_, _ = w.fAddStack.Call(w.ctx, uint64(uint32(16)))

	if status == 0 {
		return 0, nil
	}
	return int64(value), nil
}

func getPowResponseB64(token string) (string, error) {
	headers := getAuthHeaders(token)
	resp, err := httpPostJSON(deepseekCreatePowURL, headers, map[string]any{
		"target_path": "/api/v0/chat/completion",
	})
	if err != nil {
		return "", fmt.Errorf("get_pow_challenge request failed: %w", err)
	}
	defer resp.Body.Close()

	var data struct {
		Code int `json:"code"`
		Data struct {
			BizData struct {
				Challenge powChallenge `json:"challenge"`
			} `json:"biz_data"`
		} `json:"data"`
		Msg string `json:"msg"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return "", fmt.Errorf("get_pow_challenge decode failed: %w", err)
	}
	if resp.StatusCode != http.StatusOK || data.Code != 0 {
		log.Printf("[pow] status=%d code=%d msg=%s", resp.StatusCode, data.Code, data.Msg)
		return "", fmt.Errorf("get_pow_challenge failed: code=%d msg=%s", data.Code, data.Msg)
	}
	log.Printf("[pow] challenge ok diff=%d expire=%d", data.Data.BizData.Challenge.Difficulty, data.Data.BizData.Challenge.ExpireAt)

	ch := data.Data.BizData.Challenge
	answer, err := computePowAnswer(ch.Algorithm, ch.Challenge, ch.Salt, ch.Difficulty, ch.ExpireAt, ch.Signature, ch.TargetPath)
	if err != nil || answer == 0 {
		return "", fmt.Errorf("compute_pow_answer failed: %v", err)
	}
	powDict := map[string]any{
		"algorithm":   ch.Algorithm,
		"challenge":   ch.Challenge,
		"salt":        ch.Salt,
		"answer":      answer,
		"signature":   ch.Signature,
		"target_path": ch.TargetPath,
	}
	js, _ := json.Marshal(powDict)
	return base64.StdEncoding.EncodeToString(js), nil
}

// 调用对话接口，返回 *http.Response（可能是 SSE 流）
func callCompletionEndpoint(token, sessionID, prompt string, thinking, search bool, pow string) (*http.Response, error) {
	headers := getAuthHeaders(token)
	headers["x-ds-pow-response"] = pow
	payload := map[string]any{
		"chat_session_id":   sessionID,
		"parent_message_id": nil,
		"prompt":            prompt,
		"ref_file_ids":      []string{},
		"thinking_enabled":  thinking,
		"search_enabled":    search,
	}
	return httpPostJSON(deepseekCompletionURL, headers, payload)
}

// ========================== 路由与处理器 ==========================

type Message struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

// 消息预处理：合并同角色，添加标签，移除 markdown 图片
func messagesPrepare(messages []Message) string {
	type block struct {
		role string
		text string
	}

	processed := make([]block, 0, len(messages))
	for _, m := range messages {
		role := m.Role
		text := ""
		switch v := m.Content.(type) {
		case string:
			text = v
		case []interface{}:
			parts := make([]string, 0, len(v))
			for _, it := range v {
				if mm, ok := it.(map[string]interface{}); ok {
					if mm["type"] == "text" {
						if s, ok := mm["text"].(string); ok {
							parts = append(parts, s)
						}
					}
				}
			}
			text = strings.Join(parts, "\n")
		default:
			bs, _ := json.Marshal(v)
			text = string(bs)
		}
		processed = append(processed, block{role: role, text: text})
	}
	if len(processed) == 0 {
		return ""
	}

	// 合并连续同一角色
	merged := []block{processed[0]}
	for _, cur := range processed[1:] {
		last := &merged[len(merged)-1]
		if cur.role == last.role {
			last.text += "\n\n" + cur.text
		} else {
			merged = append(merged, cur)
		}
	}

	// 添加标签
	parts := make([]string, 0, len(merged))
	for idx, b := range merged {
		switch b.role {
		case "assistant":
			parts = append(parts, "<｜Assistant｜>"+b.text+"<｜end▁of▁sentence｜>")
		case "user", "system":
			if idx > 0 {
				parts = append(parts, "<｜User｜>"+b.text)
			} else {
				parts = append(parts, b.text)
			}
		default:
			parts = append(parts, b.text)
		}
	}
	final := strings.Join(parts, "")
	// 移除 markdown 图片，仅替换为普通链接样式
	re := regexp.MustCompile(`!\[(.*?)\]\((.*?)\)`)
	final = re.ReplaceAllString(final, "[$1]($2)")
	return final
}

// 将Claude模型名映射到DeepSeek模型
func mapClaudeModelToDeepseek(model string) string {
	l := strings.ToLower(model)
	configMutex.RLock()
	defer configMutex.RUnlock()

	fast := appConfig.ClaudeModelMapping["fast"]
	if fast == "" {
		fast = "deepseek-chat"
	}
	slow := appConfig.ClaudeModelMapping["slow"]
	if slow == "" {
		slow = "deepseek-chat"
	}

	if strings.Contains(l, "opus") || strings.Contains(l, "reasoner") || strings.Contains(l, "slow") {
		return slow
	}
	return fast
}

func handleListModels(c *gin.Context) {
	models := []map[string]any{
		{
			"id":         "deepseek-chat",
			"object":     "model",
			"created":    1677610602,
			"owned_by":   "deepseek",
			"permission": []any{},
		},
		{
			"id":         "deepseek-reasoner",
			"object":     "model",
			"created":    1677610602,
			"owned_by":   "deepseek",
			"permission": []any{},
		},
		{
			"id":         "deepseek-chat-search",
			"object":     "model",
			"created":    1677610602,
			"owned_by":   "deepseek",
			"permission": []any{},
		},
		{
			"id":         "deepseek-reasoner-search",
			"object":     "model",
			"created":    1677610602,
			"owned_by":   "deepseek",
			"permission": []any{},
		},
	}
	c.JSON(http.StatusOK, gin.H{
		"object": "list",
		"data":   models,
	})
}

func handleListClaudeModels(c *gin.Context) {
	models := []map[string]any{
		{
			"id":       "claude-sonnet-4-20250514",
			"object":   "model",
			"created":  1715635200,
			"owned_by": "anthropic",
		},
		{
			"id":       "claude-sonnet-4-20250514-fast",
			"object":   "model",
			"created":  1715635200,
			"owned_by": "anthropic",
		},
		{
			"id":       "claude-sonnet-4-20250514-slow",
			"object":   "model",
			"created":  1715635200,
			"owned_by": "anthropic",
		},
	}
	c.JSON(http.StatusOK, gin.H{
		"object": "list",
		"data":   models,
	})
}

func handleChatCompletions(c *gin.Context) {
	auth := c.GetHeader("Authorization")
	mc, err := determineModeAndToken(auth)
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
		return
	}
	// 释放账号（若使用了配置模式）
	defer func() {
		if mc.UseConfigToken && mc.Account != nil {
			accountMgr.Release(mc.Account)
		}
	}()

	var body struct {
		Model    string    `json:"model"`
		Messages []Message `json:"messages"`
		Stream   bool      `json:"stream"`
	}
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid json body"})
		return
	}
	if strings.TrimSpace(body.Model) == "" || len(body.Messages) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Request must include model and messages"})
		return
	}

	// 判定模型特性
	modelLower := strings.ToLower(body.Model)
	thinkingEnabled := false
	searchEnabled := false
	switch modelLower {
	case "deepseek-v3", "deepseek-chat":
		thinkingEnabled = false
		searchEnabled = false
	case "deepseek-r1", "deepseek-reasoner":
		thinkingEnabled = true
		searchEnabled = false
	case "deepseek-v3-search", "deepseek-chat-search":
		thinkingEnabled = false
		searchEnabled = true
	case "deepseek-r1-search", "deepseek-reasoner-search":
		thinkingEnabled = true
		searchEnabled = true
	default:
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": fmt.Sprintf("Model '%s' is not available.", body.Model)})
		return
	}

	// 准备 prompt
	finalPrompt := messagesPrepare(body.Messages)

	// 创建会话与 PoW（失败时在配置模式下尝试自动重新登录并重试一次）
	sessionID, err := createSession(mc.DeepseekToken)
	if (err != nil || sessionID == "") && mc.UseConfigToken && mc.Account != nil {
		if _, lerr := loginDeepseekViaAccount(mc.Account); lerr == nil {
			mc.DeepseekToken = mc.Account.Token
			// 持久化写回
			configMutex.Lock()
			for i := range appConfig.Accounts {
				if getAccountIdentifier(&appConfig.Accounts[i]) == getAccountIdentifier(mc.Account) {
					appConfig.Accounts[i].Token = mc.Account.Token
					_ = saveConfig("config.json", appConfig)
					break
				}
			}
			configMutex.Unlock()
			// 重试创建会话
			sessionID, err = createSession(mc.DeepseekToken)
		}
	}
	if err != nil || sessionID == "" {
		msg := "invalid token"
		if err != nil {
			msg = fmt.Sprintf("create_session failed: %v", err)
		}
		c.JSON(http.StatusUnauthorized, gin.H{"error": msg})
		return
	}
	powResp, err := getPowResponseB64(mc.DeepseekToken)
	if err != nil || powResp == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Failed to get PoW"})
		return
	}

	// 调用 completion 接口（SSE）
	resp, err := callCompletionEndpoint(mc.DeepseekToken, sessionID, finalPrompt, thinkingEnabled, searchEnabled, powResp)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("completion request failed: %v", err)})
		return
	}
	if !body.Stream {
		// 非流式：收集 SSE，拼装最终响应
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			bs, _ := io.ReadAll(resp.Body)
			c.Data(resp.StatusCode, "application/json", bs)
			return
		}
		var finalText, finalThinking string
		ptype := "text"

		sc := bufio.NewScanner(resp.Body)
		sc.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

		finished := false
		for sc.Scan() {
			line := sc.Text()
			if line == "" {
				continue
			}
			if strings.HasPrefix(line, "data:") {
				dataStr := strings.TrimSpace(line[5:])
				if dataStr == "[DONE]" {
					break
				}
				var chunk map[string]interface{}
				if err := json.Unmarshal([]byte(dataStr), &chunk); err != nil {
					continue
				}
				// 路径类型
				if p, _ := chunk["p"].(string); p != "" {
					if p == "response/thinking_content" {
						ptype = "thinking"
					} else if p == "response/content" {
						ptype = "text"
					} else if p == "response/search_status" {
						// 忽略
						continue
					}
				}
				// 内容
				if v, ok := chunk["v"]; ok {
					switch vv := v.(type) {
					case string:
						if searchEnabled && strings.HasPrefix(vv, "[citation:") {
							continue
						}
						if ptype == "thinking" {
							finalThinking += vv
						} else {
							finalText += vv
						}
					case []interface{}:
						for _, it := range vv {
							if mm, ok := it.(map[string]interface{}); ok {
								if mm["p"] == "status" && fmt.Sprint(mm["v"]) == "FINISHED" {
									finished = true
									break
								}
							}
						}
					}
				}
			}
			if finished {
				break
			}
		}
		created := time.Now().Unix()
		promptTokens := len(finalPrompt) / 4
		reasoningTokens := len(finalThinking) / 4
		completionTokens := len(finalText) / 4

		out := map[string]interface{}{
			"id":      sessionID,
			"object":  "chat.completion",
			"created": created,
			"model":   body.Model,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]interface{}{
						"role":              "assistant",
						"content":           finalText,
						"reasoning_content": finalThinking,
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]interface{}{
				"prompt_tokens":     promptTokens,
				"completion_tokens": reasoningTokens + completionTokens,
				"total_tokens":      promptTokens + reasoningTokens + completionTokens,
				"completion_tokens_details": map[string]interface{}{
					"reasoning_tokens": reasoningTokens,
				},
			},
		}
		c.JSON(http.StatusOK, out)
		return
	}

	// 流式：SSE 转换为 OpenAI 片段
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bs, _ := io.ReadAll(resp.Body)
		c.Data(resp.StatusCode, "application/json", bs)
		return
	}

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")

	writer := c.Writer
	flusher, _ := writer.(http.Flusher)

	created := time.Now().Unix()
	completionID := sessionID
	firstChunkSent := false
	var finalText, finalThinking string
	ptype := "text"

	flushEvent := func(payload any) {
		bs, _ := json.Marshal(payload)
		_, _ = writer.Write([]byte("data: " + string(bs) + "\n\n"))
		if flusher != nil {
			flusher.Flush()
		}
	}

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

	defer resp.Body.Close()

	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		dataStr := strings.TrimSpace(line[5:])
		if dataStr == "[DONE]" {
			// 发送最终统计片段
			promptTokens := len(finalPrompt) / 4
			thinkingTokens := len(finalThinking) / 4
			completionTokens := len(finalText) / 4
			usage := map[string]interface{}{
				"prompt_tokens":     promptTokens,
				"completion_tokens": thinkingTokens + completionTokens,
				"total_tokens":      promptTokens + thinkingTokens + completionTokens,
				"completion_tokens_details": map[string]interface{}{
					"reasoning_tokens": thinkingTokens,
				},
			}
			finish := map[string]interface{}{
				"id":      completionID,
				"object":  "chat.completion.chunk",
				"created": created,
				"model":   body.Model,
				"choices": []map[string]interface{}{
					{
						"delta":         map[string]interface{}{},
						"index":         0,
						"finish_reason": "stop",
					},
				},
				"usage": usage,
			}
			flushEvent(finish)
			_, _ = writer.Write([]byte("data: [DONE]\n\n"))
			if flusher != nil {
				flusher.Flush()
			}
			break
		}

		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(dataStr), &chunk); err != nil {
			continue
		}
		// 路径类型
		if p, _ := chunk["p"].(string); p != "" {
			if p == "response/search_status" {
				continue
			}
			if p == "response/thinking_content" {
				ptype = "thinking"
			} else if p == "response/content" {
				ptype = "text"
			}
		}

		sent := false
		if v, ok := chunk["v"]; ok {
			switch vv := v.(type) {
			case string:
				if searchEnabled && strings.HasPrefix(vv, "[citation:") {
					continue
				}
				delta := map[string]interface{}{}
				if !firstChunkSent {
					delta["role"] = "assistant"
					firstChunkSent = true
				}
				if ptype == "thinking" {
					if thinkingEnabled {
						finalThinking += vv
						delta["reasoning_content"] = vv
					}
				} else {
					finalText += vv
					delta["content"] = vv
				}
				if len(delta) > 0 {
					out := map[string]interface{}{
						"id":      completionID,
						"object":  "chat.completion.chunk",
						"created": created,
						"model":   body.Model,
						"choices": []map[string]interface{}{
							{
								"delta": delta,
								"index": 0,
							},
						},
					}
					flushEvent(out)
					sent = true
				}
			case []interface{}:
				for _, it := range vv {
					if mm, ok := it.(map[string]interface{}); ok {
						if mm["p"] == "status" && fmt.Sprint(mm["v"]) == "FINISHED" {
							// 发送 finish
							promptTokens := len(finalPrompt) / 4
							thinkingTokens := len(finalThinking) / 4
							completionTokens := len(finalText) / 4
							usage := map[string]interface{}{
								"prompt_tokens":     promptTokens,
								"completion_tokens": thinkingTokens + completionTokens,
								"total_tokens":      promptTokens + thinkingTokens + completionTokens,
								"completion_tokens_details": map[string]interface{}{
									"reasoning_tokens": thinkingTokens,
								},
							}
							finish := map[string]interface{}{
								"id":      completionID,
								"object":  "chat.completion.chunk",
								"created": created,
								"model":   body.Model,
								"choices": []map[string]interface{}{
									{
										"delta":         map[string]interface{}{},
										"index":         0,
										"finish_reason": "stop",
									},
								},
								"usage": usage,
							}
							flushEvent(finish)
							_, _ = writer.Write([]byte("data: [DONE]\n\n"))
							if flusher != nil {
								flusher.Flush()
							}
							return
						}
					}
				}
			}
		}
		if !sent {
			// 忽略无法识别块
			continue
		}
	}
}

func handleClaudeMessages(c *gin.Context) {
	auth := c.GetHeader("Authorization")
	mc, err := determineModeAndToken(auth)
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
		return
	}
	defer func() {
		if mc.UseConfigToken && mc.Account != nil {
			accountMgr.Release(mc.Account)
		}
	}()

	var body struct {
		Model    string                   `json:"model"`
		Messages []map[string]interface{} `json:"messages"`
		Stream   bool                     `json:"stream"`
		System   string                   `json:"system"`
		Tools    []interface{}            `json:"tools"`
	}
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid json body"})
		return
	}
	if strings.TrimSpace(body.Model) == "" || len(body.Messages) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Request must include model and messages"})
		return
	}

	// 规范化Claude消息为通用消息
	norm := make([]Message, 0, len(body.Messages)+1)
	if strings.TrimSpace(body.System) != "" {
		norm = append(norm, Message{Role: "system", Content: body.System})
	}
	for _, m := range body.Messages {
		role, _ := m["role"].(string)
		content := m["content"]
		switch v := content.(type) {
		case string:
			norm = append(norm, Message{Role: role, Content: v})
		case []interface{}:
			parts := make([]string, 0, len(v))
			for _, it := range v {
				if mm, ok := it.(map[string]interface{}); ok {
					if mm["type"] == "text" {
						if s, ok := mm["text"].(string); ok {
							parts = append(parts, s)
						}
					} else if mm["type"] == "tool_result" {
						if s, ok := mm["content"].(string); ok {
							parts = append(parts, s)
						}
					}
				}
			}
			norm = append(norm, Message{Role: role, Content: strings.Join(parts, "\n")})
		default:
			bs, _ := json.Marshal(v)
			norm = append(norm, Message{Role: role, Content: string(bs)})
		}
	}

	deepseekModel := mapClaudeModelToDeepseek(body.Model)

	// 判定模型特性（与 /v1/chat/completions 对齐）
	thinkingEnabled := false
	searchEnabled := false
	switch strings.ToLower(deepseekModel) {
	case "deepseek-v3", "deepseek-chat":
		thinkingEnabled = false
		searchEnabled = false
	case "deepseek-r1", "deepseek-reasoner":
		thinkingEnabled = true
		searchEnabled = false
	case "deepseek-v3-search", "deepseek-chat-search":
		thinkingEnabled = false
		searchEnabled = true
	case "deepseek-r1-search", "deepseek-reasoner-search":
		thinkingEnabled = true
		searchEnabled = true
	default:
		thinkingEnabled = false
		searchEnabled = false
	}

	prompt := messagesPrepare(norm)
	sessionID, err := createSession(mc.DeepseekToken)
	if err != nil || sessionID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "invalid token"})
		return
	}
	powResp, err := getPowResponseB64(mc.DeepseekToken)
	if err != nil || powResp == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Failed to get PoW"})
		return
	}

	// 仅实现非流式，流式留待后续
	if body.Stream {
		c.JSON(http.StatusNotImplemented, gin.H{"error": "stream not implemented yet"})
		return
	}

	resp, err := callCompletionEndpoint(mc.DeepseekToken, sessionID, prompt, thinkingEnabled, searchEnabled, powResp)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("completion request failed: %v", err)})
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		bs, _ := io.ReadAll(resp.Body)
		c.Data(resp.StatusCode, "application/json", bs)
		return
	}

	// 收集内容
	var finalText, finalThinking string
	ptype := "text"
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)
	finished := false
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "data:") {
			dataStr := strings.TrimSpace(line[5:])
			if dataStr == "[DONE]" {
				break
			}
			var chunk map[string]interface{}
			if err := json.Unmarshal([]byte(dataStr), &chunk); err != nil {
				continue
			}
			if p, _ := chunk["p"].(string); p != "" {
				if p == "response/search_status" {
					continue
				}
				if p == "response/thinking_content" {
					ptype = "thinking"
				} else if p == "response/content" {
					ptype = "text"
				}
			}
			if v, ok := chunk["v"]; ok {
				switch vv := v.(type) {
				case string:
					if ptype == "thinking" {
						finalThinking += vv
					} else {
						finalText += vv
					}
				case []interface{}:
					for _, it := range vv {
						if mm, ok := it.(map[string]interface{}); ok {
							if mm["p"] == "status" && fmt.Sprint(mm["v"]) == "FINISHED" {
								finished = true
								break
							}
						}
					}
				}
			}
		}
		if finished {
			break
		}
	}

	created := time.Now().Unix()
	promptTokens := len(prompt) / 4
	reasoningTokens := len(finalThinking) / 4
	completionTokens := len(finalText) / 4

	out := map[string]interface{}{
		"id":            fmt.Sprintf("msg_%d", created),
		"type":          "message",
		"role":          "assistant",
		"model":         body.Model,
		"content":       []map[string]interface{}{{"type": "thinking", "thinking": finalThinking}, {"type": "text", "text": finalText}},
		"stop_reason":   "end_turn",
		"stop_sequence": nil,
		"usage": map[string]interface{}{
			"input_tokens":  promptTokens,
			"output_tokens": reasoningTokens + completionTokens,
		},
	}
	c.JSON(http.StatusOK, out)
}

func handleIndex(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"service": "deepseek-go-proxy",
		"version": "0.1.0",
		"status":  "ok",
	})
}

// ========================== 中间件 ==========================

func corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Credentials", "true")
		c.Header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if c.Request.Method == http.MethodOptions {
			c.Status(http.StatusNoContent)
			c.Abort()
			return
		}
		c.Next()
	}
}

func requestLogger() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		c.Next()
		lat := time.Since(start)
		log.Printf("%s %s -> %d (%s)",
			c.Request.Method, c.Request.URL.Path, c.Writer.Status(), lat)
	}
}

// ========================== 服务器启动 ==========================

func setupRouter() *gin.Engine {
	r := gin.New()
	r.Use(gin.Recovery(), corsMiddleware(), requestLogger())

	r.GET("/", handleIndex)
	r.GET("/v1/models", handleListModels)
	r.GET("/anthropic/v1/models", handleListClaudeModels)
	r.POST("/v1/chat/completions", handleChatCompletions)
	r.POST("/anthropic/v1/messages", handleClaudeMessages)

	return r
}

func main() {
	// 加载配置
	cfg, err := loadConfig("config.json")
	if err != nil {
		log.Printf("load config failed: %v", err)
	}
	configMutex.Lock()
	appConfig = cfg
	configMutex.Unlock()

	// 初始化账号池
	accountMgr = NewAccountManager(cfg.Accounts)

	// 端口
	port := os.Getenv("PORT")
	if port == "" {
		port = "5001"
	}

	log.Printf("Starting deepseek-go-proxy on :%s", port)
	if err := setupRouter().Run("0.0.0.0:" + port); err != nil {
		log.Fatalf("server failed: %v", err)
	}
}
