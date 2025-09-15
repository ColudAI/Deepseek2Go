# syntax=docker/dockerfile:1

# ========= Build Stage =========
FROM golang:1.22-alpine AS build
RUN apk add --no-cache ca-certificates upx && update-ca-certificates
WORKDIR /src

# 拷贝全部仓库（包含 Reloaded-Version 源码与根目录 wasm）
COPY . .

ENV CGO_ENABLED=0 \
    GOOS=linux \
    GOARCH=amd64

# 使用构建缓存
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    go build -trimpath -ldflags="-s -w" -o /server ./Reloaded-Version/cmd/server

# ========= Runtime Stage =========
FROM alpine:3.19
RUN adduser -D -g '' appuser && apk add --no-cache ca-certificates && update-ca-certificates

# 注意：将工作目录设为 /app/Reloaded-Version，这样 config.json 内的 "../sha3_wasm_bg.7b9ca65ddd.wasm" 可解析到 /app/sha3_wasm_bg.7b9ca65ddd.wasm
WORKDIR /app/Reloaded-Version

# 拷贝二进制与配置、WASM
COPY --from=build /server /app/server
COPY Reloaded-Version/config.json ./config.json
COPY sha3_wasm_bg.7b9ca65ddd.wasm /app/sha3_wasm_bg.7b9ca65ddd.wasm

ENV PORT=5001
EXPOSE 5001
USER appuser

# 相对工作目录执行上级 server 二进制
CMD ["../server"]