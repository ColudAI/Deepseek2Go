#!/bin/bash

# 设置目标平台
PLATFORMS=("windows/amd64" "windows/arm64" "darwin/amd64" "darwin/arm64" "linux/amd64" "linux/arm64")

# 清理旧的构建产物
rm -rf dist
mkdir -p dist

# 循环编译
for PLATFORM in "${PLATFORMS[@]}"
do
    # 分割平台字符串
    GOOS=${PLATFORM%/*}
    GOARCH=${PLATFORM#*/}
    
    # 设置输出文件名
    OUTPUT_NAME="dist/server-${GOOS}-${GOARCH}"
    if [ $GOOS = "windows" ]; then
        OUTPUT_NAME+=".exe"
    fi
    
    # 编译
    echo "Building for ${GOOS}/${GOARCH}..."
    env GOOS=$GOOS GOARCH=$GOARCH go build -o $OUTPUT_NAME ./cmd/server
    if [ $? -ne 0 ]; then
        echo "An error has occurred! Aborting the script execution..."
        exit 1
    fi
done

echo "Build finished successfully!"