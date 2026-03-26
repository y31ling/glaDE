#!/bin/bash

# GLADE GitHub部署脚本
# 使用方法: ./deploy_to_github.sh [repository_url]

set -e

echo "🚀 GLADE GitHub部署脚本"
echo "========================="

# 检查是否提供了仓库URL
if [ $# -eq 0 ]; then
    echo "❌ 错误: 请提供GitHub仓库URL"
    echo "使用方法: $0 <repository_url>"
    echo "示例: $0 https://github.com/username/glade.git"
    exit 1
fi

REPO_URL=$1
echo "📍 目标仓库: $REPO_URL"

# 检查是否在正确的目录
if [ ! -f "main.py" ] || [ ! -f "bootstrap_linux.sh" ]; then
    echo "❌ 错误: 请在glade项目根目录运行此脚本"
    exit 1
fi

# 检查Git状态
if [ ! -d ".git" ]; then
    echo "❌ 错误: 未检测到Git仓库，请先运行 git init"
    exit 1
fi

# 检查是否有未提交的更改
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "⚠️  检测到未提交的更改"
    read -p "是否要提交这些更改? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📝 提交更改..."
        git add .
        git commit -m "Update before GitHub deployment"
    else
        echo "❌ 请先提交或撤销更改后再运行部署脚本"
        exit 1
    fi
fi

# 添加远程仓库
echo "🔗 配置远程仓库..."
if git remote | grep -q "origin"; then
    echo "   更新现有origin..."
    git remote set-url origin "$REPO_URL"
else
    echo "   添加新的origin..."
    git remote add origin "$REPO_URL"
fi

# 推送到GitHub
echo "📤 推送到GitHub..."
git push -u origin main

# 检查推送结果
if [ $? -eq 0 ]; then
    echo "✅ 成功部署到GitHub!"
    echo ""
    echo "🎉 部署完成!"
    echo "📋 接下来的步骤:"
    echo "   1. 访问: ${REPO_URL%%.git}"
    echo "   2. 检查README.md显示是否正常"
    echo "   3. 设置仓库描述和标签"
    echo "   4. 启用GitHub Pages (可选)"
    echo "   5. 配置协作者权限 (可选)"
    echo ""
    echo "📚 项目文档:"
    echo "   - README_GITHUB.md: 完整的GitHub展示文档"
    echo "   - LICENSE: MIT许可证"
    echo "   - .github/workflows/ci.yml: 自动化测试"
    echo ""
    echo "🔧 本地开发:"
    echo "   - 克隆: git clone $REPO_URL"
    echo "   - 安装: bash bootstrap_linux.sh"
    echo "   - 运行: ./run_glade.sh"
else
    echo "❌ 推送失败!"
    echo "可能的原因:"
    echo "   1. 仓库URL不正确"
    echo "   2. 没有推送权限"
    echo "   3. 网络连接问题"
    echo "   4. 仓库不存在或未初始化"
    echo ""
    echo "🔧 解决方案:"
    echo "   1. 检查GitHub仓库是否已创建"
    echo "   2. 确认有推送权限"
    echo "   3. 检查SSH密钥或访问令牌"
    exit 1
fi