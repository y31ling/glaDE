#!/bin/bash

echo "🔑 GitHub认证设置脚本"
echo "===================="
echo ""
echo "由于GitHub已停止支持密码认证，您需要使用Personal Access Token (PAT)"
echo ""
echo "📋 步骤1: 创建Personal Access Token"
echo "1. 访问: https://github.com/settings/tokens"
echo "2. 点击 'Generate new token (classic)'"
echo "3. 设置过期时间（建议90天或更长）"
echo "4. 选择权限范围："
echo "   ✓ repo (完整仓库访问权限)"
echo "   ✓ workflow (如果需要GitHub Actions)"
echo "5. 点击 'Generate token'"
echo "6. 复制生成的token（只显示一次！）"
echo ""
echo "📋 步骤2: 配置本地认证"
echo ""

read -p "请输入您的GitHub用户名: " github_username
echo ""
read -s -p "请输入您的Personal Access Token: " github_token
echo ""
echo ""

# 更新远程URL以包含认证信息
cd /home/luukiaun/glafic251018/work/glade

# 构建包含认证的URL
auth_url="https://${github_username}:${github_token}@github.com/y31ling/glaDE.git"

echo "🔗 更新远程仓库URL..."
git remote set-url origin "$auth_url"

echo "✅ 认证配置完成！"
echo ""
echo "🚀 现在可以推送到GitHub："
echo "git push -u origin main"
echo ""
echo "⚠️  安全提示："
echo "- Token已保存在本地git配置中"
echo "- 请妥善保管您的Personal Access Token"
echo "- 如果token泄露，请立即在GitHub上撤销并重新生成"