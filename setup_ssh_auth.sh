#!/bin/bash

echo "🔐 GitHub SSH认证设置脚本"
echo "========================"
echo ""

# 检查是否已有SSH密钥
if [ -f ~/.ssh/id_rsa.pub ]; then
    echo "✅ 检测到现有SSH密钥:"
    cat ~/.ssh/id_rsa.pub
    echo ""
    read -p "是否使用现有密钥？(y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "🔑 生成新的SSH密钥..."
        ssh-keygen -t rsa -b 4096 -C "glade@$(hostname)" -f ~/.ssh/id_rsa_glade
        echo "✅ 新密钥已生成: ~/.ssh/id_rsa_glade.pub"
        cat ~/.ssh/id_rsa_glade.pub
    fi
else
    echo "🔑 生成SSH密钥..."
    read -p "请输入您的邮箱地址: " email
    ssh-keygen -t rsa -b 4096 -C "$email"
    echo "✅ SSH密钥已生成:"
    cat ~/.ssh/id_rsa.pub
fi

echo ""
echo "📋 接下来的步骤:"
echo "1. 复制上面显示的SSH公钥"
echo "2. 访问: https://github.com/settings/ssh/new"
echo "3. 粘贴公钥并保存"
echo "4. 测试连接: ssh -T git@github.com"
echo ""

read -p "已完成SSH密钥配置？(y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔗 更新远程仓库为SSH URL..."
    cd /home/luukiaun/glafic251018/work/glade
    git remote set-url origin git@github.com:y31ling/glaDE.git
    
    echo "🧪 测试SSH连接..."
    if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
        echo "✅ SSH连接成功！"
        echo "🚀 现在可以推送到GitHub："
        echo "git push -u origin main"
    else
        echo "❌ SSH连接失败，请检查密钥配置"
    fi
else
    echo "⏸️  请完成SSH密钥配置后再运行此脚本"
fi