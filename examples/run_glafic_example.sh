#!/bin/bash
# run_glafic.py 使用示例

echo "======================================================================"
echo "Run GLAFIC Tool - 使用示例"
echo "======================================================================"

GLADE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$GLADE_ROOT"

echo ""
echo "示例1: 基本使用（自动查找并运行）"
echo "----------------------------------------------------------------------"
echo "命令: python tools/run_glafic.py <folder_path>"
echo ""
echo "注意：请将 <folder_path> 替换为实际包含 best_params.txt 的文件夹路径"
echo ""

# 示例2
echo ""
echo "示例2: 指定输出目录和前缀"
echo "----------------------------------------------------------------------"
echo "命令: python tools/run_glafic.py <folder> --output ./output --prefix my_run"
echo ""

# 示例3
echo ""
echo "示例3: 详细输出模式（显示 glafic 完整输出）"
echo "----------------------------------------------------------------------"
echo "命令: python tools/run_glafic.py <folder> --verbose"
echo ""

# 实际可运行的示例（如果文件存在）
echo ""
echo "======================================================================"
echo "实际运行示例（如果文件存在）"
echo "======================================================================"
echo ""

# 查找第一个可用的 best_params 文件
SAMPLE_FOLDER=$(find "$GLADE_ROOT/../results" -name "*_best_params.txt" -type f 2>/dev/null | head -1 | xargs dirname)

if [ -n "$SAMPLE_FOLDER" ]; then
    echo "找到示例文件夹: $SAMPLE_FOLDER"
    echo ""
    echo "运行命令:"
    echo "  python tools/run_glafic.py \"$SAMPLE_FOLDER\" --prefix example_run"
    echo ""
    
    read -p "是否运行示例？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 tools/run_glafic.py "$SAMPLE_FOLDER" --prefix example_run
    fi
else
    echo "未找到示例文件夹（无 best_params.txt 文件）"
    echo "请先运行优化程序生成结果"
fi

echo ""
echo "======================================================================"
echo "更多信息请查看: tools/README_run_glafic.md"
echo "======================================================================"
