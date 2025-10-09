"""
清理results文件夹脚本
Clean Results Folder Script

使用方法：
1. 关闭所有打开的Excel文件
2. 运行此脚本: python clean_results.py
"""

import os
import shutil
import time

def clean_results_folder():
    """清理results文件夹中的所有文件"""
    results_dir = "results"
    
    if not os.path.exists(results_dir):
        print(f"❌ {results_dir}文件夹不存在")
        return
    
    print(f"🧹 开始清理 {results_dir} 文件夹...")
    print(f"⚠️  请确保所有Excel文件已关闭！")
    print(f"⏳ 3秒后开始清理...")
    time.sleep(3)
    
    files = os.listdir(results_dir)
    if not files:
        print(f"✅ {results_dir}文件夹已经是空的")
        return
    
    success_count = 0
    fail_count = 0
    failed_files = []
    
    for filename in files:
        file_path = os.path.join(results_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                success_count += 1
                print(f"   ✓ 删除: {filename}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                success_count += 1
                print(f"   ✓ 删除文件夹: {filename}")
        except Exception as e:
            fail_count += 1
            failed_files.append(filename)
            print(f"   ✗ 删除失败: {filename}")
            print(f"     错误: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"📊 清理完成!")
    print(f"   成功删除: {success_count} 个文件/文件夹")
    if fail_count > 0:
        print(f"   失败: {fail_count} 个文件")
        print(f"\n❌ 删除失败的文件:")
        for f in failed_files:
            print(f"   - {f}")
        print(f"\n💡 提示: 请关闭这些文件后重新运行此脚本")
    else:
        print(f"   ✅ 所有文件已清理完毕！")
    print(f"{'='*60}")

if __name__ == "__main__":
    clean_results_folder()
