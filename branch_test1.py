"""
branch_test1.py - 测试文件
用于测试 Git 分支功能的示例代码
"""

def hello_world():
    """打印问候语"""
    print("Hello from branch_test1!")
    print("这是一个用于测试 Git 分支的文件")

def add(a, b):
    """简单的加法函数"""
    return a + b

if __name__ == "__main__":
    hello_world()
    
    # 测试加法函数
    result = add(5, 3)
    print(f"5 + 3 = {result}")
