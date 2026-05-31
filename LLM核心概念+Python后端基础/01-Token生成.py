"""
01-Token生成.py - 学习LLM Token概念与Temperature参数

Token是LLM理解语言的基本单位，不是单词，也不是字符，而是介于两者之间的片段
Temperature参数控制输出的随机性：0=最确定，1.0=最随机
"""

# ================================================================================
# 第一部分：用 tiktoken 实际统计 Token 数量
# ================================================================================

# 导入 tiktoken 库 - OpenAI官方的Token统计库
import tiktoken

# 获取指定编码模型的编码器
# "cl100k_base" 是 GPT-3.5/GPT-4 系列使用的编码方式
enc = tiktoken.get_encoding("cl100k_base")

# 定义要统计Token的文本
text = "帮我查询北京今天的天气"

# enc.encode(text) 将文本编码成Token序列（一个整数列表）
# len() 计算这个列表的长度，就是Token数量
print(f"'{text}' = {len(enc.encode(text))} tokens")


# ================================================================================
# 第二部分：测试不同 Temperature（温度）参数对输出的影响
# ================================================================================

# 从 openai 库导入 OpenAI 客户端类
# 虽然我们用的是 Ollama，但它兼容 OpenAI API 格式，所以可以用这个库
from openai import OpenAI

# 创建 OpenAI 客户端对象
# api_key="ollama" - Ollama不需要真正的API密钥，但OpenAI库要求填
# base_url - Ollama服务的本地地址和端口
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

# 使用 for 循环测试不同的 temperature 值
# [0, 0.5, 1.0] - 三个不同的温度值
for temp in [0, 0.5, 1.0]:
    
    # 调用 LLM 聊天接口
    resp = client.chat.completions.create(
        
        # 指定使用的模型 - "deepseek-r1:7b" 是DeepSeek R1的70亿参数版本
        model="deepseek-r1:7b",
        
        # 消息列表，指定用户输入的内容
        messages=[{"role": "user", "content": "用一句话介绍北京"}],
        
        # temperature - 温度参数，控制输出的随机性
        # 0 - 输出最确定，每次几乎一样
        # 0.5 - 中等随机性
        # 1.0 - 高随机性，输出变化很大
        temperature=temp
    )
    
    # 打印结果
    # resp.choices[0] - 第一个选择（因为我们只请求一个）
    # .message.content - 消息的实际内容
    # [:50] - 只取前50个字符，避免输出太长
    print(f"Temperature={temp}: {resp.choices[0].message.content[:50]}...")


"""
================================================================================
运行前准备：
1. 确保已安装 tiktoken 和 openai 库
   pip install tiktoken openai

2. 确保已安装 Ollama 并下载了对应模型（这里用 deepseek-r1:7b）
   ollama pull deepseek-r1:7b

3. 确保 Ollama 服务正在运行
   ollama serve

运行预期结果：
- Temperature=0: 每次输出几乎相同，最保守
- Temperature=0.5: 有一定变化，但不会太离谱
- Temperature=1.0: 每次输出差异很大，更有创造性但可能出错

================================================================================
"""
