# ====================== 1. 导入所有依赖 ======================
import re
import os
import requests
from tavily import TavilyClient
from openai import OpenAI

# ====================== 2. 智能体指令模板（System Prompt） ======================
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。
# 可用工具:- `get_weather(city: str)`: 查询指定城市的实时天气。- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。
# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：
Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]
Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]
请开始吧！
# 重要提示:
# - 每次只输出一对Thought-Action
# - Action必须在同一行，不要换行
# - 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束
"""

# ====================== 3. 工具1：查询天气（基于 requests 调用公开API） ======================
def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息。
    """
    url = f"https://wttr.in/{city}?format=j1"
    try:
        # requests.get：发起HTTP GET请求（你问的requests核心用法）
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # 解析天气数据
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']

        return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"

    except requests.exceptions.RequestException as e:
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"

# ====================== 4. 工具2：搜索旅游景点（基于 Tavily API） ======================
def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"

    tavily = TavilyClient(api_key=api_key)
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        if response.get("answer"):
            return response["answer"]

        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐。"
        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"

# ====================== 5. 工具字典（统一管理所有可用工具） ======================
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

# ====================== 6. 大模型通用客户端（兼容OpenAI/本地模型） ======================
class OpenAICompatibleClient:
    """
    调用任何兼容OpenAI接口的LLM服务
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用LLM API生成回复"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"

# ====================== 7. 全局配置（重点！这里替换你的密钥和地址） ======================
# ---------- 配置1：Tavily 搜索密钥（必配） ----------
TAVILY_API_KEY = "你的Tavily密钥"
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

# ---------- 配置2：大模型 LLM 配置（二选一：本地Ollama / OpenAI官方） ----------
# 方案A：使用【本地Ollama模型】（免费、无需翻墙，推荐新手）
API_KEY = "ollama"  # Ollama固定填这个
BASE_URL = "http://localhost:11434/v1"  # Ollama本地固定地址
MODEL_ID = "qwen:7b"  # 你本地拉取的模型名，根据实际修改

# 方案B：使用【OpenAI官方GPT】（需要密钥+外网，有条件再用）
# API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
# BASE_URL = "https://api.openai.com/v1"
# MODEL_ID = "gpt-3.5-turbo"

# 初始化大模型客户端
llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)

# ====================== 8. 主循环（智能体核心执行逻辑） ======================
if __name__ == "__main__":
    # 用户提问
    user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
    prompt_history = [f"用户请求: {user_prompt}"]
    print(f"用户输入: {user_prompt}\n" + "="*40)

    # 最大循环5次，防止死循环
    for i in range(5):
        print(f"--- 循环 {i+1} ---\n")
        full_prompt = "\n".join(prompt_history)

        # 调用大模型思考
        llm_output = llm.generate(full_prompt, AGENT_SYSTEM_PROMPT)

        # 截断多余输出（修复原文正则错误）
        pattern = r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|$)'
        match = re.search(pattern, llm_output, re.DOTALL)
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("已截断多余的 Thought-Action 对")

        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)

        # 解析 Action 执行工具
        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            observation = "错误: 未能解析到 Action 字段。请严格遵循格式。"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "="*40)
            prompt_history.append(observation_str)
            continue

        action_str = action_match.group(1).strip()

        # 判断是否结束任务
        if action_str.startswith("Finish"):
            finish_match = re.match(r"Finish\[(.*)\]", action_str)
            if finish_match:
                final_answer = finish_match.group(1)
                print(f"任务完成，最终答案: {final_answer}")
            break

        # 解析工具名和参数
        tool_match = re.search(r"(\w+)\(", action_str)
        args_match = re.search(r"\((.*)\)", action_str)
        if not tool_match or not args_match:
            observation = "错误: 工具调用格式错误"
        else:
            tool_name = tool_match.group(1)
            args_str = args_match.group(1)
            kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
            # 执行工具
            if tool_name in available_tools:
                observation = available_tools[tool_name](**kwargs)
            else:
                observation = f"错误:未定义的工具 '{tool_name}'"

        # 记录工具返回结果（Observation）
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)