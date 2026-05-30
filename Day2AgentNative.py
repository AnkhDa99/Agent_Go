import re
import requests
from openai import OpenAI

# 初始化本地Ollama模型（兼容openAI接口）
client = OpenAI(api_key= "ollama", base_url="http://localhost:11434/v1")
MODEL = "deepseek-r1:7b"

# 定义原生工具（get_weather）
def get_weather(city: str) -> str:
    try:
        url = f"https://wttr.in/{city}?format=j1"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        desc = data["current_condition"][0]["weatherDesc"][0]["value"]
        temp = data["current_condition"][0]["temp_C"]
        return f"【{city}实时天气】{desc}，气温{temp}℃"
    except Exception as e:
        return f"天气查询失败：{str(e)}"
    
# 工具注册表（原生手写，对标langchain工具管理）
TOOL_REGISTRY = {
    "get_weather": get_weather
}

# 严格范式提示词
SYSTEM_PROMPT = """
你是标准化ReAct智能体，严格输出固定格式，禁止自由发挥！
格式规范：
Thought: 你的推理思考过程
Action: 工具名(参数名="参数值") 或 Finish[最终回答]
支持工具：get_weather(city="城市名")
"""

# 原生LLM推理函数
def llm_infer(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# 手撕核心解析器(LangChain底层源码复刻)
def parse_agent_response(response: str):
    """
    原生手写Agent输出解析器
    返回：类型(finish/tool)、工具名、参数/结果
    """
    # 正则匹配Thought和Action
    thought_pattern = r"Thought:\s*(.*?)(?=\nAction:)"
    action_pattern = r"Action:\s*(.*)"
    
    thought = re.search(thought_pattern, response, re.S).group(1).strip()
    action = re.search(action_pattern, response).group(1).strip()

    # 判断任务是否结束
    if action.startswith("Finish"):
        result = action.replace("Finish[", "").replace("]", "")
        return "finish", None, result, thought
    
    # 解析工具名和参数
    tool_name = re.match(r"^(\w+)\(", action).group(1)
    # 解析键值对参数
    args = dict(re.findall(r'(\w+)="([^"]+)"', action))
    return "tool", tool_name, args, thought

if __name__ == "__main__":
    model_output = llm_infer("请查询北京的实时天气，并根据天气情况推荐是否适合出行。")
    print("模型输出：", model_output)

    type_, tool, args, thought = parse_agent_response(model_output)
    print(f"解析结果｜思考：{thought}")
    if type_ == "tool":
        tool_res = TOOL_REGISTRY[tool](**args)
        print(f"工具执行结果：{tool_res}")
    else:
        print(f"最终回答：{args}")