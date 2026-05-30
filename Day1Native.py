# Day1：原生智能体实现（无框架依赖，手写范式规则）
# 查询天气，硬编码输出Thought和Action，模拟Agent流程
# 1. 定义原生工具（get_weather）
# 2. 定义原生系统提示（手写范式规则）
# 3. 实现原生Agent核心循环（手动维护历史，模拟LLM推理）
# 4. 输出最终结果，展示原生智能体的工作流程


import requests

# 原生工具定义（无任何封装）
def get_weather(city: str) -> str:
    """原生天气查询工具，无框架依赖"""
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

# 原生系统提示（手写范式规则）
SYSTEM_PROMPT = """
你是原生智能体，严格遵循ReAct范式，仅输出两种格式：
1. 工具调用：Thought: 思考内容\nAction: get_weather(city="城市名")
2. 任务结束：Thought: 思考内容\nAction: Finish[最终回答内容]
禁止多余文字、禁止换行错乱、禁止直接回答问题！
"""

# 原生Agent核心循环（手动实现，框架底层原型）
def native_agent_loop(user_query: str, max_iter: int = 3):
    history = []
    for i in range(max_iter):
        # 拼接上下文（原生手动维护）
        prompt = SYSTEM_PROMPT + f"\n用户问题：{user_query}\n历史记录：{history}"
        # 模拟LLM推理（聚焦流程，后续Day2对接真实模型）
        if i == 0:
            thought = "用户需要查询城市天气，必须调用天气工具获取实时数据"
            action = 'get_weather(city="北京")'
        else:
            thought = "已获取完整天气信息，可直接回答用户问题"
            action = "Finish[已完成天气查询任务]"
        
        print(f"===== 第{i+1}轮迭代 =====")
        print(f"Thought: {thought}")
        print(f"Action: {action}")

        # 原生执行逻辑
        if action.startswith("Finish"):
            final_ans = action.replace("Finish[", "").replace("]", "")
            return f"任务完成：{final_ans}"
        
        # 原生工具调用
        obs = get_weather("北京")
        print(f"Observation: {obs}")
        history.append(f"第{i+1}轮：{thought} | 执行结果：{obs}")
    
    return "任务超时：超出最大迭代次数"

if __name__ == "__main__":
    res = native_agent_loop("帮我查询北京今天的天气")
    print(f"\n最终结果：{res}")