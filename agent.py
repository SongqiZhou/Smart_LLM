import gradio as gr
from zhipuai import ZhipuAI

# 定义 CoT 环境类
class CoTEnvironment:
    def __init__(self, api_key, model="glm-4-plus", top_p=0.1, temperature=0.1, max_tokens=600):
        self.client = ZhipuAI(api_key=api_key)
        self.model = model
        self.top_p = top_p
        self.temperature = temperature
        self.max_tokens = max_tokens

    def ask_model(self, system_prompt, user_prompt):
        """
        通过 ZhipuAI 调用 GLM API
        :param system_prompt: 系统设定的角色说明
        :param user_prompt: 用户问题
        :return: 模型返回的答案
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                top_p=self.top_p,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,  # 不使用流式响应
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

# 定义智能体类
class CoTAgent:
    def __init__(self, name, system_message, environment):
        self.name = name
        self.system_message = system_message
        self.environment = environment

    def think(self, user_prompt):
        return self.environment.ask_model(self.system_message, user_prompt)

# 定义具体角色
class ProblemAnalyzer(CoTAgent):
    """
    分析问题，将复杂问题分解为逻辑步骤。
    """
    def __init__(self, environment):
        super().__init__(
            name="Problem Analyzer",
            system_message=(
                "你是理解复杂问题的专家。"
                "你的职责是确定问题的关键要素，并将其分解为一个结构化的、循序渐进的计划。"
                "专注于分析的清晰度和完整性。"
            ),
            environment=environment,
        )

class StepReasoner(CoTAgent):
    """
    基于分解的步骤进行详细推理。
    """
    def __init__(self, environment):
        super().__init__(
            name="Step-by-Step Reasoner",
            system_message=(
                "你是一个乐于解答各种问题的助手，你的任务是为针对上述步骤给出回答。"
            ),
            environment=environment,
        )

class Verifier(CoTAgent):
    """
    验证推理的逻辑一致性和结果的可靠性。
    """
    def __init__(self, environment):
        super().__init__(
            name="Logical Verifier",
            system_message=(
                "你是一个乐于解答各种问题的助手，验证推理过程的逻辑一致性和结果的可靠性。"
                "检查每个结论的正确性，并找出任何潜在的错误或差距。"
            ),
            environment=environment,
        )


# 定义 Gradio 应用逻辑
def cot_process(api_key, model, top_p, temperature, max_tokens, question):
    # 初始化环境和智能体
    cot_environment = CoTEnvironment(
        api_key=api_key,
        model=model,
        top_p=top_p,
        temperature=temperature,
        max_tokens=max_tokens
    )
    analyzer = ProblemAnalyzer(environment=cot_environment)
    reasoner = StepReasoner(environment=cot_environment)
    verifier = Verifier(environment=cot_environment)

    # 智能体协作
    steps = analyzer.think(question)
    reasoning = reasoner.think(steps)
    verification = verifier.think(reasoning)

    return steps, reasoning, verification

# Gradio 界面定义
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Chain-of-Thought (CoT) Interactive Process")

        # API Key 输入
        api_key_input = gr.Textbox(label="API Key", placeholder="Enter your ZhipuAI API Key", type="password")

        # 模型选择
        model_selection = gr.Dropdown(
            label="Select Model",
            choices=["glm-4-plus", "glm-4-0520", "glm-4-air", "glm-4-flash", "glm-4"],  # 可根据支持的模型扩展glm-4-airx、glm-4-long 、 glm-4-flashx 、 glm-4-flash
            value="glm-4-plus"  # 默认选项
        )

        # 参数设置 (top_p, temperature, max_tokens)
        with gr.Row():
            top_p_input = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.1, step=0.01)
            temperature_input = gr.Slider(label="temperature", minimum=0.0, maximum=1.0, value=0.1, step=0.01)
            max_tokens_input = gr.Number(label="max_tokens", value=600, precision=0)

        # 用户输入问题
        question_input = gr.Textbox(label="Question", placeholder="Enter a question to analyze.")

        # 输出区域
        analysis_output = gr.Textbox(label="Analysis", interactive=False)
        reasoning_output = gr.Textbox(label="Reasoning", interactive=False)
        verification_output = gr.Textbox(label="Verification", interactive=False)

        # 按钮触发
        submit_button = gr.Button("Submit")

        # 动作绑定
        submit_button.click(
            fn=cot_process,
            inputs=[
                api_key_input,
                model_selection,
                top_p_input,
                temperature_input,
                max_tokens_input,
                question_input
            ],
            outputs=[analysis_output, reasoning_output, verification_output]
        )

    return demo

# 启动应用
if __name__ == "__main__":
    gradio_interface().launch()
