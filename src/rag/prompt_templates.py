"""Prompt 模板模块 - 定义系统提示词和对话模板"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import Optional, List


class PromptManager:
    """
    Prompt 管理器 - 管理不同场景的提示词模板
    
    特性：
    - 支持多种角色定义（运维专家、技术支持等）
    - 可配置的上下文注入
    - 支持历史对话摘要
    """
    
    def __init__(self):
        """初始化 Prompt 管理器"""
        self.templates = {}
        self._register_default_templates()
    
    def _register_default_templates(self):
        """注册默认的提示词模板"""
        self.templates["ops_expert"] = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是一名资深运维专家，拥有丰富的企业级运维经验。
                
                你的任务是根据提供的知识库内容，准确回答用户的运维相关问题。
                
                规则：
                1. 严格基于知识库内容回答，不要编造信息
                2. 如果知识库中没有相关信息，请明确说明"知识库中未找到相关信息"
                3. 回答要简洁明了，直接给出解决方案
                4. 如果问题涉及多个步骤，请分点说明
                5. 对于命令行操作，请使用代码块格式输出
                """
            ),
            ("human", "{context}\n\n用户问题：{question}")
        ])
        
        self.templates["detailed"] = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是一名专业的技术文档解释员。
                
                请详细解释知识库中的内容，帮助用户理解复杂的技术概念。
                
                要求：
                1. 使用通俗易懂的语言
                2. 提供具体的示例
                3. 解释技术术语
                4. 如果有多种解决方案，请逐一说明
                """
            ),
            ("human", "参考文档：\n{context}\n\n请解释：{question}")
        ])
        
        self.templates["troubleshooting"] = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是一名故障排查专家。
                
                根据知识库中的信息，帮助用户诊断和解决问题。
                
                排查步骤：
                1. 分析问题现象
                2. 列出可能的原因
                3. 提供排查方法
                4. 给出解决方案
                5. 提供预防措施
                """
            ),
            ("human", "知识库：\n{context}\n\n故障描述：{question}\n\n请提供排查方案：")
        ])
    
    def get_template(self, template_name: str = "ops_expert") -> ChatPromptTemplate:
        """
        获取指定名称的提示词模板
        
        Args:
            template_name: 模板名称
            
        Returns:
            ChatPromptTemplate 实例
            
        Raises:
            ValueError: 如果模板不存在
        """
        if template_name not in self.templates:
            raise ValueError(f"模板 {template_name} 不存在")
        
        return self.templates[template_name]
    
    def create_custom_template(
        self,
        template_name: str,
        system_prompt: str,
        human_prompt: str = "{context}\n\n用户问题：{question}"
    ):
        """
        创建自定义提示词模板
        
        Args:
            template_name: 模板名称
            system_prompt: 系统提示词
            human_prompt: 用户提示词模板
        """
        self.templates[template_name] = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def build_prompt(
        self,
        question: str,
        context: str,
        template_name: str = "ops_expert",
        history_summary: Optional[str] = None
    ) -> str:
        """
        构建完整的 Prompt
        
        Args:
            question: 用户问题
            context: 检索到的上下文内容
            template_name: 使用的模板名称
            history_summary: 历史对话摘要（可选）
            
        Returns:
            完整的 Prompt 字符串
        """
        template = self.get_template(template_name)
        
        inputs = {
            "question": question,
            "context": context
        }
        
        if history_summary:
            inputs["history_summary"] = history_summary
        
        return template.format(**inputs)
    
    def truncate_context(self, context: str, max_length: int = 3000) -> str:
        """
        截断上下文以适应模型窗口限制
        
        Args:
            context: 原始上下文内容
            max_length: 最大长度
            
        Returns:
            截断后的上下文
        """
        if len(context) <= max_length:
            return context
        
        truncated = context[:max_length]
        last_newline = truncated.rfind("\n")
        
        if last_newline != -1:
            truncated = truncated[:last_newline]
        
        return truncated + "\n\n...（内容已截断）"