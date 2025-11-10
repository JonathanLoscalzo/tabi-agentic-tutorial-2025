from tabi_llm_app import QAEngine
from langchain_ollama import ChatOllama


class ChatQAEngine:

    def __init__(self, qa_engine: QAEngine):
        self.qa_engine = qa_engine

        self._chat = ChatOllama(
            model=self.qa_engine.llm_model,
            temperature=self.qa_engine.temperature,
            options={
                "top_p": 0.9,
                "top_k": 40,
                "temperature": self.qa_engine.temperature,
            },
        )

    def start_chat(self, question: str):
        prompt, context, sources = self.qa_engine.generate_prompt(question, False)
        return prompt, context, sources

    def chat(self, messages: list[dict]):
        response = self._chat.invoke(messages)
        return response

    async def achat(self, messages: list[dict]):
        response = await self._chat.ainvoke(messages)
        return response
