# -*- coding: utf-8 -*-
"""
Script to ask a single question to an MwsAgent instance.
"""
import os

os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

import sys
import codecs

sys.stdin = codecs.getreader('utf-8')(sys.stdin.buffer)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
import json
import time

from crewai import Crew, Task

from agents.semantic_search import SemanticSearch

from agents import (
    KnowledgeExpertAgent, IntentRecognizerAgent, EmotionAnalyzerAgent,
    ActionAdvisorAgent, SummaryGeneratorAgent, QualityControllerAgent
)

from agents import ProxyAgent

import logging

API_KEY = "sk-KNo006G2a48UVE3IxFlQEQ"


def ask_all_agents(question):
    """
    Instantiates all agents, defines tasks, and runs them sequentially using CrewAI.
    """

    shared_context = {}

    intent_agent = ProxyAgent(IntentRecognizerAgent(
        role="распознаватель намерений",
        goal="Определи намерение клиента из текста запроса.",
        backstory="Ты эксперт по классификации клиентских запросов в контакт-центре МТС."
    ), shared_context)

    emotion_agent = ProxyAgent(EmotionAnalyzerAgent(
        role="аналитик эмоций",
        goal="Оцени эмоциональное состояние клиента.",
        backstory="Ты специалист по анализу тональности клиентских сообщений."
    ), shared_context)

    knowledge_agent = ProxyAgent(KnowledgeExpertAgent(
        role="эксперт базы знаний",
        goal="Дай эталонный ответ для подсказки.",
        backstory="Ты работаешь с базой знаний МТС.",
        semantic_search_engine=semantic_search_engine
    ), shared_context)

    action_agent = ProxyAgent(ActionAdvisorAgent(
        role="советник по действиям",
        goal="Предложи действия для оператора.",
        backstory="Ты обучен помогать операторам решать проблемы клиентов."
    ), shared_context)

    summary_agent = ProxyAgent(SummaryGeneratorAgent(
        role="генератор резюме",
        goal="Сформируй резюме обращения для CRM.",
        backstory="Ты готовишь отчеты для CRM."
    ), shared_context)

    qa_agent = ProxyAgent(QualityControllerAgent(
        role="контролер качества",
        goal="Проверь соответствие ответа стандартам МТС и Кодексу.",
        backstory="Ты отвечаешь за качество коммуникации в поддержке МТС."
    ), shared_context)

    tasks = [
        Task(
            description=f"Клиент обратился в поддержку.\nПоследний запрос клиента: \"{question}\"\nОцени эмоциональное состояние клиента.",
            expected_output="Эмоциональное состояние клиента.",
            agent=emotion_agent
        ),
        Task(
            description=f"Клиент обратился в поддержку.\nПоследний запрос клиента: \"{question}\"\nОпредели намерение клиента из текста запроса.",
            expected_output="Намерение клиента.",
            agent=intent_agent
        ),
        Task(
            description=f"Клиент обратился в поддержку.\nПоследний запрос клиента: \"{question}\"\nДай эталонный ответ для подсказки.",
            expected_output="Эталонный ответ.",
            agent=knowledge_agent
        ),
        Task(
            description=f"Клиент обратился в поддержку.\nПоследний запрос клиента: \"{question}\"\nПредложи, что должен сделать оператор.",
            expected_output="Рекомендации для оператора.",
            agent=action_agent
        ),
        Task(
            description=f"Клиент обратился в поддержку.\nПоследний запрос клиента: \"{question}\"\nСформируй краткое резюме обращения для CRM.",
            expected_output="Краткое резюме обращения.",
            agent=summary_agent
        ),
        Task(
            description=f"Клиент обратился в поддержку.\nПоследний запрос клиента: \"{question}\"\nПроверь, соответствует ли ответ стандартам общения.",
            expected_output="Оценка соответствия ответа стандартам.",
            agent=qa_agent
        )
    ]

    crew = Crew(
        agents=[emotion_agent, intent_agent, knowledge_agent, action_agent, summary_agent, qa_agent],
        tasks=tasks,
        process="sequential",
        verbose=False
    )

    total_start_time = time.time()
    try:

        crew.kickoff()

        total_end_time = time.time()
        total_time = total_end_time - total_start_time

        results_data = {
            "question": question,
            "intent": shared_context.get('intent', 'N/A'),
            "emotion": shared_context.get('emotion', 'N/A'),
            "reference_answer": shared_context.get('reference_answer', 'N/A'),
            "action": shared_context.get('action', 'N/A'),
            "summary": shared_context.get('summary', 'N/A'),
            "qa": shared_context.get('qa', 'N/A'),
            "discount_offered": shared_context.get('discount_offered', False),
            "discount_details": shared_context.get('discount_details', 'N/A'),
            "alternative_service": shared_context.get('alternative_service', 'N/A'),
            "total_processing_time": total_time
        }

        try:

            json_output = json.dumps(results_data, ensure_ascii=True)

            print(json_output)
            sys.stdout.flush()

        except Exception as e:
            logging.error(f"Ошибка при сериализации или отправке JSON в stdout: {e}")

    except Exception as e:
        logging.error(f"Ошибка при выполнении Crew: {e}")
        print(f"Ошибка при выполнении Crew: {e}")


if __name__ == "__main__":

    log_file = "app.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8',
        stream=sys.stderr,
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    try:
        semantic_search_engine = SemanticSearch.with_api_provider(api_key=API_KEY)
        logging.info("Загрузка индекса SemanticSearch...")
        semantic_search_engine.load_index(
            df_path="../data/faiss_index/corpus.csv",
            index_path="../data/faiss_index/faiss.index",
            text_column="name"
        )
        logging.info("SemanticSearch инициализирован и индекс загружен.")
    except Exception as e:
        logging.error("Ошибка при инициализации SemanticSearch.")
        pass

    question = input()
    text = None

    try:
        outer_data = json.loads(question)
        print(f"Распаршены внешние данные: {outer_data}")

        question_json_string = outer_data.get("text")
        text = question_json_string

    except json.JSONDecodeError as e:
        print(f"Ошибка парсинга JSON: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

    logging.info(text)
    ask_all_agents(text)
