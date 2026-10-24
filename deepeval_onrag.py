import json
import os
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, RetrievedContextData, SingleTurnParams
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent / ".env")

GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqModel(DeepEvalBaseLLM):
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        super().__init__(model=GROQ_MODEL)

    def _load_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=GROQ_BASE_URL)

    def load_model(self) -> DeepEvalBaseLLM:
        return self

    def get_model_name(self) -> str:
        return GROQ_MODEL

    @staticmethod
    def _messages(prompt: str, schema: type[BaseModel] | None) -> list[dict[str, str]]:
        if schema is not None:
            prompt += "\nReturn valid JSON matching this schema:\n" + json.dumps(
                schema.model_json_schema()
            )
        return [{"role": "user", "content": prompt}]

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> str:
        request: dict[str, Any] = {
            "model": GROQ_MODEL,
            "messages": self._messages(prompt, schema),
            "temperature": 0.2,
        }
        if schema is not None:
            request["response_format"] = {"type": "json_object"}
        client = self._load_client()
        response = client.chat.completions.create(**request)
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(f"Groq returned an empty response for {GROQ_MODEL}.")
        return content

    async def a_generate(
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> str:
        request: dict[str, Any] = {
            "model": GROQ_MODEL,
            "messages": self._messages(prompt, schema),
            "temperature": 0.2,
        }
        if schema is not None:
            request["response_format"] = {"type": "json_object"}
        async with AsyncOpenAI(api_key=self.api_key, base_url=GROQ_BASE_URL) as client:
            response = await client.chat.completions.create(**request)
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(f"Groq returned an empty response for {GROQ_MODEL}.")
        return content


groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add your Groq API key to the project .env file."
    )
groq_model = GroqModel(api_key=groq_api_key)

with open("eval_result1.json", "r", encoding="utf-8") as f:
    results = json.load(f)

    question_set = []

    for result in results:
        context = result.get("context_used")

        if isinstance(context, str):
            context_used: list[str | RetrievedContextData] = (
                [context] if context else []
            )
        elif isinstance(context, list):
            context_used = [item for item in context if isinstance(item, str)]
        else:
            context_used = []

        testcase = LLMTestCase(
            input=result.get("question"),
            actual_output=result.get("actual_answer"),
            expected_output=result.get("expected_answer"),
            retrieval_context=context_used,
        )

        question_set.append(testcase)

    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7, model=groq_model, async_mode=False
    )
    correctness = GEval(
        name="correctness",
        criteria=(
            "A check whether the actual answer is based on the contextual answer provided in the PDF and is relevant to the question asked. "
        ),
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=groq_model,
        async_mode=False,
    )
    faithfulness = FaithfulnessMetric(threshold=0.7, model=groq_model, async_mode=False)
    contextual_relevancy = ContextualRelevancyMetric(
        threshold=0.7, model=groq_model, async_mode=False
    )

    evaluate_model = evaluate(
        metrics=[
            answer_relevancy_metric,
            correctness,
            faithfulness,
            contextual_relevancy,
        ],
        test_cases=question_set,
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(use_cache=False, write_cache=False),
    )

    evaluated_results = []

    for i, (original_result, test_result) in enumerate(
        zip(results, evaluate_model.test_results), start=1
    ):
        answer_relevancy_score = None
        correctness_score = None
        faithfulness_score = None
        contextual_relevancy_score = None

        print(f"Evaluation run {i}:")

        for metric in test_result.metrics_data or []:
            metric_name = "".join(
                character for character in metric.name.lower() if character.isalnum()
            )

            if metric_name in {"answerrelevancy", "answerrelevancymetric"}:
                answer_relevancy_score = metric.score
            elif metric_name in {"correctness", "correctnessgeval"}:
                correctness_score = metric.score
            elif metric_name == "faithfulness":
                faithfulness_score = metric.score
            elif metric_name in {"contextualrelevancy", "contextualrelevancymetric"}:
                contextual_relevancy_score = metric.score

        evaluated_result = {
            **original_result,
            "answer_relevancy_score": answer_relevancy_score,
            "correctness_score": correctness_score,
            "faithfulness_score": faithfulness_score,
            "contextual_relevancy_score": contextual_relevancy_score,
        }

        evaluated_results.append(evaluated_result)


with open("evaluated_results.json", "w", encoding="utf-8") as f:
    for res in evaluated_results:
        f.write(json.dumps(res) + "\n")
