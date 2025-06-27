import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from main import dspy_generate_pipeline

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def gpt_grade_metric(query, expected, response):
    """GPT-based grading metric for more nuanced evaluation."""
    try:
        grading_prompt = f"""
        You are an evaluation assistant grading the quality of a response.

        Query: "{query}"
        Expected Answer: "{expected}"
        Model Response: "{response}"

        Rate the response on a scale of 0-1, where:
        - 1: Perfect answer, fully addresses the query
        - .7-.9: Good answer, mostly accurate and helpful
        - .4-.6: Partial answer, some relevant information
        - 0-.3: Poor answer, incorrect or unhelpful

        Only respond with a number between 0-1.
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a fair and consistent grader."},
                {"role": "user", "content": grading_prompt},
            ],
            temperature=0,
            max_tokens=5,
        )
        content = response.choices[0].message.content
        if content is None:
            return 0

        # Extract numeric score
        score_text = content.strip()
        try:
            return float(score_text)
        except ValueError:
            return 0

    except Exception as e:
        print(f"⚠️  GPT grading failed: {e}")
        return 0


if __name__ == "__main__":
    """Main evaluation function."""
    print("🚀 Starting DSPy Evaluation")
    print("=" * 60)

    # Initialize pipeline
    print("🔧 Initializing pipeline...")
    pipeline = dspy_generate_pipeline()

    # Load devset and trainset from examples
    with open("devset.json") as f:
        devset = json.load(f)

    scores = []
    for item in devset:
        response = pipeline(query=item["query"])
        scores.append(gpt_grade_metric(item["query"], item["expected"], response.answer))

    print(f"Average score: {sum(scores) / len(scores)}")
