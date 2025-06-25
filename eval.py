import json
import os

import dspy
from dotenv import load_dotenv
from openai import OpenAI

from main import dspy_generate_pipeline

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def exact_match_metric(example, prediction, trace=None):
    """Simple exact match metric."""
    # Handle both string and prediction object responses
    if hasattr(prediction, 'answer'):
        model_response = prediction.answer
    else:
        model_response = str(prediction)
    return int(example.expected.lower() == model_response.lower())


def gpt_grade_metric(example, prediction, trace=None):
    """GPT-based grading metric for more nuanced evaluation."""
    try:
        query, expected = example.query, example.expected
        # Handle both string and prediction object responses
        if hasattr(prediction, 'answer'):
            model_response = prediction.answer
        else:
            model_response = str(prediction)
            
        grading_prompt = f"""
        You are an evaluation assistant grading the quality of a response.

        Query: "{query}"
        Expected Answer: "{expected}"
        Model Response: "{model_response}"

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


def main():
    """Main evaluation function."""
    print("🚀 Starting DSPy Evaluation")
    print("=" * 60)

    # Initialize pipeline
    print("🔧 Initializing pipeline...")
    pipeline = dspy_generate_pipeline()

    # Load devset and trainset from examples
    with open("devset.json") as f:
        data = json.load(f)

    devset = [dspy.Example(**ex).with_inputs("query", "context") for ex in data[3:]]
    trainset = [dspy.Example(**ex).with_inputs("query", "context") for ex in data[:3]]

    # Test exact match evaluation
    exact_evaluator = dspy.Evaluate(
        devset=devset, 
        metric=exact_match_metric,
        num_threads=1  
    )
    exact_results = exact_evaluator(pipeline)
    print(f"📊 Exact Match Results: {exact_results}")

    # Test GPT-based evaluation
    gpt_evaluator = dspy.Evaluate(
        devset=devset, 
        metric=gpt_grade_metric,
        num_threads=1
    )
    gpt_results = gpt_evaluator(pipeline)
    print(f"📊 GPT Results: {gpt_results}")

    # Test the pipeline with a sample query
    print("\n🧪 Testing pipeline with sample query...")
    test_response = pipeline(query="What's your return policy?")
    print(f"Original pipeline response: {test_response.answer}")

    print("\n🚀 Starting BootstrapFewShot optimization...")
    try:
        optimizer = dspy.BootstrapFewShot(
            metric=gpt_grade_metric,
            max_bootstrapped_demos=2,  # Reduce for faster optimization
            max_labeled_demos=2
        )
        
        optimized_pipeline = optimizer.compile(
            student=pipeline,
            trainset=trainset
        )
        
        print("✅ Optimization completed successfully!")
        
        print("\n📊 Evaluating Optimized Pipeline...")
        
        exact_results_opt = exact_evaluator(optimized_pipeline)
        print(f"📊 Optimized - Exact Match Results: {exact_results_opt}")
        
        gpt_results_opt = gpt_evaluator(optimized_pipeline)
        print(f"📊 Optimized - GPT Results: {gpt_results_opt}")
        
        # Compare results
        print("\n📈 Comparison:")
        
        # Extract float values from results (DSPy may return tuples or floats)
        exact_baseline = exact_results if isinstance(exact_results, (int, float)) else exact_results[0]
        exact_optimized = exact_results_opt if isinstance(exact_results_opt, (int, float)) else exact_results_opt[0]
        gpt_baseline = gpt_results if isinstance(gpt_results, (int, float)) else gpt_results[0]
        gpt_optimized = gpt_results_opt if isinstance(gpt_results_opt, (int, float)) else gpt_results_opt[0]
        
        print(f"Exact Match: {exact_baseline} -> {exact_optimized} (Δ: {exact_optimized - exact_baseline:+.1f})")
        print(f"GPT Score: {gpt_baseline} -> {gpt_optimized} (Δ: {gpt_optimized - gpt_baseline:+.1f})")
        
        # Test optimized pipeline
        print("\n🧪 Testing optimized pipeline...")
        optimized_response = optimized_pipeline(query="What's your return policy?")
        print(f"Optimized response: {optimized_response.answer}")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        print("This may be due to remaining serialization issues. Check the error details above.")


if __name__ == "__main__":
    main()
