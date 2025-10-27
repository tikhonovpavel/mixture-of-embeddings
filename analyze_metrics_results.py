import pandas as pd
import json
import fire
import os


def analyze_results(
    input_file: str,
    num_samples_to_show: int = 5,
    show_samples: bool = False
):
    """
    Analyzes the .jsonl file from the evaluation script, calculates average metrics,
    and prints a summary report along with some generation samples.

    Args:
        input_file (str): Path to the input .jsonl file containing evaluation results.
        num_samples_to_show (int): The number of sample generations to display.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    # Load the results from the JSONL file
    results = []
    with open(input_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    if not results:
        print("Error: The results file is empty.")
        return

    # Convert to a pandas DataFrame for easier analysis
    df = pd.DataFrame(results)

    # --- Metrics Calculation ---
    # We will now have 4 sets of metrics to compare:
    # 1. Original Model on texts A
    # 2. Dual-Stream A on texts A
    # 3. Original Model on texts B
    # 4. Dual-Stream B on texts B
    metrics = {
        'dual_stream_A': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': [], 'bert_score_f1': []},
        'original_A': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': [], 'bert_score_f1': []},
        'dual_stream_B': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': [], 'bert_score_f1': []},
        'original_B': {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': [], 'bert_score_f1': []},
    }
    
    avg_original_loss_A = df['original_model_loss_A'].mean()
    avg_original_loss_B = df['original_model_loss_B'].mean()

    for _, row in df.iterrows():
        # Dual Stream A vs Original A (on Reference A)
        metrics['dual_stream_A']['rouge1'].append(row['metrics_dual_stream_A']['rouge']['rouge1'])
        metrics['dual_stream_A']['rouge2'].append(row['metrics_dual_stream_A']['rouge']['rouge2'])
        metrics['dual_stream_A']['rougeL'].append(row['metrics_dual_stream_A']['rouge']['rougeL'])
        metrics['dual_stream_A']['bleu'].append(row['metrics_dual_stream_A']['bleu']['bleu'])
        metrics['dual_stream_A']['bert_score_f1'].append(row['metrics_dual_stream_A']['bert_score_f1'])
        
        metrics['original_A']['rouge1'].append(row['metrics_original_A']['rouge']['rouge1'])
        metrics['original_A']['rouge2'].append(row['metrics_original_A']['rouge']['rouge2'])
        metrics['original_A']['rougeL'].append(row['metrics_original_A']['rouge']['rougeL'])
        metrics['original_A']['bleu'].append(row['metrics_original_A']['bleu']['bleu'])
        metrics['original_A']['bert_score_f1'].append(row['metrics_original_A']['bert_score_f1'])

        # Dual Stream B vs Original B (on Reference B)
        metrics['dual_stream_B']['rouge1'].append(row['metrics_dual_stream_B']['rouge']['rouge1'])
        metrics['dual_stream_B']['rouge2'].append(row['metrics_dual_stream_B']['rouge']['rouge2'])
        metrics['dual_stream_B']['rougeL'].append(row['metrics_dual_stream_B']['rouge']['rougeL'])
        metrics['dual_stream_B']['bleu'].append(row['metrics_dual_stream_B']['bleu']['bleu'])
        metrics['dual_stream_B']['bert_score_f1'].append(row['metrics_dual_stream_B']['bert_score_f1'])

        metrics['original_B']['rouge1'].append(row['metrics_original_B']['rouge']['rouge1'])
        metrics['original_B']['rouge2'].append(row['metrics_original_B']['rouge']['rouge2'])
        metrics['original_B']['rougeL'].append(row['metrics_original_B']['rouge']['rougeL'])
        metrics['original_B']['bleu'].append(row['metrics_original_B']['bleu']['bleu'])
        metrics['original_B']['bert_score_f1'].append(row['metrics_original_B']['bert_score_f1'])

    # Calculate averages
    avg_metrics = {model: {metric: pd.Series(values).mean() for metric, values in data.items()} for model, data in metrics.items()}

    # --- Display Results ---
    print("\n" + "="*80)
    print(" " * 25 + "Evaluation Metrics Summary")
    print("="*80)
    print(f"Analyzed {len(df)} sample pairs from '{input_file}'\n")

    summary_df = pd.DataFrame(avg_metrics).T
    summary_df.index.name = "Model"
    summary_df = summary_df.rename(columns={
        'rouge1': 'ROUGE-1', 'rouge2': 'ROUGE-2', 'rougeL': 'ROUGE-L', 'bleu': 'BLEU', 'bert_score_f1': 'BERTScore-F1'
    })
    
    print(summary_df.to_string(float_format="%.4f"))
    print("\n" + "-"*80)
    print(f"Average Original Model Loss (on texts A): {avg_original_loss_A:.4f}")
    print(f"Average Original Model Loss (on texts B): {avg_original_loss_B:.4f}")
    print("-" * 80)


    # # --- Display Samples ---
    if show_samples:
        print("\n" + "="*80)
        print(" " * 28 + "Generation Samples")
        print("="*80)
        
        samples_to_show = df.head(num_samples_to_show)
        for index, row in samples_to_show.iterrows():
            print(f"\n--- Sample Pair #{row['sample_id']} ---")
            print(f"\n[STREAM A]")
            print(f"  Prompt:     '{row['prompt_A']}'")
            print(f"  Reference:  '{row['reference_A']}'")
            print(f"  Original:   '{row['generated_original_A']}'")
            print(f"  DualStream: '{row['generated_dual_stream_A']}'")

            print(f"\n[STREAM B]")
            print(f"  Prompt:     '{row['prompt_B']}'")
            print(f"  Reference:  '{row['reference_B']}'")
            print(f"  Original:   '{row['generated_original_B']}'")
            print(f"  DualStream: '{row['generated_dual_stream_B']}'")
            print("-" * 80)
            
if __name__ == '__main__':
    fire.Fire(analyze_results)
