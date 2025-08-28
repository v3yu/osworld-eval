#!/bin/bash

cd /lustre/scratch/users/guangyi.liu/agent/GUI-Agent
SCRIPT_PATH="/lustre/scratch/users/guangyi.liu/agent/GUI-Agent/run.py"
MODEL="qwen2.5-vl"

# Array of domains to run
# DOMAINS=("normal" "multi567" "compare" "multipro" "shopping" "wikipedia")
DOMAINS=("shopping" "normal")

# Print header
echo "====================================================="
echo "Starting runs with model: $MODEL"
echo "====================================================="

# Loop through each domain and run the script
for domain in "${DOMAINS[@]}"; do
    echo "Starting run with domain: $domain"
    # Set max_steps based on domain
    if [[ "$domain" == "shopping" || "$domain" == "wikipedia" ]]; then
        MAX_STEPS=15
    elif [[ "$domain" == "normal" || "$domain" == "compare" ]]; then
        MAX_STEPS=30
    elif [[ "$domain" == "multi567" || "$domain" == "multipro" ]]; then
        MAX_STEPS=50
    else
        MAX_STEPS=30  # Default value
    fi
    
    echo "Using max_steps: $MAX_STEPS for domain: $domain"
    
    # Run the script with the specified domain and model, redirecting output to log file
    python "$SCRIPT_PATH" --domain "$domain" --model "$MODEL" --max_steps "$MAX_STEPS"
    
    # Check if the run was successful
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Run completed successfully for domain: $domain"
    else
        echo "✗ Run failed for domain: $domain"
    fi
    
    echo "-----------------------------------------------------"
    
    # Optional: add a short pause between runs
    sleep 2
done

echo "====================================================="
echo "All runs completed at $(date +"%Y-%m-%d %H:%M:%S")"
echo "====================================================="