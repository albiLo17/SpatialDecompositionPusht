#!/bin/bash
# Test script for new components: KNN aggregation and FiLM conditioning
# This script runs a quick training test to verify both components work

echo "=========================================="
echo "Testing KNN aggregation and FiLM conditioning"
echo "=========================================="

# Test 1: KNN aggregation with median (baseline)
echo ""
echo "Test 1: Training with median aggregation (baseline)..."
python SD_pusht/scripts/train_local_flow_2d.py \
    --dataset-path datasets/pusht_cchi_v7_replay.zarr.zip \
    --max-demos 10 \
    --batch-size 64 \
    --epochs 2 \
    --eval-every 0 \
    --seed 0 \
    --use-position-decoder \
    --position-decoder-num-particles 8 \
    --position-decoder-particles-aggregation median \
    --position-loss-coeff 1.0 \
    --contact-threshold 0.1 \
    --output-dir log/test_median

# Test 2: KNN aggregation
echo ""
echo "Test 2: Training with KNN aggregation..."
python SD_pusht/scripts/train_local_flow_2d.py \
    --dataset-path datasets/pusht_cchi_v7_replay.zarr.zip \
    --max-demos 10 \
    --batch-size 64 \
    --epochs 2 \
    --eval-every 0 \
    --seed 0 \
    --use-position-decoder \
    --position-decoder-num-particles 8 \
    --position-decoder-particles-aggregation knn \
    --position-loss-coeff 1.0 \
    --contact-threshold 0.1 \
    --output-dir log/test_knn

# Test 3: FiLM conditioning (with concatenation baseline)
echo ""
echo "Test 3: Training with concatenation (baseline, no FiLM)..."
python SD_pusht/scripts/train_local_flow_2d.py \
    --dataset-path datasets/pusht_cchi_v7_replay.zarr.zip \
    --max-demos 10 \
    --batch-size 64 \
    --epochs 2 \
    --eval-every 0 \
    --seed 0 \
    --use-position-decoder \
    --position-decoder-particles-aggregation median \
    --position-loss-coeff 1.0 \
    --contact-threshold 0.1 \
    --output-dir log/test_concat

# Test 4: FiLM conditioning
echo ""
echo "Test 4: Training with FiLM conditioning..."
python SD_pusht/scripts/train_local_flow_2d.py \
    --dataset-path datasets/pusht_cchi_v7_replay.zarr.zip \
    --max-demos 10 \
    --batch-size 64 \
    --epochs 2 \
    --eval-every 0 \
    --seed 0 \
    --use-position-decoder \
    --position-decoder-particles-aggregation median \
    --use-film-conditioning \
    --film-hidden-dim 64 \
    --position-loss-coeff 1.0 \
    --contact-threshold 0.1 \
    --output-dir log/test_film

# Test 5: Both KNN and FiLM together
echo ""
echo "Test 5: Training with BOTH KNN aggregation AND FiLM conditioning..."
python SD_pusht/scripts/train_local_flow_2d.py \
    --dataset-path datasets/pusht_cchi_v7_replay.zarr.zip \
    --max-demos 10 \
    --batch-size 64 \
    --epochs 2 \
    --eval-every 0 \
    --seed 0 \
    --use-position-decoder \
    --position-decoder-num-particles 8 \
    --position-decoder-particles-aggregation knn \
    --use-film-conditioning \
    --film-hidden-dim 64 \
    --position-loss-coeff 1.0 \
    --contact-threshold 0.1 \
    --output-dir log/test_knn_film

echo ""
echo "=========================================="
echo "All tests completed!"
echo "Check the output directories:"
echo "  - log/test_median/    (median aggregation)"
echo "  - log/test_knn/       (KNN aggregation)"
echo "  - log/test_concat/    (concatenation, no FiLM)"
echo "  - log/test_film/      (FiLM conditioning)"
echo "  - log/test_knn_film/  (both KNN and FiLM)"
echo "=========================================="
