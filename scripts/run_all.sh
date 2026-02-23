#!/bin/bash
set -e

# PlantVillage Transfer Learning Benchmark (18 runs)
# 双 GPU 并行: bash scripts/run_pytorch.sh & bash scripts/run_keras.sh & wait

echo "=== keras plantvillage resnet50 fold0 ==="
python ass2_code.py train --framework keras --model resnet50 --fold 0

echo "=== keras plantvillage resnet50 fold1 ==="
python ass2_code.py train --framework keras --model resnet50 --fold 1

echo "=== keras plantvillage resnet50 fold2 ==="
python ass2_code.py train --framework keras --model resnet50 --fold 2

echo "=== keras plantvillage vgg16 fold0 ==="
python ass2_code.py train --framework keras --model vgg16 --fold 0

echo "=== keras plantvillage vgg16 fold1 ==="
python ass2_code.py train --framework keras --model vgg16 --fold 1

echo "=== keras plantvillage vgg16 fold2 ==="
python ass2_code.py train --framework keras --model vgg16 --fold 2

echo "=== keras plantvillage mobilenetv2 fold0 ==="
python ass2_code.py train --framework keras --model mobilenetv2 --fold 0

echo "=== keras plantvillage mobilenetv2 fold1 ==="
python ass2_code.py train --framework keras --model mobilenetv2 --fold 1

echo "=== keras plantvillage mobilenetv2 fold2 ==="
python ass2_code.py train --framework keras --model mobilenetv2 --fold 2

echo "=== pytorch plantvillage resnet50 fold0 ==="
python ass2_code.py train --framework pytorch --model resnet50 --fold 0

echo "=== pytorch plantvillage resnet50 fold1 ==="
python ass2_code.py train --framework pytorch --model resnet50 --fold 1

echo "=== pytorch plantvillage resnet50 fold2 ==="
python ass2_code.py train --framework pytorch --model resnet50 --fold 2

echo "=== pytorch plantvillage vgg16 fold0 ==="
python ass2_code.py train --framework pytorch --model vgg16 --fold 0

echo "=== pytorch plantvillage vgg16 fold1 ==="
python ass2_code.py train --framework pytorch --model vgg16 --fold 1

echo "=== pytorch plantvillage vgg16 fold2 ==="
python ass2_code.py train --framework pytorch --model vgg16 --fold 2

echo "=== pytorch plantvillage mobilenetv2 fold0 ==="
python ass2_code.py train --framework pytorch --model mobilenetv2 --fold 0

echo "=== pytorch plantvillage mobilenetv2 fold1 ==="
python ass2_code.py train --framework pytorch --model mobilenetv2 --fold 1

echo "=== pytorch plantvillage mobilenetv2 fold2 ==="
python ass2_code.py train --framework pytorch --model mobilenetv2 --fold 2

echo "All runs complete."

# Auto-aggregate
python ass2_code.py aggregate
