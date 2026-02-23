#!/bin/bash
set -e

# PlantVillage Transfer Learning — pytorch (9 runs)

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

echo "pytorch runs complete."
