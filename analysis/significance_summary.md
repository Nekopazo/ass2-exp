# Significance Summary

| Dataset | Model | Metric | p-value | Significant (a=0.05) | Cohen's d | Effect Size | Cliff's d | D (PT-Keras) |
|---------|-------|--------|---------|---------------------|-----------|-------------|-----------|-------------|
| plantvillage | resnet50 | test_accuracy | 0.021272 | Yes | 3.8952 | Large | 1.0000 | 0.003008 |
| plantvillage | resnet50 | test_macro_f1 | 0.010721 | Yes | 5.5310 | Large | 1.0000 | 0.003298 |
| plantvillage | resnet50 | images_per_sec_avg | 0.289983 | No | 0.8233 | Large | 1.0000 | 26.933333 |
| plantvillage | vgg16 | test_accuracy | 0.030236 | Yes | 3.2445 | Large | 1.0000 | 0.004573 |
| plantvillage | vgg16 | test_macro_f1 | 0.032269 | Yes | 3.1357 | Large | 1.0000 | 0.005941 |
| plantvillage | vgg16 | images_per_sec_avg | 0.007533 | Yes | 6.6144 | Large | 1.0000 | 3.500000 |
| plantvillage | mobilenetv2 | test_accuracy | 0.019367 | Yes | 4.0881 | Large | 1.0000 | 0.425590 |
| plantvillage | mobilenetv2 | test_macro_f1 | 0.012748 | Yes | 5.0644 | Large | 1.0000 | 0.527480 |
| plantvillage | mobilenetv2 | images_per_sec_avg | 0.029179 | Yes | -3.3055 | Large | -0.5556 | -31.033333 |
| cifar10 | resnet50 | test_accuracy | 0.276638 | No | -0.8554 | Large | -0.3333 | -0.023861 |
| cifar10 | resnet50 | test_macro_f1 | 0.276148 | No | -0.8566 | Large | -0.3333 | -0.023972 |
| cifar10 | resnet50 | images_per_sec_avg | 0.012476 | Yes | -5.1204 | Large | -1.0000 | -99.633333 |
| cifar10 | vgg16 | test_accuracy | 0.009484 | Yes | 5.8864 | Large | 1.0000 | 0.019389 |
| cifar10 | vgg16 | test_macro_f1 | 0.009547 | Yes | 5.8664 | Large | 1.0000 | 0.019514 |
| cifar10 | vgg16 | images_per_sec_avg | 0.005187 | Yes | -7.9850 | Large | -1.0000 | -71.533333 |
| cifar10 | mobilenetv2 | test_accuracy | 0.008418 | Yes | 6.2527 | Large | 1.0000 | 0.557778 |
| cifar10 | mobilenetv2 | test_macro_f1 | 0.010584 | Yes | 5.5673 | Large | 1.0000 | 0.578147 |
| cifar10 | mobilenetv2 | images_per_sec_avg | 0.001523 | Yes | -14.7751 | Large | -1.0000 | -208.566667 |
