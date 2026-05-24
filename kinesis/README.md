---
license: mit
---
# Kinesis Assets

## Description
This dataset contains essential assets required for setting up and running the KINESIS framework. For more details, please refer to the [paper](doi.org/10.48550/arXiv.2503.14637) and the [code repository](https://github.com/amathislab/Kinesis).

## Contents
- **Initial pose data**: Starting poses used to initialize the musculoskeletal model during motion imitation.
- **Text-to-motion**: Synthetic reference motions for natural language-driven control.

## Included files
```
kinesis-assets/
  ├── initial_pose/
  │    ├── initial_pose_test.pkl
  │    └── initial_pose_train.pkl
  └── t2m/
       ├── mdm_backward_0.pkl
       ...
```

HuggingFace paper link: arxiv.org/abs/2503.14637