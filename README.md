# Discrete Flow Matching implemented in PyTorch

Implementation of [Discrete Flow Matching](https://arxiv.org/abs/2407.15595), which is a generative model for generating discrete things such as text with flow matching. The code is implemented in PyTorch.

## How to run

### Environment setup

1. (Optional) Create virtual environment with `python -m venv .venv` and activate with `source .venv/bin/activate`
2. Install the dependencies: `pip install -r requirements.txt`

- Only direct dependencies are specified right now
- The code was tested with Python 3.12

Run `python discrete_flow_matching/train.py` to start training a text generation model logging to wandb.

## Summary of discrete flow matching compared to continuous flow matching

- During training, we mask out text tokens according to the timestep
- The model is trained to predict the original unmasked tokens with cross entropy loss
- In sampling, we unmask text gradually with the sampled tokens

## References

- [Discrete Flow Matching](https://arxiv.org/abs/2407.15595)
- [Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design](https://arxiv.org/abs/2402.04997)([YouTube presentation](https://www.youtube.com/watch?v=yzc29vhM2Aw)): Combines discrete and continuous flow matching. Appendix F was very useful for the implementation
