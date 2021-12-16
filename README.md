# PyTorch TensorMemmap

- a .ptmap format for huggingface weights that provides for direct mapping between disk and memory

- CPU models are no longer limited by available RAM

- pytorch_tensormap.py can be run from the terminal to generate ptmap files or convert a model

- example.py demonstrates use of a ptmap model by importing pytorch_tensormap and using a context hack

- put your models (~/.cache/huggingface/transformers) on very fast solid state media

- execution is slow so batch data together
