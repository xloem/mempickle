import transformers

import tensormap

tensormap = tensormap.TensorMap('752929ace039baa8ef70fe21cdf9ab9445773d20e733cf693d667982e210837e.323c769945a351daa25546176f8208b3004b6f563438a7603e7932bae9025925.tensormap')

pipeline = transformers.pipeline('text-generation', model='gpt2', model_kwargs = dict(state_dict = tensormap.read()))

print(pipeline('hello'))
