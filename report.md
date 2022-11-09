## Sparsity structure

In this part I choose [deberta-v2-xxlarge](https://huggingface.co/microsoft/deberta-v2-xxlarge?text=The+goal+of+life+is+%5BMASK%5D.) (1.5B) for the encoder-only model, [gpt-xl]( https://huggingface.co/gpt2-xl?text=My+name+is+Thomas+and+my+main) (1.5B) for the decoder-only model, [t5](https://huggingface.co/t5-3b?text=My+name+is+Wolfgang+and+I+live+in+Berlin) (3B) for the encoder-decoder model. I plotted the histogram of the log absolute value of all the parameters to show how sparse the models are. 

For DeBERTa-v2-xxlarge, there are only 4.49e-06 of total parameters with scale larger than 1. For GPT2-xl, there are only 2.43e-06 of total parameters with scale larger than 1. For T5, there are only 

I also plotted the ratio per layer. It seems that the middle layers tend to have more small parameters and the layers on the top of encoder or decoder have more large parameters. 

## Prune

Given a pretrained model, I rely on `torch.nn.utils.prune` to prune it under different ratio. I choose to perform a global pruning for only once, which will set a certain amount of the smallest parameters in the whole model to 0 instead of keeping the same ratio for each layer. 

`torch.nn.utils.prune` achieves this goal by generating a buffer pruning mask for each weight and bias in a module. Before the forward pass, a pre hook will automatically combine the original parameters and their mask to generate a pruned weight/bias. The original weights can be kept the same. We can also make a permanent pruning with the `remove` functionality. 

## Evaluation results

The previous three models are too big to train on the benchmarks. So in future experiments I use GPT2-base, BERT-large and T5-base instead. 

For GPT2-base, I followed the standard process to get results on WikiText2 and WikiText103. I also sparsified the fine-tuned model at 10%, 50%, 90%, 95%, 99% and gathered their performance on the two tasks. For BERT-base, I followed the standard process to get results on WikiText2 and SQuAD. For T5-base, I followed the standard process to get results on SQuAD and WMT English to Romanian translation. 

The plots of different models on different benchmarks are in `src/plot`. The performance drops a lot when pruning ratio is larger or equal to 0.5. There is not much difference in model size and runtime between different sparsified models. That is, current pruning method can not improve efficiency but only hurt performance. 

## Challenges of sparsification

If we prune a model unstructuredly, we can not reduce the model size or computation. If we prune a model by removing a whole part, then the performance will be greatly suffered. There is a tradeoff between performance and efficiency. 


