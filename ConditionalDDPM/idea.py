"""
1. can we control the idea of choosing the condition in guidance free based on the binary map which is the flood no flood label?
a. so regions in which there is no flood those regions remain the same,
but the region in which there is flood the model does the sar to optical translation.

2. what are the various ways of incorporating the condition. 
a. can it be something else other than concatenating?
b. can the condition be introduced as the noise by protruding the condition?

"""

"""
SOMETHING VERY IMPORTANT IN CLASSIFIER FREE GUIDANCE STRATEGY

If the model never learned what label 0 means as an unconditional condition, then nonEps may not be meaningful.

So when we inspect the conditional Model.py or conditional trainer, we must check:

Does the code randomly drop labels during training?

If not, classifier-free guidance may not be correctly implemented.

"""