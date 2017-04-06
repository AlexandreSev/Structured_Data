


# Structured_Data

## Presentation of unconstrained text recognition 

Text and letter recognition is a trending research topic. Applications of neural networks showed high accuracy. We develop a model for unconstrained recognition of words in natural images. It's an unconstrained recognition because there are no fixed lexicon and words have unknown length. 

To solve this unconstrained recognition problem we created a joint model thanks to a Conditional Random Field. This joint model is a combination of a character predictor model and a N-gram predictor model. Moreover we optimised this joint model by back-propagating the structured output loss. We observed better performances with the joint model than with the character sequence model alone.

## Implementation of deep structured output learning for unconstrained text recognition

We have implemented three models in order to solve this unconstrained recognition :
- a character sequence model (model_1)
- a N grams model (model_2)
- a joint model wich is a combination of the two first model thanks to a CRF (model_3)

Before any model, we preprocessed all pictures : we needed to resize all pictures to the same size. Then the two first have been implemented with our own CNN and then with Resnet.

# Demo 

Please, follow the notebook Demonstration.ipynb.

# Sources

- M. Jaderberg, K. Simonyan, A. Vedaldi, and A. Zisserman.  Synthetic data and artificial neural networks for natural scene text recognition. arXiv preprint arXiv :1406.2227, 2014.

- Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. In S. Haykin and B. Kosko, editors, Intelligent Signal Processing, pages 306–351. IEEE Press, 2001.

- Andrea Vedaldi Max Jaderberg, Karen Simonyan and Andrew Zisserman. Deep structured output learning for unconstrained text recognition. 10 April 2015.

- Joseph  Redmon  and  Ali  Farhadi.    Yolo9000  :  Better,  faster,  stronger. arXiv  preprint arXiv : 1612.08242, 2016.



For any remark, advice or question, please send us an email at one of the following email addresses:

<ul>
<li> finas.melanie@gmail.com </li>
<li> isnardy.a@gmail.com </li>
<li> hamid.jalalzai@gmail.com </li>
<li> sevin.alexandre@gmail.com </li>
</ul>
