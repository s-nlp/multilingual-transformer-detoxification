# SmurfCat at PAN 2024 TextDetox: Alignment of Multilingual Transformers for Text Detoxification
A code for PAN-2024 Multilingual Text Detoxification

**Elisei Rykov<sup>1</sup>**, **Konstantin Zaytsev<sup>2</sup>**, **Ivan Anisimov<sup>1</sup>**, **Alexandr Voronin<sup>1</sup>**

<sup>1</sup>Skolkovo Institute of Science and Technology, <sup>2</sup>HSE University
{Elisei.Rykov, Ivan.Anisimov, Alexander.Voronin}@skol.tech, kzaytsev@hse.ru

This paper presents a solution for the Multilingual Text Detoxification task in the PAN-2024 competition of the SmurfCat team. Using data augmentation through machine translation and a special filtering procedure, we collected an additional multilingual parallel dataset for text detoxification. Using the obtained data, we fine-tuned several multilingual sequence-to-sequence models, such as mT0 and Aya, on a text detoxification task. We applied the ORPO alignment technique to the final model. Our final model has only 3.7 billion parameters and achieves state-of-the-art results for the Ukrainian language and near state-of-the-art results for other languages. In the competition, our team achieved first place in the automated evaluation with a score of 0.52 and second place in the final human evaluation with a score of 0.74.

Paper: https://ceur-ws.org/Vol-3740/paper-276.pdf

Model: https://huggingface.co/s-nlp/mt0-xl-detox-orpo


# Preference data collection
For the best submission, we first performed supervised fine tuning on multilingual detoxification data (including translations obtained via Google Translator) and then performed preference tuning via ORPO. For preference tuning, we inferred a fine-tuned checkpoint on an unseen data set using diverse beam search. Furthermore, for preference tuning, we selected as rejects inference outputs with low similarity and high toxicity, and vice versa. See more in the [paper](https://ceur-ws.org/Vol-3740/paper-276.pdf).

# Training and inference scripts
`training.py` and `dpo_training.py` are scripts for training the base model and the preference-based checkpoint, respectively. 

`dpo_inference.py` is the script for sampling preference data.

`benchmark.py` is the model inference script. 


# Citation
```
@inproceedings{smurfcat_at_pan,
  author       = {Elisei Rykov and
                  Konstantin Zaytsev and
                  Ivan Anisimov and
                  Alexandr Voronin},
  editor       = {Guglielmo Faggioli and
                  Nicola Ferro and
                  Petra Galusc{\'{a}}kov{\'{a}} and
                  Alba Garc{\'{\i}}a Seco de Herrera},
  title        = {SmurfCat at {PAN} 2024 TextDetox: Alignment of Multilingual Transformers
                  for Text Detoxification},
  booktitle    = {Working Notes of the Conference and Labs of the Evaluation Forum {(CLEF}
                  2024), Grenoble, France, 9-12 September, 2024},
  series       = {{CEUR} Workshop Proceedings},
  volume       = {3740},
  pages        = {2866--2871},
  publisher    = {CEUR-WS.org},
  year         = {2024},
  url          = {https://ceur-ws.org/Vol-3740/paper-276.pdf},
  timestamp    = {Wed, 21 Aug 2024 22:46:00 +0200},
  biburl       = {https://dblp.org/rec/conf/clef/RykovZAV24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
