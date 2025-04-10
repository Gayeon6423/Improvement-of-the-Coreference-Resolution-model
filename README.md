<<<<<<< HEAD
<h1 align="center">
    Improvement of the Coreference Resolution model through data augmentation based on LLM Adversarial filtering 
</h1>
<div align="center">
</div>

This is the official repository for Improvement of the Coreference Resolution model through data augmentation based on LLM Adversarial filtering.

# Setup
The `figma` Python package provides an easy API to use figma models, enabling efficient and accurate coreference resolution with few lines of code.

```bash
git clone https://github.com/Gayeon6423/Improvement-of-the-Coreference-Resolution-model.git
cd coref-main
pip install -r requirements.txt
```


# Augment
To augment a LitBank dataset, modify discriminate_model, generate_model, filtering_case_index in *config.json*.
Run the following:
```
conda activate coref_env
cd augmentation
python filetering.py
```

# Train
To train a Maverick model, modify *model/conf/root.yaml* with your custom setup. 
By default, this file contains the settings for training and evaluating on the LitBank augmentation dataset.

To train a new model, follow the steps in  [Environment](#environment) section and run the following script:
```
conda activate coref_env
cd model/maverick
python train.py
```

# Evaluate
To evaluate an existing model, it is necessary to set up two different environment variables.
1. Set the dataset path in conf/root.yaml, by default it is set to OntoNotes.
2. Set the model checkpoint path in conf/evaluation/default_evaluation.yaml.

Finally run the following:
```
conda activate coref_env
cd model/maverick
python evaluate.py
```
This will directly output the CoNLL-2012 scores, and, under the model/experiments/ folder,  a output.jsonlines file containing the model outputs in OntoNotes style.

## 참고 사항 ##
- augmentation/human_evaluation.ipynb : 인간평가 후 라벨링 하는 파일

## 추가 구현 사항 ## 
- 데이터 편향 분석 및 워드 클라우드 생성 코드 구축
- data.py : 데이터 합성 fuction 추가
=======
# Improvement-of-the-Coreference-Resolution-model
Improvement of the Coreference Resolution model through data augmentation based on LLM Adversarial filtering 
>>>>>>> d1fe5511a244987748478f2c41873e1408b5a012
