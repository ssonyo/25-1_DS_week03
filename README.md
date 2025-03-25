

# Week 03 과제
**Due 3/31 23:59 p.m.**

25.03.26 01:33) 데이터셋 변경

## Assignment1. Sentence Embedding ; SimCSE implementation

1. `train.py`의 TODO를 채워 Unsupervised SimCSE를 구현하고,
2. embedding 을 faiss 에 저장해 유사 문장 검색을 해봅시다.
   (faiss는 gpu 버전이므로 vessl을 쓰거나 colab을 쓰세용.)
4. query와 가장 유사한 문장 5개를 뽑아보고 캡처해서 제출합시다.
   
❗daily dialogue data 로 학습했으므로 일상적인 query를 날리면 됩니다.

![image](https://github.com/user-attachments/assets/f3eca10a-d0ac-4371-a815-6b0b1e855c2c)



## Assignment2. End-to-end Token embedding

0. `gen_data.py` 를 실행해 data/simple_corpus.txt 를 얻읍시다. (`Assignment02` 디렉토리 안에서 실행하세요~)
1. `embedding.py`는 1주차 과제 transformer 의 embedding 과 같은 것을 사용했습니다.
   안 하신 분들은 구현해보시고, 하신 분들은 복붙하셔도 됩니다.
2. 이 코드는 decoder 대신 classifier을 사용해 다음 단어를 맞추는 방식 (sequence-to-one)으로 embedding layer와 model parameter을 end-to-end로 함께 학습합니다.
3. nn.Embedding 과 Contextual embedding 이 어떻게 변화하는지 다양한 방법으로 관찰해봅시다
4. `train.py`의 마지막 feature을 캡처해서 제출합시다!

``` bash
25-1_DS_Week03
├── Assignment01
│   ├── dataset.py
│   ├── eval.ipynb
│   ├── model.py
│   └── train.py
├── Assignment02
│   ├── __init__.py
│   ├── data.py
│   ├── embeddings.py
│   ├── encoder.py
│   ├── model.py
│   └── train.ipynb
├── assignment1.png (제출!)
├── assignment2.png (제출!)
├── gen_data.py
├── requirements.txt
├── .gitignore
└── README.md

```
