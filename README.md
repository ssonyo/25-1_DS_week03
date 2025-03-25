

# Week 03 과제

0. gen_data.py 를 실행해 data/simple_corpus.txt 를 얻읍시다. 두 task에서 같은 데이터셋을 쓸 예정입니다.

## Assignment1. Sentence Embedding ; SimCSE implementation

1. TODO를 채워 SimCSE를 구현하고,
2. embedding 을 faiss 에 저장해 유사 문장 검색을 해봅시다.


## Assignment2. End-to-end Token embedding

1. 1주차 과제 transformer 의 embedding 과 같은 것을 사용했습니다.
안 하신 분들은 구현해보시고, 하신분들은 복붙하셔도 됩니다.
2. 이 코드는 decoder 대신 classifier을 사용해 다음 단어를 맞추는 방식 (sequence-to-one)으로 embedding과 model parameter을 end-to-end로 함께 학습합니다.
3. nn.Embedding 과 Contextual embedding 이 어떻게 변화하는지 관찰해봅시다!

   ![image](https://github.com/user-attachments/assets/3671bcb2-73c1-48f2-b9fa-b6c23f117325)
