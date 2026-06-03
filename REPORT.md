# mini GPT 구현 과제 보고서

## 0. 반·팀원

| 항목  | 내용                 |
| --- | ------------------ |
| 반   | SW-AI 301반         |
| 팀명  | mini GPT 구현팀       |
| 팀원  | 고민석, 나지운, 박건우, 박민석 |

---

## 1. 구현 현황

| 단계  | 구현 내용                                                                          | 구현 파일                                 | 담당자 |
| --- | ------------------------------------------------------------------------------ | ------------------------------------- | --- |
| 1   | UTF-8 byte-level BPE tokenizer                                                 | `src/bpe.py`                          | 나지운 |
| 2   | GPTDataset, create_dataloader, InputEmbedding                                  | `src/dataset.py`, `src/embeddings.py` | 박민석 |
| 3   | MultiHeadAttention, causal mask                                                | `src/attention.py`                    | 박건우 |
| 4   | LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel, generate_text_simple | `src/model.py`                        | 고민석 |
| 5   | loss 계산, checkpoint, generate, train_model                                     | `src/train.py`                        | 공통  |
| 6   | NSMC 감성 분류 Dataset과 classifier                                                 | `src/finetune.py`                     | 미구현 |

---

## 2. 테스트 통과 현황

| 실행 명령                               | 결과  | 비고                                |
| ----------------------------------- | --- | --------------------------------- |
| `pytest tests/test_bpe.py -v`       | 통과  | Colab: 6 passed in 0.03s          |
| `pytest tests/test_dataset.py -v`   | 통과  | Colab: 4 passed in 3.44s          |
| `pytest tests/test_attention.py -v` | 통과  | Colab: 2 passed in 1.87s          |
| `pytest tests/test_model.py -v`     | 통과  | Colab: 7 passed in 2.07s          |
| `pytest tests/test_train.py -v`     | 통과  | Colab: 5 passed in 9.87s          |
| `pytest tests/test_finetune.py -v`  | 실패  | Colab: 3 failed, 1 passed in 2.23s |
| `pytest tests/ -v`                  | 실패  | Colab: 3 failed, 25 passed in 3.35s |

실패한 테스트가 있다면 에러 요약을 적습니다.

| 실패한 테스트                                                                                        | 에러 요약                                                                  | 해결 시도                                                       |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------- |
| `tests/test_finetune.py::TestMakeSentimentDataset::test_make_sentiment_dataset_splits_rows`    | `make_sentiment_dataset` 미구현으로 `NotImplementedError` 발생                | 현재 발표 범위를 Attention/사전학습 흐름까지로 조정하여 fine-tuning은 미구현 상태로 남김 |
| `tests/test_finetune.py::TestReviewSentimentDataset::test_review_sentiment_dataset_getitem`    | `ReviewSentimentDataset.__getitem__` 미구현으로 `NotImplementedError` 발생    | fine-tuning 단계 구현 대상에 포함                                    |
| `tests/test_finetune.py::TestGPTForSequenceClassification::test_sequence_classification_shape` | `GPTForSequenceClassification.__init__` 미구현으로 `NotImplementedError` 발생 | fine-tuning classifier 구현 대상에 포함                            |

---

## 3. 데이터

| 항목         | 내용                                                                                                   |
| ---------- | ---------------------------------------------------------------------------------------------------- |
| 원본 데이터     | NSMC                                                                                                 |
| 원본 경로      | `data/ratings_train.txt`, `data/ratings_test.txt`                                                    |
| 사전 학습 데이터  | `data/nsmc_lm_train.txt`, `data/nsmc_lm_val.txt`                                                     |
| 미세 조정 데이터  | `data/nsmc_sentiment_train.jsonl`, `data/nsmc_sentiment_val.jsonl`, `data/nsmc_sentiment_test.jsonl` |
| 전처리 방식     | 빈 리뷰 제거, 공백 정리, train/validation 분리                                                                  |
| 사용한 데이터 크기 | 미사용: 현재 `data/.gitkeep`만 존재하고 NSMC 데이터 미다운로드                                                         |

---

## 4. BPE

| 항목               | 내용                                                                                                  |
| ---------------- | --------------------------------------------------------------------------------------------------- |
| 구현 파일            | `src/bpe.py`                                                                                        |
| BPE 방식           | UTF-8 byte-level BPE                                                                                |
| 특수 토큰 ID         | `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`                                                          |
| byte token ID 범위 | 4~259                                                                                               |
| vocab_size       | 4096                                                                                                |
| 학습 corpus 크기     | 전체 1,379,486자 중 409,600자 사용                                                                         |
| 어휘 학습 시간         | 734.31초                                                                                             |
| vocabulary 저장 경로 | `/content/gpt-lab/data/nsmc_bpe_vocab_4096.json`                                                    |
| 인코딩/디코딩 복원 예시    | `decode(encode("이 영화는 좋았다. byte-level BPE test!")) == "이 영화는 좋았다. byte-level BPE test!"`, 토큰 ID 22개 |

---

## 5. 모델 구조

| 항목             | 내용                                                             |
| -------------- | -------------------------------------------------------------- |
| 구현 파일          | `src/model.py`                                                 |
| 전체 구조          | InputEmbedding -> N x TransformerBlock -> LayerNorm -> LM head |
| vocab_size     | 4096                                                           |
| context_length | 128                                                            |
| emb_dim        | 192                                                            |
| n_heads        | 4                                                              |
| n_layers       | 4                                                              |
| drop_rate      | 0.1                                                            |
| qkv_bias       | False                                                          |
| 총 파라미터 수       | 약 3,534,976개                                                   |

---

## 6. 사전 학습

### 6.1 하이퍼파라미터

| 구분  | 항목                   | 값        |
| --- | -------------------- | -------- |
| 모델  | vocab_size           | 4096     |
| 모델  | context_length       | 128      |
| 모델  | emb_dim              | 192      |
| 모델  | n_heads              | 4        |
| 모델  | n_layers             | 4        |
| 학습  | corpus_size          | 409,600자 |
| 학습  | batch_size           | 16       |
| 학습  | num_epochs           | 1        |
| 학습  | eval_freq, eval_iter | 100, 10  |
| 최적화 | learning_rate        | 0.0003   |
| 최적화 | weight_decay         | 0.1      |

### 6.2 결과

| 항목              | 내용                                                          |
| --------------- | ----------------------------------------------------------- |
| train loss      | 7.536383752261891                                           |
| validation loss | 7.279816484451294                                           |
| 손실 그래프          | 미생성                                                         |
| 생성 샘플           | 영화는의에다. 이를은이은가들이 그리을을에는이다한을의한한이다. 한도. 을이이거의만가는이은..는.의이아로도아로 |
| checkpoint 경로   | `checkpoint_epoch_1.pt`                                     |

---

## 7. 미세 조정

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/finetune.py` |
| 과제 | NSMC 리뷰 긍정/부정 분류 |
| 데이터 포맷 | JSONL, `text`, `label` |
| max_length | 128 |
| batch_size | 16 |
| 구현 상태 | 미구현: 발표 범위에서 제외 |
| backbone learning rate | 미실행 |
| classifier learning rate | 미실행 |
| validation loss / accuracy | 미측정 |
| test loss / accuracy | 미측정 |
| 오류 예시 | `make_sentiment_dataset`, `ReviewSentimentDataset.__getitem__`, `GPTForSequenceClassification` 미구현으로 테스트 실패 |

---

## 8. 실험 환경

| 항목         | 내용                       |
| ---------- | ------------------------ |
| Python     | Colab Python 3.12.13     |
| PyTorch    | PyTorch 2.11.0+cu128     |
| 실행 환경      | Colab GPU                |
| GPU/CPU 정보 | Tesla T4, CUDA 사용 가능     |
| 총 학습 소요 시간 | BPE 734.31초, 사전 학습 2.98초 |
|            |                          |

---

## 9. 고찰

- 어려웠던 점: tokenizer, attention, model, train 기능을 외부 pretrained 모델 없이 직접 연결해야 했다. fine-tuning은 현재 발표 범위 밖으로 두어 미구현 상태다.
- 한국어 byte-level BPE 구현에서 조심한 점: 한국어를 글자 단위로 자르지 않고 UTF-8 byte sequence 기준으로 encode/decode가 복원되도록 구현했다.
- loss가 줄어든 이유 또는 줄어들지 않은 이유: 1 epoch 축소 학습 결과 train loss는 7.5364, validation loss는 7.2798로 측정되었다.
- 과적합·과소적합 여부: validation loss가 train loss보다 낮지만, 1 epoch와 축소 corpus 기준이라 일반화 여부를 판단하기에는 추가 학습이 필요하다.
- 하이퍼파라미터 변경 시도와 결과: vocab_size 4096, context_length 128, emb_dim 192, n_heads 4, n_layers 4, batch_size 16, lr 0.0003 설정으로 1 epoch 실험을 진행했다.
- 다음에 개선하고 싶은 점: 더 큰 corpus와 더 많은 epoch로 사전 학습을 진행하고, 현재 미구현인 NSMC 감성 분류 미세 조정을 구현해 전체 테스트를 통과시킨다.
