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
| 6   | NSMC 감성 분류 Dataset과 classifier                                                 | `src/finetune.py`                     | 공통  |

---

## 2. 테스트 통과 현황

| 실행 명령                               | 결과  | 비고                                |
| ----------------------------------- | --- | --------------------------------- |
| `pytest tests/test_bpe.py -v`       | 통과  | Colab: 6 passed in 0.03s          |
| `pytest tests/test_dataset.py -v`   | 통과  | Colab: 4 passed in 3.44s          |
| `pytest tests/test_attention.py -v` | 통과  | Colab: 2 passed in 1.87s          |
| `pytest tests/test_model.py -v`     | 통과  | Colab: 7 passed in 2.07s          |
| `pytest tests/test_train.py -v`     | 통과  | Colab: 5 passed in 9.87s          |
| `pytest tests/test_finetune.py -v`  |     |                                   |
| `pytest tests/ -v`                  | 미실행 | 개별 테스트 기준 총 28개 통과, 전체 명령은 별도 미실행 |

실패한 테스트가 있다면 에러 요약을 적습니다.

| 실패한 테스트 | 에러 요약 | 해결 시도 |
| --- | --- | --- |
| 없음 | Colab 개별 테스트 28개 모두 통과 | 전체 테스트 명령 `pytest tests/ -v`는 별도 실행 필요 |

---

## 3. 데이터

| 항목 | 내용 |
| --- | --- |
| 원본 데이터 | NSMC |
| 원본 경로 | `data/ratings_train.txt`, `data/ratings_test.txt` |
| 사전 학습 데이터 | `data/nsmc_lm_train.txt`, `data/nsmc_lm_val.txt` |
| 미세 조정 데이터 | `data/nsmc_sentiment_train.jsonl`, `data/nsmc_sentiment_val.jsonl`, `data/nsmc_sentiment_test.jsonl` |
| 전처리 방식 | 빈 리뷰 제거, 공백 정리, train/validation 분리 |
| 사용한 데이터 크기 | 미사용: 현재 `data/.gitkeep`만 존재하고 NSMC 데이터 미다운로드 |

---

## 4. BPE

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/bpe.py` |
| BPE 방식 | UTF-8 byte-level BPE |
| 특수 토큰 ID | `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3` |
| byte token ID 범위 | 4~259 |
| vocab_size | 3000 |
| 학습 corpus 크기 | 미측정: 데이터 미다운로드 |
| 어휘 학습 시간 | 미측정 |
| vocabulary 저장 경로 | 미생성: `data/nsmc_bpe_vocab_3000.json` 예정 |
| 인코딩/디코딩 복원 예시 | `decode(encode("이 영화는 좋았다")) == "이 영화는 좋았다"` |

---

## 5. 모델 구조

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/model.py` |
| 전체 구조 | InputEmbedding -> N x TransformerBlock -> LayerNorm -> LM head |
| vocab_size | 3000 |
| context_length | 128 |
| emb_dim | 192 |
| n_heads | 4 |
| n_layers | 4 |
| drop_rate | 0.1 |
| qkv_bias | False |
| 총 파라미터 수 | 약 3,114,112개 |

---

## 6. 사전 학습

### 6.1 하이퍼파라미터

| 구분 | 항목 | 값 |
| --- | --- | --- |
| 모델 | vocab_size | 3000 |
| 모델 | context_length | 128 |
| 모델 | emb_dim | 192 |
| 모델 | n_heads | 4 |
| 모델 | n_layers | 4 |
| 학습 | batch_size | 16 |
| 학습 | num_epochs | 미실행 |
| 학습 | eval_freq, eval_iter | 미실행 |
| 최적화 | lr, weight_decay | 미실행 |

### 6.2 결과

| 항목 | 내용 |
| --- | --- |
| train loss | 미측정 |
| validation loss | 미측정 |
| 손실 그래프 | 미생성 |
| 생성 샘플 | 미생성 |
| checkpoint 경로 | 미생성 |

---

## 7. 미세 조정

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/finetune.py` |
| 과제 | NSMC 리뷰 긍정/부정 분류 |
| 데이터 포맷 | JSONL, `text`, `label` |
| max_length | 128 |
| batch_size | 16 |
| backbone learning rate | 미실행 |
| classifier learning rate | 미실행 |
| validation loss / accuracy | 미측정 |
| test loss / accuracy | 미측정 |
| 오류 예시 | 미측정 |

---

## 8. 실험 환경

| 항목 | 내용 |
| --- | --- |
| Python | Colab Python 3.12.13, 로컬 PowerShell PATH에서는 `python` 명령 미인식 |
| PyTorch | PyTorch 2.x 권장, Colab 테스트에서 torch 기반 테스트 통과 |
| 실행 환경 | Colab 테스트 실행, 학습은 미실행 |
| GPU/CPU 정보 | 미확인 |
| 총 학습 소요 시간 | 미측정 |

---

## 9. 고찰

- 어려웠던 점: tokenizer, attention, model, train, finetune 기능을 외부 pretrained 모델 없이 직접 연결해야 했다.
- 한국어 byte-level BPE 구현에서 조심한 점: 한국어를 글자 단위로 자르지 않고 UTF-8 byte sequence 기준으로 encode/decode가 복원되도록 구현했다.
- loss가 줄어든 이유 또는 줄어들지 않은 이유: 실제 사전 학습을 실행하지 않아 loss 변화는 아직 확인하지 못했다.
- 과적합·과소적합 여부: 학습 및 검증 loss가 없어 아직 판단하지 못했다.
- 하이퍼파라미터 변경 시도와 결과: 현재는 기본 설정만 정리했고 변경 실험은 진행하지 않았다.
- 다음에 개선하고 싶은 점: Python 3.11 환경을 구성해 전체 테스트를 통과시키고, NSMC 데이터로 BPE 학습과 사전 학습, 감성 분류 미세 조정을 실행해 수치를 보완한다.
