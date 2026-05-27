# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
from collections import Counter
import json

PAD_TOKEN = "<pad>"  # padding: 문장 길이 맞출 때 빈자리 채우기
UNK_TOKEN = "<unk>"  # unknown: 어휘사전에 없는 토큰
BOS_TOKEN = "<bos>"  # begin of sentence: 문장 시작 신호
EOS_TOKEN = "<eos>"  # end of sentence: 문장 끝 신호

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        for token, idx in SPECIAL_IDS.items():
            self.id_to_token[idx] = token
            self.token_to_id[token] = idx
        
        for byte_value in range(NUM_BYTES):
            self.id_to_token[byte_value + BYTE_OFFSET] = bytes([byte_value])
            self.token_to_id[bytes([byte_value])] = byte_value + BYTE_OFFSET

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        self._init_special_tokens()

        # [1]
        token_ids = [b + BYTE_OFFSET for b in corpus.encode("utf-8")]
        
        while len(self.id_to_token) < self.vocab_size: # [종료조건 1] 목표한 vocab_size에 도달했을 때
            # [2] pair counting & find best pair
            pairs = list(zip(token_ids, token_ids[1:]))
            pair_counts = Counter(pairs)
            best_pair = max(pair_counts, key=lambda x: pair_counts[x])

            if max(pair_counts.values()) < 2: # [종료조건 2] 더 이상 쌍이 없을 때
                break

            # [3] merge
            new_id = len(self.id_to_token)
            new_token_ids = []
            i = 0
            while i < len(token_ids):
                if i < len(token_ids) - 1 and (token_ids[i], token_ids[i+1]) == best_pair:
                    new_token_ids.append(new_id)
                    i += 2
                else:
                    new_token_ids.append(token_ids[i])
                    i += 1
            token_ids = new_token_ids
        
            # [4]
            self.id_to_token[new_id] = best_pair
            self.token_to_id[best_pair] = new_id
            self.merges.append(best_pair)

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        data = {
            "vocab_size": self.vocab_size,
            "id_to_token": {},
            "merges": []
        }

        for key, value in self.id_to_token.items():
            if isinstance(value, bytes):
                data["id_to_token"][key] = {"type": "bytes", "data": list(value)}
            elif isinstance(value, tuple):
                data["id_to_token"][key] = {"type": "tuple", "data": list(value)}
            else:
                data["id_to_token"][key] = value
        
        for pair in self.merges:
            data["merges"].append(list(pair))

        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        with open(path, "r") as f:
            data = json.load(f)
        
        self.vocab_size = data["vocab_size"]
        
        for key, value in data["id_to_token"].items():
            if isinstance(value, dict):
                raw = value["data"]
                if value["type"] == "bytes":
                    self.id_to_token[int(key)] = bytes(raw)
                    self.token_to_id[bytes(raw)] = int(key)
                elif value["type"] == "tuple":
                    self.id_to_token[int(key)] = tuple(raw)
                    self.token_to_id[tuple(raw)] = int(key)
            else:
                self.id_to_token[int(key)] = value
                self.token_to_id[value] = int(key)
    
        for pair in data["merges"]:
            self.merges.append(tuple(pair))


    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        # [1]
        token_ids = [b + BYTE_OFFSET for b in text.encode("utf-8")]

        # [2]
        for best_pair in self.merges:
            new_token_ids = []
            i = 0
            while i < len(token_ids):
                if i < len(token_ids) - 1 and (token_ids[i], token_ids[i+1]) == best_pair:
                    new_token_ids.append(self.token_to_id[best_pair])
                    i += 2
                else:
                    new_token_ids.append(token_ids[i])
                    i += 1
            token_ids = new_token_ids

        # [3]
        if add_bos_eos:
            return [self.get_bos_id()] + token_ids + [self.get_eos_id()]
        return token_ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        byte_tokens = []

        for id in ids:
            if skip_special and id in (self.get_pad_id(), self.get_unk_id(), self.get_bos_id(), self.get_eos_id()):
                continue
            byte_tokens.append(self.expand(id))

        return b"".join(byte_tokens).decode("utf-8")
        
    def expand(self, id):
        token = self.id_to_token[id]
        if isinstance(token, bytes):
            return token
        else:
            left, right = token
            return self.expand(left) + self.expand(right)
