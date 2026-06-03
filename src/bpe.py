# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

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
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        self.id_to_token = {}
        self.token_to_id = {}

        for token, token_id in SPECIAL_IDS.items(): 
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        for byte_value in range(NUM_BYTES):
            token_id = BYTE_OFFSET + byte_value
            byte_token = bytes([byte_value])
            self.id_to_token[token_id] = byte_token
            self.token_to_id[byte_token] = token_id

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
        코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        self._init_special_tokens()
        self.merges = []

        ids = [BYTE_OFFSET + byte for byte in corpus.encode("utf-8")]
        next_token_id = BYTE_OFFSET + NUM_BYTES

        while len(self.id_to_token) < self.vocab_size and len(ids) >= 2:
            pair_counts = Counter(zip(ids, ids[1:]))
            if not pair_counts:
                break

            best_pair, count = pair_counts.most_common(1)[0]
            if count < 2:
                break

            self.merges.append(best_pair)
            self.id_to_token[next_token_id] = best_pair
            self.token_to_id[best_pair] = next_token_id
            ids = self._replace_pair(ids, best_pair, next_token_id)
            next_token_id += 1

        return self

    def save(self, path: str | Path):
        """
        vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        path = Path(path)
        vocab = []
        for token_id, token in sorted(self.id_to_token.items()):
            if isinstance(token, bytes):
                token_info = {"type": "byte", "value": list(token)}
            elif isinstance(token, tuple):
                token_info = {"type": "merge", "value": list(token)}
            else:
                token_info = {"type": "special", "value": token}
            vocab.append({"id": token_id, "token": token_info})

        payload = {
            "vocab_size": self.vocab_size,
            "vocab": vocab,
            "merges": [list(pair) for pair in self.merges],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str | Path):
        """
        save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.vocab_size = int(payload.get("vocab_size", self.vocab_size))
        self.id_to_token = {}
        self.token_to_id = {}
        for item in payload["vocab"]:
            token_id = int(item["id"])
            token_info = item["token"]
            token_type = token_info["type"]
            value = token_info["value"]
            if token_type == "byte":
                token = bytes(value)
            elif token_type == "merge":
                token = tuple(int(x) for x in value)
            else:
                token = value
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        self.merges = [tuple(int(x) for x in pair) for pair in payload.get("merges", [])]
        return self

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        if not self.id_to_token:
            self._init_special_tokens()

        ids = [BYTE_OFFSET + byte for byte in text.encode("utf-8")]
        for pair in self.merges:
            new_id = self.token_to_id.get(pair)
            if new_id is not None:
                ids = self._replace_pair(ids, pair, new_id)

        if add_bos_eos:
            ids = [self.get_bos_id()] + ids + [self.get_eos_id()]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        if not self.id_to_token:
            self._init_special_tokens()

        byte_values: list[int] = []
        for token_id in ids:
            self._append_token_bytes(int(token_id), byte_values, skip_special=skip_special)
        return bytes(byte_values).decode("utf-8", errors="replace")

    @staticmethod
    def _replace_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        result: list[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    def _append_token_bytes(self, token_id: int, out: list[int], skip_special: bool) -> None:
        token = self.id_to_token.get(token_id)
        if token is None:
            if not skip_special:
                out.extend(UNK_TOKEN.encode("utf-8"))
            return

        if isinstance(token, bytes):
            out.extend(token)
        elif isinstance(token, tuple):
            for child_id in token:
                self._append_token_bytes(child_id, out, skip_special=skip_special)
        else:
            if not skip_special:
                out.extend(str(token).encode("utf-8"))
