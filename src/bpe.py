# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

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
        self.merges = []

        for token, token_id in SPECIAL_IDS.items():
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        for byte_value in range(NUM_BYTES):
            token_id = BYTE_OFFSET + byte_value
            token = bytes([byte_value])
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

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
        ids = [BYTE_OFFSET + b for b in corpus.encode("utf-8")]

        while len(self.id_to_token) < self.vocab_size:
            pair_counts = Counter(zip(ids, ids[1:]))
            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break

            new_id = len(self.id_to_token)
            self.id_to_token[new_id] = best_pair
            self.token_to_id[best_pair] = new_id
            self.merges.append(best_pair)

            merged = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged

    def save(self, path: str | Path):
        """
        vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        path = Path(path)
        vocab = []
        for token_id, token in sorted(self.id_to_token.items()):
            if isinstance(token, str):
                item = {"id": token_id, "type": "str", "value": token}
            elif isinstance(token, bytes):
                item = {"id": token_id, "type": "bytes", "value": list(token)}
            elif isinstance(token, tuple):
                item = {"id": token_id, "type": "tuple", "value": list(token)}
            else:
                raise TypeError(f"Unsupported token type: {type(token)!r}")
            vocab.append(item)

        data = {
            "vocab_size": self.vocab_size,
            "id_to_token": vocab,
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str | Path):
        """
        save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.vocab_size = data.get("vocab_size", self.vocab_size)
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = [tuple(pair) for pair in data["merges"]]

        for item in data["id_to_token"]:
            token_id = int(item["id"])
            token_type = item["type"]
            value = item["value"]

            if token_type == "str":
                token = value
            elif token_type == "bytes":
                token = bytes(value)
            elif token_type == "tuple":
                token = tuple(value)
            else:
                raise ValueError(f"Unknown token type: {token_type}")

            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

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

        ids = [BYTE_OFFSET + b for b in text.encode("utf-8")]

        for pair in self.merges:
            merge_id = self.token_to_id.get(pair)
            if merge_id is None:
                continue

            merged = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                    merged.append(merge_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged

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

        def expand(token_id: int) -> bytes:
            token = self.id_to_token.get(token_id)
            if token is None:
                return UNK_TOKEN.encode("utf-8")
            if isinstance(token, bytes):
                return token
            if isinstance(token, tuple):
                return b"".join(expand(part_id) for part_id in token)
            if isinstance(token, str):
                if skip_special and token in SPECIAL_TOKENS:
                    return b""
                return token.encode("utf-8")
            raise TypeError(f"Unsupported token type: {type(token)!r}")

        byte_sequence = b"".join(expand(token_id) for token_id in ids)
        return byte_sequence.decode("utf-8", errors="replace")
