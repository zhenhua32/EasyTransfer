# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
# Copyright (c) 2019 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import collections
import sentencepiece as spm
import six
import tensorflow as tf
import unicodedata
from six.moves import range

# 下划线
SPIECE_UNDERLINE = u"▁".encode("utf-8")


def encode_pieces(sp_model, text, return_unicode=True, sample=False):
    """turn sentences into word pieces.
    将句子变成 word pieces(子词)
    """

    if six.PY2 and isinstance(text, six.text_type):
        # 变成 bytes 类型
        text = six.ensure_binary(text, "utf-8")

    # 是否进行简单处理
    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    # 新的子词
    new_pieces = []
    for piece in pieces:
        # 对于每一个子词
        piece = printable_text(piece)
        # 如果长度大于 1, 且最后一个字符是逗号, 且最后第二个字符是整数, 比如 12,
        if len(piece) > 1 and piece[-1] == "," and piece[-2].isdigit():
            # 去掉最后一个字符, 然后将 _ 删除掉
            cur_pieces = sp_model.EncodeAsPieces(
                six.ensure_binary(piece[:-1]).replace(SPIECE_UNDERLINE, b""))
            # 如果第一个字符不是 _ , 且 cur_pieces 的第一个元素的第一个元素是 _
            if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                # 如果 cur_pieces[0] 长度等于 1, 相当于 cur_pieces[0] 是 [_]
                if len(cur_pieces[0]) == 1:
                    # 就从第一个元素以后开始取, 抛弃 cur_pieces 的第一个元素
                    cur_pieces = cur_pieces[1:]
                else:
                    # 否则将第一个元素替换成 第一个元素的 [1:] 以后的元素, 就是对第一个元素删除它的第一个元素
                    cur_pieces[0] = cur_pieces[0][1:]
            # 将 piece 的最后一个元素添加进去
            cur_pieces.append(piece[-1])
            # 直接扩展数组
            new_pieces.extend(cur_pieces)
        else:
            # 直接添加, 不做处理
            new_pieces.append(piece)

    # note(zhiliny): convert back to unicode for py2
    if six.PY2 and return_unicode:
        ret_pieces = []
        for piece in new_pieces:
            if isinstance(piece, str):
                piece = six.ensure_text(piece, "utf-8")
            ret_pieces.append(piece)
        new_pieces = ret_pieces

    return new_pieces


def encode_ids(sp_model, text, sample=False):
    """
    编码 ids
    """
    pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
    # 转换成 id
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return ids


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input
    转换成 unicode
    """
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return six.ensure_text(text, "utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return six.ensure_text(text, "utf-8", "ignore")
        elif isinstance(text, six.text_type):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`.
    将文本编码成适合打印或 tf.logging 记录的
    """

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return six.ensure_text(text, "utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        # 对于 python2, 转换成 str 类型
        if isinstance(text, str):
            return text
        elif isinstance(text, six.text_type):
            return six.ensure_binary(text, "utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file: str) -> collections.OrderedDict:
    """Loads a vocabulary file into a dictionary.
    将词汇表文件加载到字典中
    """
    vocab = collections.OrderedDict()
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            # 每一行都是一个词汇
            token = token.strip().split()[0] if token.strip() else " "
            if token not in vocab:
                # key 是词汇, value 是当前 vocab 的长度, 就是从 0 开始的
                vocab[token] = len(vocab)
    return vocab


def convert_by_vocab(vocab: dict, items: list) -> list:
    """Converts a sequence of [tokens|ids] using the vocab.
    使用词汇表转换序列
    """
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def whitespace_tokenize(text: str) -> list:
    """Runs basic whitespace cleaning and splitting on a piece of text.
    清理前后空格, 然后用空白符分割
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

class FullTokenizer(object):
    """Runs end-to-end tokenziation.
    运行一个端到端的分词器
    """

    def __init__(self, vocab_file=None, do_lower_case=True, spm_model_file=None):
        self.vocab = None
        self.sp_model = None
        if spm_model_file:
            # https://github.com/google/sentencepiece
            # https://pypi.org/project/sentencepiece/
            # sentencepiece 是个分词器
            self.sp_model = spm.SentencePieceProcessor()
            tf.logging.info("loading sentence piece model")
            # Handle cases where SP can't load the file, but gfile can.
            sp_model_ = tf.gfile.GFile(spm_model_file, "rb").read()
            self.sp_model.LoadFromSerializedProto(sp_model_)
            # Note(mingdachen): For the purpose of consisent API, we are
            # generating a vocabulary for the sentence piece tokenizer.
            # 生成词汇字典
            self.vocab = {self.sp_model.IdToPiece(i): i for i
                          in range(self.sp_model.GetPieceSize())}
        else:
            # 从文件中加载词汇表, 词汇表的 key 是单词, val 是数字序号
            self.vocab = load_vocab(vocab_file)
            # 使用一个基础的分词器
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
            # 一个 wordpiece 分词器
            self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        # 词汇表的逆转, 从 id 到词汇
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def __len__(self):
        """ Size of the full vocabulary with the added tokens 
        词汇表的大小
        """
        return len(self.vocab)

    def tokenize(self, text):
        """
        分词, 令牌化
        """
        if self.sp_model:
            split_tokens = encode_pieces(self.sp_model, text, return_unicode=False)
        else:
            split_tokens = []
            # 先对文本调用基础的分词器
            for token in self.basic_tokenizer.tokenize(text):
                # 然后对每一个词调用 wordpiece 分词器
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """
        将 token 转换成 id 格式
        """
        if self.sp_model:
            return [self.sp_model.PieceToId(
                printable_text(token)) for token in tokens]
        else:
            return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        """
        将 id 转换成 token 格式
        """
        if self.sp_model:
            return [self.sp_model.IdToPiece(id_) for id_ in ids]
        else:
            return convert_by_vocab(self.inv_vocab, ids)

class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.).
    一个基础的分词器
    """

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        # 是否对输入进行小写化
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text.
        对文本进行分词
        """
        # 转换成 unicode 格式
        text = convert_to_unicode(text)
        # 清理字符
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        # 处理中文字符
        text = self._tokenize_chinese_chars(text)

        # 使用空白分割器
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 没想到这个是标准库 https://docs.python.org/zh-cn/3/library/unicodedata.html
        # 正规化字符
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            # 跳过这一类的
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text.
        分割标点
        """
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        # 小于长度时
        while i < len(chars):
            # 取一个字符
            char = chars[i]
            # 如果是标点
            if _is_punctuation(char):
                # 直接添加进去, 加一个小数组
                output.append([char])
                # 又是开始一个新的单词
                start_new_word = True
            else:
                # 不是标点
                # 如果是开始一个新的字符, 添加一个空数组
                if start_new_word:
                    # 这个空数组就是占位用的, 这样之后, 后面才可以用 output[-1] 添加字符
                    output.append([])
                # 变成 False
                start_new_word = False
                # 在最后一个元素中添加这个字符
                output[-1].append(char)
            i += 1

        # 返回数组, 将每个小数组变成字符串
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character.
        在 CJK 字符周围添加空格
        """
        output = []
        for char in text:
            cp = ord(char)
            # 如果是中文字符, 就在前后加上空格
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character.
        检查是否是中文字符
        """
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text.
        对无效的字符进行删除, 并对空格进行清理
        """
        output = []
        for char in text:
            # 对每一个字符进行处理
            # 返回 unicode code point, 是个 int 类型
            cp = ord(char)
            # 跳过这些无效的字符
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation.
    wordpiece 分词
    因为单词并不一定在词汇表中, 所以需要调用子词分割器
    """

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        # 词汇表
        self.vocab = vocab
        # 未知字符
        self.unk_token = unk_token
        # 每个词的最大字符长度
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        将 text 的 piece 分割成 word 的 piece

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        使用贪婪的最长匹配算法, 执行分词, 使用指定的词汇表

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        # 空白分割, 对每一个 token 进行处理
        for token in whitespace_tokenize(text):
            # 变成字符数组
            chars = list(token)
            # 超出最大长度, 则返回未知字符
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            # 是否是坏的
            is_bad = False
            # 起点
            start = 0
            sub_tokens = []
            # 当起点小于字符数组长度
            while start < len(chars):
                # 终点
                end = len(chars)
                # 当前子串
                cur_substr = None
                # 当起点小于终点时
                while start < end:
                    # 子串是起点到终点的字符串
                    substr = "".join(chars[start:end])
                    # 起点大于 0 时, 表示不是单词的开头, 所以要加上 ## 前缀
                    if start > 0:
                        substr = "##" + six.ensure_text(substr)
                    # 如果当前子串在词汇表中, 则跳出循环
                    if substr in self.vocab:
                        # 当前子串就是该子串
                        cur_substr = substr
                        break
                    end -= 1
                
                # 如果当前子串是空的, 表示是个坏的, 退出循环
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                # 起点变终点, 也就是退出了循环
                start = end

            # 如果是坏的, 就用未知字符
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character.
    检查是否是空白字符
    """
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character.
    检查是否是控制字符
    """
    # These are technically control characters but we count them as whitespace
    # characters.
    # 对这些字符的处理是和 _is_whitespace 函数一致的
    if char == "\t" or char == "\n" or char == "\r":
        return False
    # 主要还是依靠这个函数返回的类目来判断
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character.
    判断是否是标点符号
    """
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    # 在 33-47 之间 或者 在 58-64 之间的字符都是标点符号
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    # 然后依旧是根据类别进行判断
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
