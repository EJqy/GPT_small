import regex as re  # 使用 regex 而非内置 re，因为它支持 Unicode 类别（如 \p{L}）
from collections.abc import Iterable

class BPETokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):

        # 1. 建立双向映射，方便查表
        self.vocab = vocab  # ID -> 字节块
        self.id_to_byte = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()} # 字节块 -> ID
        
        # 2. 将合并规则转换为Rank字典。
        # BPE 编码时，必须优先应用在训练阶段较早出现的合并规则。
        # 字典结构为: {(byte_a, byte_b): 顺序索引}
        self.merges = {pair: i for i, pair in enumerate(merges)}
        
        self.special_tokens = special_tokens or []
        
        # 3. 构建特殊 Token 的正则表达式
        if self.special_tokens:
            # 关键：必须按照长度从长到短排序（reverse=True）。
            # 这样正则引擎会优先匹配最长的特殊标记，防止重叠标记（如 <|a|><|b|>）被错误拆分。
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            # 使用 re.escape 确保标记中的特殊字符（如 | 或 [ ）被当作普通字符处理
            special_pattern = "|".join(re.escape(t) for t in sorted_special)
            self.special_regex = re.compile(special_pattern)
        else:
            self.special_regex = None

        # 4. GPT-2 官方预分词正则表达式。
        # 它的作用是在应用 BPE 合并前，先将文本切分成单词、标点、数字等逻辑块。
        # 这样做是为了防止 BPE 规则跨越单词或标点（例如：防止将 "dog" 的末尾和 "." 合并）。
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
    def encode(self, text: str) -> list[int]:

        if not text:
            return []

        # --- 步骤 2: 情况 A - 快速路径 (Fast Path) ---
        # 如果我们在初始化时没有定义任何特殊标记（或者特殊标记列表为空），
        # 那么整个文本都可以被视为一段连续的“普通文本”。
        # 我们直接调用内部方法 _encode_text_segment 进行 BPE 处理并返回结果。
        if not self.special_regex:
            return self._encode_text_segment(text)

        # --- 步骤 3: 情况 B - 处理含有特殊标记的复杂文本 ---
        # 此时文本中可能混有普通文字和特殊标记，我们需要像“剪刀”一样把它们切开。
        tokens = []
        
        # last_pos 用于记录上一次匹配结束的位置，帮助我们定位“特殊标记”之间的“缝隙”。
        last_pos = 0
        
        # 使用 finditer 遍历文本中所有符合特殊标记模式的匹配项。
        # finditer 的好处是它提供了 match.start() 和 match.end()，
        # 这让我们能够精确地知道特殊标记在哪里开始，在哪里结束。
        for match in self.special_regex.finditer(text):
            
            # 3.1 提取并处理“前置普通文本”
            # 这里的区间是 [last_pos, match.start())。
            # " hello <|endoftext|> world"
            # 这段文本是夹在两个特殊标记之间（或者开头到第一个特殊标记之间）的普通文字。
            pre_text = text[last_pos:match.start()]
            
            # 如果这两个标记之间确实有文字（长度 > 0）
            if pre_text:
                # 调用核心 BPE 逻辑。_encode_text_segment 会执行：
                # 1. GPT-2 预分词正则切分。
                # 2. 字节化。
                # 3. 按照 merges 规则进行贪婪合并。
                tokens.extend(self._encode_text_segment(pre_text))
                # pre_tokens : [1,2,3,...] self._encode_text_segment: [4,5,6] tokens.extend -> [1,2,3,...,4,5,6]
                # token.append() : [1,2,3,...,[4,5,6]]
            
            # 3.2 处理“当前特殊标记”
            # match.group() 拿到的就是被识别出来的特殊标记字符串（如 "<|endoftext|>"）。
            special_tok = match.group()
            
            # 核心原则：特殊标记不参与 BPE 合并！
            # 我们直接将其编码为 UTF-8 字节，然后在词表中查找其 ID。
            # 注意：这些标记在 train_bpe 阶段必须已经被手动加入到了词表中。
            tokens.append(self.byte_to_id[special_tok.encode("utf-8")])
            
            # 3.3 更新游标
            # 将游标移动到当前匹配项的末尾，为寻找下一个片段做准备。
            last_pos = match.end()
            
        # --- 步骤 4: 处理“收尾文本” ---
        # 如果最后一个特殊标记后面还有文字（例如 "Hello<|end|>World" 中的 "World"），
        # 或者整个文本根本没有特殊标记匹配（虽然逻辑上 Case A 已处理，但这里是双重保险），
        # 我们需要处理从 last_pos 到字符串末尾的所有剩余字符。
        remaining_text = text[last_pos:]
        if remaining_text:
            # 剩余部分同样作为普通文本片段进行 BPE 编码。
            tokens.extend(self._encode_text_segment(remaining_text))
            
        # 返回拼接好的所有 ID 列表
        return tokens
