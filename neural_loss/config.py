from dataclasses import dataclass, field
from typing  import Union


@dataclass
class LossModelConfig:
    # 关于训练的参数
    lr: float = 0.001                   # *学习率
    batch_size: int = 32                # *训练批量大小
    eval_batch_size: int = 128          # *验证和测试的批量大小
    epochs: int = 20                    # *训练迭代次数

    # 关于数据的参数
    min_seq_len: int = 40                # **生成数据最小序列长度
    max_seq_len: int = 50                # **生成数据最大序列长度
    num_seqs: int = 1000                 # *_训练_数据中每个长度生成的序列数量
    num_seqs_val: int = 100            # *_验证_数据中每个长度生成的序列数量
    num_seqs_test: int =100            # *_测试_数据中每个长度生成的序列数量
    repeat: int =10                     # *生成训练数据时，每对数据随机重复次数
    random_len: bool = False            # *生成数据时，序列长度是否按概率随机生成
    target_type: Union[int, str] = 3    # *目标序列取值类型，可选 [full, 3, 5]。full表示[1,2,...,max_len]，2表示[0,1]，3表示[0,1,2]，5表示[0,1,2,3,4]

    # 关于模型的参数
    embed_num: int = field(init=False)  # 模型需要embedding的目标元素的数量，取值为DataConfig.target_type + 1
    padd_idx: int = field(init=False)   # 生成数据中的padding符号，取值为DataConfig.target_type
    model_dim: int = 64                 # *模型维度
    num_heads: int = 8                  # *头数
    num_layers: int = 2                 # *层数
    out_active: str =  'relu'        # 模型输出层激活函数，可选sigmoid, tanh, relu
    long_output: bool = True

    # 关于近似指标的参数
    metric: str = 'dcg'            # 指标名称：avg_recall 或 dcg
    at_k: Union[int, None] = 10        # average_recall@k中的k

    def __post_init__(self):
        assert self.metric in ['avg_recall', 'dcg'], 'metric name must be avg_recall or dcg'
        if self.target_type == 'full':
            self.embed_num = self.max_seq_len
            self.padd_idx = None
        else:
            self.embed_num = self.target_type + 1
            self.padd_idx = self.target_type

        if self.metric == 'dcg':
            self.at_k = None
