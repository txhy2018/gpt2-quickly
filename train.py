import tensorflow as tf
from transformers import GPT2Config
from transformers import TFGPT2LMHeadModel
from transformers import XLNetTokenizer
from transformers import BertTokenizer
from transformers.modeling_tf_utils import shape_list
import configs
import random
import click
import time
import pickle
from pathlib import Path
import numpy as np
import gc
from tqdm import tqdm

# from tensorflow.keras.mixed_precision import experimental as mixed_precision


# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)


def load_tokenizer() -> BertTokenizer:
    # 1.1、使用
    tokenizer = BertTokenizer.from_pretrained(
        configs.data.path,  # 文件路径：./dataset/test/
        max_len=configs.model.max_length,  # 模型最大词长度
        add_special_token=False
    )
    tokenizer.return_attention_mask = None
    return tokenizer


def get_dataset() -> tf.data.Dataset:
    # 2.1、加载文件：./dataset/test/pickle/路径下处理好的训练文件，这个是经过分词器处理好，可以输给GPT-2模型训练的数据格式
    p = Path(configs.data.pickle_path)
    s_pickle_files = list(p.glob("*.s.pickle"))
    m_pickle_files = list(p.glob("*.m.pickle"))
    l_pickle_files = list(p.glob("*.l.pickle"))
    # 2.2、根据预设数量对训练文件进行分组
    s_group_num = 8
    m_group_num = 4
    l_group_num = 2
    s_pickle_files = [
        (s_pickle_files[i: i + s_group_num], 's')
        for i in range(0, len(s_pickle_files), s_group_num)
    ]
    m_pickle_files = [
        (m_pickle_files[i: i + m_group_num], 'm')
        for i in range(0, len(m_pickle_files), m_group_num)
    ]
    l_pickle_files = [
        (l_pickle_files[i: i + l_group_num], 'l')
        for i in range(0, len(l_pickle_files), l_group_num)
    ]
    pickle_files = s_pickle_files + m_pickle_files + l_pickle_files
    random.shuffle(pickle_files)  # 随机打乱文件
    
    # 2.3、遍历分组后的文件
    for (sub_pickle_files, size) in pickle_files:
        input_ids = []
        # 2.4、遍历分组后的子文件
        for pickle_file in sub_pickle_files:
            print(f"loading {pickle_file}")
            # 2.5、读取每个文件，并遍历每一行样本
            pickle_datas = list(
                open(pickle_file, "rb").read().split("换行".encode()))
            bad_count = 0
            for line in tqdm(pickle_datas):
                if line:
                    ids = pickle.loads(line)
                    # if ids.shape[-1] != configs.model.max_length:
                    #     bad_count += 1
                    #     continue
                    for row in ids:
                        input_ids.append(row)
            print("bad ids count: ", bad_count)
        input_ids = np.array(input_ids)
        
        # 2.6、分词后的数据，处理成tensorflow数据格式
        ids = input_ids[:, :-1]
        labels = input_ids[:, 1:]
        # ids = ids.astype('int32')
        # labels = ids.astype('int32')
        # print(ids, labels)
        print(ids.shape, labels.shape, ids.dtype, labels.dtype)
        if size == 'l':
            batch_size = 32
        if size == 'm':
            batch_size = 48
        if size == 's':
            batch_size = 64
        dataset = (
            tf.data.Dataset.from_tensor_slices((ids, labels))
            .shuffle(ids.shape[0], reshuffle_each_iteration=False)
            .repeat()
            .batch(batch_size)
        )
        yield len(input_ids), dataset


def build_loss(tokenizer):
    def custom_loss(labels, logits):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        # make sure only labels that are not equal to -100 affect the loss
        active_loss = tf.not_equal(tf.reshape(labels, (-1,)), tokenizer.pad_token_id)
        reduced_logits = tf.boolean_mask(
            tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss
        )
        labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
        return loss_fn(labels, reduced_logits)

    return custom_loss


class CustomAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

    def __init__(self, *args, tokenizer=None, **kwargs):
        self.tokenizer = tokenizer
        super(CustomAccuracy, self).__init__(*args, **kwargs)

    def update_state(self, labels, logits, sample_weight=None):
        active_loss = tf.not_equal(tf.reshape(labels, (-1,)), self.tokenizer.pad_token_id)
        reduced_logits = tf.boolean_mask(
            tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss
        )
        labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
        return super().update_state(labels, reduced_logits, sample_weight)


def init_model(
    tokenizer: BertTokenizer,
    model_path: str = configs.model_path,
) -> TFGPT2LMHeadModel:

    try:  # 3.1、存在模型时，则直接加载：./dataset/models/tf_model.h5
        model = TFGPT2LMHeadModel.from_pretrained(
            model_path, return_dict=False)
         # 3.2、不存在时则使用 TFGPT2LMHeadModel 进行初始化获取模型结构 model 
    except EnvironmentError:
        config = GPT2Config(
            architectures=["TFGPT2LMHeadModel"],
            model_type="TFGPT2LMHeadModel",
            tokenizer_class="XLNetTokenizer",
            vocab_size=tokenizer.vocab_size,
            n_positions=configs.model.n_positions,
            n_ctx=configs.model.n_ctx,
            n_embd=configs.model.n_embd,
            n_layer=configs.model.n_layer,
            n_head=configs.model.n_head,
            pad_token_id=tokenizer.pad_token_id,
            task_specific_params={
                "text-generation": {"do_sample": True, "max_length": 120}
            },
            return_dict=False,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
        )
        model = TFGPT2LMHeadModel(config)
    # 3.3、自定义损失函数
    loss = build_loss(tokenizer)
    # loss = model.compute_loss
    # 3.4、优化器设置
    lr = 5e-5
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # 3.5、评估器设置
    metric = CustomAccuracy("accuracy", tokenizer=tokenizer)
    # 3.6、编译器设置
    model.compile(
        optimizer=optimizer,
        loss=[loss, *[None] * model.config.n_layer],
        metrics=[metric],
    )
    # 3.7、返回模型 model
    return model


def train(model: TFGPT2LMHeadModel, train_dataset, epochs: int, train_steps: int):
    # 4.1、自定义自动保存模型，防止程序中断白训练了
    class AutoSaveCallback(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs=None):
            self.model.save_pretrained(f"{configs.model_path}")  # 保存预训练模型：./dataset/models/，会自动保存为：config.json  tf_model.h5
    # 4.2、自定义反馈功能
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=f"{configs.model_path}/logs", update_freq=50  # 保存日志
        ),
        AutoSaveCallback(),
    ]

    t1 = time.time()
    # 4.3、模型训练
    model.fit(
        train_dataset,  # 训练集
        epochs=epochs,
        steps_per_epoch=train_steps,
        callbacks=callbacks,  # 添加反馈功能
        batch_size=None,
    )
    print(f"total train time {time.time() - t1}")


@click.command()
@click.option('--epochs', default=4, help='number of epochs')
@click.option('--train_steps', default=500, help='number of train_steps')
def main(epochs, train_steps):
    # 1、加载分词器
    tokenizer = load_tokenizer()
    
    # 2、获取数据集
    for _total_num, train_dataset in get_dataset():
        # 3、初始化模型
        model = init_model(
            tokenizer, 
            configs.model_path,  # ./dataset/models/
        )
        # 4、模型预训练
        train(model, train_dataset, epochs, train_steps)
        del train_dataset
        del model
        gc.collect()


if __name__ == '__main__':
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        main()
