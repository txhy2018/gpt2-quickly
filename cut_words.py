import configs
import jieba
from tqdm import tqdm
import click
from typing import List
import os
from multiprocessing import Process, Manager, Queue
from pathlib import Path


def cut_words(processer_num, text, result_dict):
    with open(os.path.join(configs.data.path, f'raw.cut.temp.{processer_num}.txt'), 'w') as out_f:
        texts = text.split('\n')
        for line in tqdm(texts):
            try:
                cuts = " ".join(jieba.cut(line))  # jieba分割文本，分割后以空格隔开
                out_f.write(cuts+'\n')  # 保存分割后文本，以\n隔开
            except UnicodeDecodeError:
                pass
            except KeyError:
                pass
            except Exception as e:
                pass


def multiply_cut(handler, tasks):
    manager = Manager()
    result_dict = manager.dict()  # didn't work and don't know why
    jobs = []
    for processer_num, task in enumerate(tasks):  # 多进程任务
        p = Process(target=handler, args=(
            processer_num, task, result_dict))
        jobs.append(p)

    for job in jobs:
        job.start()

    for job in jobs:
        job.join()

    for job in jobs:
        try:
            job.close()  # It may raise exception in python <=3.6
        except:
            pass
    print("[all_task done]")


def split_data(
        text,
        n_processes
) -> List[str]:
    text_task = []
    num_pre_task = len(text) // n_processes
    for i in range(0, len(text), num_pre_task):
        text_task.append(text[i: i + num_pre_task])  # 将文本分为 num_pre_task 段文本
    return text_task


@click.command()
@click.option('--n_processes', default=1, help='Number of processes.')
def preprocess(n_processes):
    print(f'reading {configs.data.raw}')
    with open(configs.data.raw, 'r') as f:  # 1、读取原始文本数据集
        data = f.read().replace('  ', ' ').replace('\n\n', '\n')
        print(f"total words: {len(data)}")

    print(f"split data into {n_processes} pieces")
    text_task = split_data(data, n_processes)  # 2、将一个文本字符串分为几段

    multiply_cut(cut_words, text_task)  # 3、将分段文本进行多进程jieba分割，并分别保存到raw.cut.temp.{processer_num}.txt文件，加快速度

    path = Path(configs.data.path)
    with open(configs.data.raw_cut, 'w') as all_cut_file:  # 4、汇总合并多进程保存的分割文本文件，以\n换行符隔开
        for filename in path.glob('raw.cut.temp.*'):  # 读取多进程分割后的文件
            with open(filename) as cut_file:
                all_cut_file.write(cut_file.read()+'\n')
                print(f'dropping {filename}')
                os.system(f'rm {filename}')  # 删除多进程分割后的文件


if __name__ == '__main__':
    preprocess()
