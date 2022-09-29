# 代码修改

## import部分：

```python
17行 from torch.nn.parallel import DistributedDataParallel
20行 from utilities.distribute import is_main_process, init_distributed_mode
21行 from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, BatchSampler
31行 import warnings #防止提示warnings，可不加
32行 from tqdm import tqdm
```



## main函数部分：

```python
119行 warnings.filterwarnings("ignore") #防止提示warnings，可不加
142-145行 parser增加三个参数
147行 init_distributed_mode(f_args)
148行 os.environ["CUDA_VISIBLE_DEVICES"] = str(f_args.gpu)
189-190行 model转入多GPU模式
216行 增加数据拼接变量 weak_and_syn_data = ConcatDataset([weak_data, syn_data])
218-233行 增加多GPU模式选项
254-255行 增加多GPU下打乱数据功能
268-277行 保存模型部分，增加多GPU模式选项，并在第一块GPU上保存模型（if is_main_process():）
279行 else选项的None改成了torch.device(f_args.gpu)
state = torch.load(model_path, map_location=torch.device("cpu") if not torch.cuda.is_available() else torch.device(f_args.gpu))
280-283行 增加多GPU模式选项
```

## 新增修改：

```python
218-226行 删除BatchSampler部分，删除val_loader和test_loader中的sampler参数，同时增加pin_memory=True参数
251行 改成weak_and_syn_data_sampler.set_epoch(epoch)
```

