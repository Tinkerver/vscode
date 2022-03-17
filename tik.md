```python
block_bite_size = 32 # block是几个字节
dtype_bytes_size = 8 # 数据类型大小
data_each_block = block_bite_size // dtype_bytes_size # 一个block能放多少数据
self.ub_tensor_size = (
            ub_size_bytes // dtype_bytes_size // 2 // self.data_each_block *
            self.data_each_block) # 一块unified buffer能存多少个数据
self.input_num = functools_reduce(lambda x, y: x * y, self.shape_x) #一共会输入多少个数据
self.data_num_each_core = self.input_num // self.aicore_num # 均分任务，每个核心处理多少个数
self.vector_mask_max = 8 * self.data_each_block # 每次repeat最多算八个block，一个repeat能算多少个数