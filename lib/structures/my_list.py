import numpy as np
import torch


class MyList(list):
    def __getitem__(self, index):
        """support indexing using torch.Tensor."""
        if isinstance(index, torch.Tensor):
            if isinstance(index, torch.BoolTensor):
                return [self[i] for i, idx in enumerate(index) if idx]
            else:
                return [self[int(i)] for i in index]
        elif isinstance(index, (list, tuple)):
            if len(index) > 0 and isinstance(index[0], bool):
                return [self[i] for i, idx in enumerate(index) if idx]
            else:
                return [self[int(i)] for i in index]
        elif isinstance(index, np.ndarray):
            if index.dtype == np.bool:
                return [self[i] for i, idx in enumerate(index) if idx]
            else:
                return [self[int(i)] for i in index]

        return list.__getitem__(self, index)


if __name__ == "__main__":
    a = [None, "a", 1, 2.3]
    a = MyList(a)
    print(a)
    print(type(a), isinstance(a, list))
    print("\ntorch bool index")
    index = torch.tensor([True, False, True, False])
    print(index)
    print(a[index])

    print("torch int index")
    index = torch.tensor([0, 2, 3])
    print(index)
    print(a[index])

    print("\nnumpy bool index")
    index = np.array([True, False, True, False])
    print(index)
    print(a[index])

    print("numpy int index")
    index = np.array([0, 2, 3])
    print(index)
    print(a[index])

    print("\nlist bool index")
    index = [True, False, True, False]
    print(index)
    print(a[index])

    print("list int index")
    index = [0, 2, 3]
    print(index)
    print(a[index])

    print("\ntuple bool index")
    index = (True, False, True, False)
    print(index)
    print(a[index])

    print("tuple int index")
    index = (0, 2, 3)
    print(index)
    print(a[index])

    # print(a[1:-1])
