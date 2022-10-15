import os
import detectron2.utils.comm as comm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
import pickle

try:
    import horovod.torch as hvd
except (ModuleNotFoundError, ImportError):
    print("You requested to import horovod which is missing or not supported for your OS.")
    HVD_AVAILABLE = False
else:
    HVD_AVAILABLE = True

global _USE_PT
global _USE_HVD
global _USE_PL
global _PL_LOCAL_RANK
_USE_PT = False
_USE_HVD = False
_USE_PL = False
_PL_LOCAL_RANK = 0


def reduce_dict(input_dict, average=True):
    global _USE_HVD
    if _USE_HVD:
        return reduce_dict_hvd(input_dict, average=average)
    else:
        return comm.reduce_dict(input_dict, average=average)


def reduce_dict_hvd(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    global _USE_HVD
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        if _USE_HVD:  # NOTE: hvd
            hvd.allreduce_(
                values,
                op=hvd.Average if average else hvd.Adasum,
                name="reduce_dict",
            )
        else:
            dist.all_reduce(values)
            if average:
                values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def all_gather(data, group=None):
    global _USE_HVD

    if _USE_HVD:
        return all_gather_hvd(data, group=group)
    else:
        return comm.all_gather(data, group=group)


gather = comm.gather


def synchronize():
    global _USE_HVD
    if _USE_HVD:
        hvd.broadcast_object(0)
        return
    return comm.synchronize()


def all_gather_hvd(data, group=None):
    global _USE_HVD
    assert _USE_HVD, f"_USE_HVD: {_USE_HVD}"
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list]
    if _USE_HVD:
        # NOTE: concatenated on the first dimension
        tensor_list = hvd.allgather(
            tensor[
                None,
            ]
        )
    else:
        dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def _serialize_to_tensor(data, group):
    global _USE_HVD
    if _USE_HVD:
        backend = "nccl"
    else:
        backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024**3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024**3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    global _USE_HVD
    if _USE_HVD:
        world_size = get_world_size()
    else:
        world_size = dist.get_world_size(group=group)
    assert world_size >= 1, "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)]
    if _USE_HVD:
        size_list = hvd.allgather(local_size)  # a tensor with (world_size,) actually
    else:
        dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def init_dist(launcher, backend="nccl", **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    if launcher == "pytorch":
        init_dist_pytorch(backend, **kwargs)
    elif launcher == "hvd":
        init_hvd()
    else:
        raise ValueError(f"Invalid launcher type: {launcher}")


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def init_dist_env_variables(args):
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)


def init_dist_pytorch(num_gpus_per_machine, num_machines=1, backend="nccl", **kwargs):
    global _USE_PT
    if _USE_PT:
        return True
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)
    _USE_PT = True
    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()
    return True


def init_hvd():
    global _USE_HVD
    if _USE_HVD:
        return True
    if not HVD_AVAILABLE:
        raise RuntimeError("horovod is not available")
    else:
        hvd.init()
        _USE_HVD = True
        assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
        # Horovod: pin GPU to local rank.
        local_rank = get_local_rank()
        assert local_rank < torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        return True


def is_dist_avail_and_initialized():
    global _USE_HVD
    if _USE_HVD:
        return True
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def shared_random_seed():
    return comm.shared_random_seed()


def get_world_size():
    global _USE_HVD
    if _USE_HVD:
        return hvd.size()
    else:
        return comm.get_world_size()


def get_rank():
    if _USE_HVD:
        return hvd.rank()
    else:
        return comm.get_rank()


def get_local_rank():
    if _USE_HVD:
        return hvd.local_rank()
    elif _USE_PT:
        return int(os.environ.get("LOCAL_RANK", "0"))
    elif _USE_PL:
        return _PL_LOCAL_RANK
    else:
        return comm.get_local_rank()


def init_pl_local_rank(local_rank=0):
    # NOTE: need to be manually set with self.local_rank in pl
    global _USE_PL
    global _PL_LOCAL_RANK
    _USE_PL = True
    _PL_LOCAL_RANK = local_rank


def get_local_size():
    global _USE_PT
    global _USE_HVD
    if _USE_HVD:
        return hvd.local_size()
    elif _USE_PT:
        return torch.cuda.device_count()
    else:
        return comm.get_local_size()


def is_main_process():
    return get_rank() == 0
