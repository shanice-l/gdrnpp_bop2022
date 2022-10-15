# https://github.com/fxia22/egl_example/blob/master/get_available_devices.py
import subprocess
import os
import torch


def get_available_devices():
    executable_path = os.path.join(os.path.dirname(__file__), "build")

    num_devices = int(subprocess.check_output(["{}/query_devices".format(executable_path)]))

    available_devices = []
    for i in range(num_devices):
        try:
            if b"NVIDIA" in subprocess.check_output(["{}/test_device".format(executable_path), str(i)]):
                available_devices.append(i)
        except subprocess.CalledProcessError as e:
            print(e)
    return available_devices


if __name__ == "__main__":
    runs = 50
    from tqdm import tqdm
    import time

    t0 = time.perf_counter()
    for i in tqdm(range(runs)):
        get_available_devices()
    dt = time.perf_counter() - t0
    print("get available devices: {}s {}fps".format(dt / runs, runs / dt))

    t0 = time.perf_counter()
    for i in tqdm(range(runs)):
        torch.cuda.device_count()
    dt = time.perf_counter() - t0
    print("pytorch: {}s {}fps".format(dt / runs, runs / dt))

    print(get_available_devices())

    print(torch.cuda.device_count())
    """
    get available devices: 0.3493817949295044s 2.8621983586802866fps
    pytorch: 0.0006400346755981445s 1562.4153473645fps
    """
