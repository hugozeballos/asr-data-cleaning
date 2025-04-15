import multiprocessing as mp
from train_titan import train

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    train()
