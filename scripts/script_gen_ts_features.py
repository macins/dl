import argparse
import multiprocessing
from src.feature_gen import TSFeatureGen, ta_map
from config import DataConfig
from tqdm.auto import tqdm
import gc

data_cfg = DataConfig()
SYMBOLS = data_cfg.SYMBOLS

def gen_one_feature(symbol, name, use_f32=True):
    fg = TSFeatureGen(symbol=symbol, need_save=True, use_f32=use_f32)
    fg.gen(name)
    return

def run_feature_gen(symbols, names, max_processes=1, use_f32=True):
    processes = []
    for n in names:
        for s in tqdm(symbols, desc=f"Generating {n}..."):
            if len(processes) >= max_processes:
                for p in processes:
                    p.join()
                processes = [p for p in processes if p.is_alive()]
                
            process = multiprocessing.Process(target=gen_one_feature, args=(s,n,use_f32))
            processes.append(process)
            process.start()

            gc.collect()
    return

def parse_args():
    parser = argparse.ArgumentParser(description="Generate time series features.")
    parser.add_argument(
        "--max_processes", type=int, default=1, help="Maximum number of concurrent processes."
    )
    parser.add_argument(
        "--use_f64", action="store_false", help="Whether to use float64 to store features."
    )
    return parser.parse_args()
    
if __name__ == "__main__":
    symbols = SYMBOLS
    names = list(ta_map.keys())[11:]
    # names = ["mom"]
    args = parse_args()
    use_f32 = not args.use_f64
    
    run_feature_gen(symbols, names, args.max_processes, use_f32)