import os
import gzip
import json
import argparse
import numpy as np
from os import listdir
from operator import itemgetter 
from os.path import isfile, join

def main(args):
    path = './datasets/{}/train/episodes'
    save_path = args.save_path
    if args.is_pointnav:
        val = 'robothor-pointnav'
        path = path.format(val)
        save_path = save_path + val
    else:
        val = 'robothor-objectnav'
        path = path.format(val)
        save_path = save_path + val
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    files = [f for f in listdir(path) if isfile(join(path, f))]

    for f in files:
        with gzip.open(join(path, f), "rb") as ep:
            data = json.loads(ep.read(), encoding="utf-8")
            int_samples = list(np.random.randint(len(data), size=args.sample_sz))
            sampled_values = itemgetter(*int_samples)(data)
            
            with gzip.open(f"{save_path}/{f}", 'w') as fout:
                js = json.dumps(sampled_values, indent=4)
                js_bytes = js.encode('utf-8')
                fout.write(js_bytes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")

    parser.add_argument(
        "--is_pointnav",
        nargs="?",
        type=bool,
        default=True,
        help="Determine whether should sample from point nav episodes or object nav"
    )

    parser.add_argument(
        "--sample_sz",
        nargs="?",
        type=int,
        default=100,
        help="How many episodes to sample from episodic json"
    )

    parser.add_argument(
        "--save_path",
        nargs="?",
        type=str,
        default='./episodes/',
        help="Folder to save episode jsons"
    )

    args = parser.parse_args()
    main(args)
