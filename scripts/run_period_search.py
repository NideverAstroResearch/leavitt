#!/usr/bin/env python
"""
Batch period search for NSC DR2 variable stars.

Reads a text file of NSC DR2 object IDs (one per line), runs get_period()
on each using the leavitt Variable class, and writes results to a CSV.

Usage
-----
    python scripts/run_period_search.py samples/golden_RRab.txt
    python scripts/run_period_search.py samples/golden_RRab.txt -o results/rrab_periods.csv
    python scripts/run_period_search.py samples/all_gold_sample.txt --workers 8 --nbins 2000

Output columns: objid, period, n_obs, error
"""

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from leavitt.timeseries import Variable


def process_star(objid, statistic, nbins, sigma):
    try:
        star = Variable(objid, datarelease='dr2', sigma=sigma)
        n_obs = len(star.timeseries)
        period = star.get_period(statistic=statistic, nbins=nbins)
        return objid, period, n_obs, None
    except Exception as e:
        return objid, None, None, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Batch period search for NSC DR2 variable stars.'
    )
    parser.add_argument('input',
        help='Text file with one NSC DR2 object ID per line')
    parser.add_argument('-o', '--output', default=None,
        help='Output CSV (default: <input_stem>_periods.csv next to input file)')
    parser.add_argument('--statistic', default='hybrid',
        choices=['hybrid', 'ls', 'ls_mb', 'lk'],
        help='Periodogram statistic (default: hybrid)')
    parser.add_argument('--nbins', type=int, default=1000,
        help='Frequency grid size for LK/hybrid (default: 1000)')
    parser.add_argument('--sigma', type=float, default=3.0,
        help='Sigma-clipping threshold per band (default: 3.0, 0 to disable)')
    parser.add_argument('--workers', type=int, default=4,
        help='Parallel Data Lab query workers (default: 4)')
    args = parser.parse_args()

    sigma = args.sigma if args.sigma > 0 else None

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    objids = [l.strip() for l in input_path.read_text().splitlines() if l.strip()]
    if not objids:
        print("ERROR: no object IDs found in input file.", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + '_periods.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Period search: {len(objids)} stars → {output_path}")
    print(f"  statistic={args.statistic}  nbins={args.nbins}  "
          f"sigma={sigma}  workers={args.workers}\n")

    n_done = 0
    n_failed = 0

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['objid', 'period', 'n_obs', 'error'])

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_star, objid, args.statistic, args.nbins, sigma): objid
                for objid in objids
            }
            with tqdm(total=len(objids), unit='star') as pbar:
                for future in as_completed(futures):
                    objid, period, n_obs, error = future.result()
                    writer.writerow([
                        objid,
                        f'{period:.6f}' if period is not None else '',
                        n_obs if n_obs is not None else '',
                        error or '',
                    ])
                    f.flush()
                    n_done += 1
                    if error:
                        n_failed += 1
                        tqdm.write(f"FAILED  {objid}: {error}")
                    else:
                        tqdm.write(f"{objid}  P={period:.4f} d  (n_obs={n_obs})")
                    pbar.set_postfix(failed=n_failed)
                    pbar.update(1)

    print(f"\nDone. {n_done - n_failed}/{n_done} succeeded, "
          f"{n_failed} failed → {output_path}")


if __name__ == '__main__':
    main()
