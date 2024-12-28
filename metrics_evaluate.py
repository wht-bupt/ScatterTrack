"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.
Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
Modified by Rufeng Zhang
"""

import argparse
import glob
import os
import logging
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path
from sda_evaluation import sda_evaluation
from yacs.config import CfgNode as CN


def parse_args():


    parser = argparse.ArgumentParser(description="""
Compute metrics for trackers using MOTChallenge ground-truth data.
Files
-----
All file content, ground truth and test files, have to comply with the
format described in 
Milan, Anton, et al. 
"Mot16: A benchmark for multi-object tracking." 
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/
Structure
---------
Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt/gt.txt
    <GT_ROOT>/<SEQUENCE_2>/gt/gt.txt
    ...
Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...
Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--score_threshold', type=float, help='Score threshold', default=-1)
    parser.add_argument('--eval_official', action='store_true')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, help='LAP solver to use', default='scipy')
    parser.add_argument('--evaluation_factors', '--ef', type=str, help='choose different evaluation factors', choices=['mota', 'mts', 'sda'], required=True)
    parser.add_argument('--track_results', '--tr', type=str, help='choose different track results', required=True)
    return parser.parse_args()


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logging.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, distth=0.99))
            names.append(k)
        else:
            logging.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


if __name__ == '__main__':

    args = parse_args()
    args.eval_official = True
    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver

    if args.evaluation_factors == 'mota':
        groundtruths = 'data/' + args.evaluation_factors + '_gts'
        tests = args.track_results + '/tracks_' + args.evaluation_factors

        gtfiles = glob.glob(os.path.join(groundtruths, '*.txt'))
        tsfiles = [f for f in glob.glob(os.path.join(tests, '*.txt')) if not os.path.basename(f).startswith('eval')]

        logging.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
        logging.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
        logging.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
        logging.info('Loading files.')

        gt = OrderedDict([(Path(f).parts[-1][:3], mm.io.loadtxt(f, fmt=args.fmt, min_confidence=1)) for f in gtfiles])
        ts = OrderedDict(
            [(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=args.fmt, min_confidence=args.score_threshold))
            for f in tsfiles])

        mh = mm.metrics.create()
        accs, names = compare_dataframes(gt, ts)

        logging.info('Running metrics')
        metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
                'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
                'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
        summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
        div_dict = {
            'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
            'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
        for divisor in div_dict:
            for divided in div_dict[divisor]:
                summary[divided] = (summary[divided] / summary[divisor])
        fmt = mh.formatters
        change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                        'partially_tracked', 'mostly_lost']
        for k in change_fmt_list:
            fmt[k] = fmt['mota']
        print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))
        if args.eval_official:
            metrics = mm.metrics.motchallenge_metrics + ['num_objects']
            summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
            print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
            logging.info('Completed')

    elif args.evaluation_factors == 'mts':
        groundtruths = 'data/' + args.evaluation_factors + '_gts'
        tests = args.track_results + '/tracks_' + args.evaluation_factors
    

        gtfiles = glob.glob(os.path.join(groundtruths, '*.txt'))
        tsfiles = [f for f in glob.glob(os.path.join(tests, '*.txt')) if not os.path.basename(f).startswith('eval')]

        logging.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
        logging.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
        logging.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
        logging.info('Loading files.')

        gt = OrderedDict([(Path(f).parts[-1][:3], mm.io.loadtxt(f, fmt=args.fmt, min_confidence=1)) for f in gtfiles])
        ts = OrderedDict(
            [(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=args.fmt, min_confidence=args.score_threshold))
            for f in tsfiles])

        mh = mm.metrics.create()
        accs, names = compare_dataframes(gt, ts)

        logging.info('Running metrics')
        metrics = ['num_unique_objects', 'mostly_tracked', 'partially_tracked', 'mostly_lost']
        summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
        div_dict = {'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
        for divisor in div_dict:
            for divided in div_dict[divisor]:
                summary[divided] = (summary[divided] / summary[divisor])
        fmt = mh.formatters        
        change_fmt_list = ['mostly_tracked', 'partially_tracked', 'mostly_lost']
        for k in change_fmt_list:
            fmt[k] = fmt['mota']
        print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))
        logging.info('Completed')
    else:
        track_result_path = os.path.join(args.track_results, 'tracks')
        gt_data_path = 'data/gts'
        default_yaml = 'config/scatternet.yaml'
        default_cfg = open(default_yaml)
        cfg = CN.load_cfg(default_cfg)
        cfg.freeze

        sda_evaluation(cfg, track_result_path, gt_data_path)
