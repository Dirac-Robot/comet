#!/usr/bin/env python3
"""Recall-frequency report from the persistent `memory.recall` journal.

Answers "did recall frequency / trigger-channel share change before vs after a
given time?" — e.g. the instruction-form trigger rewrite (a4e02f8, CoMeT).

The journal lives in CoBrA's structured logs (the CoMeT retriever logs through
loguru; the CoBrA sink captures `extra`). Each retrieval emits one line with
event='memory.recall', n_recalled, node_ids, n_trigger_matched, trigger_matched.

Usage:
    python scripts/recall_freq_report.py --since '2026-06-12T15:41:00+09:00' \
        [--logs ~/.cobra/logs] [--nodes ~/.cobra/store/nodes]

Splits recall events at --since and reports, for each side:
  - recall events / day
  - mean nodes recalled per event
  - trigger-channel share (n_trigger_matched / n_recalled)
If --nodes is given, also splits trigger-matched nodes by trigger FORM
(instruction-form "… → open raw" vs legacy "When I …") so you can see whether
the new form actually wins more trigger-channel recalls.

Caveat: recall is query-driven, so day-to-day workload (what was being worked
on) is a heavy confound — read the trigger-FORM split, not the raw counts, as
the cleaner signal, and only over windows with comparable activity.
"""
import argparse
import datetime as dt
import glob
import json
import os
import re


def _parse_iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def _node_trigger_form(nodes_dir: str) -> dict:
    """node_id -> 'instruction' | 'legacy' | 'other', by trigger text shape."""
    form = {}
    for f in glob.glob(os.path.join(nodes_dir, '*.json')):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        trig = (d.get('trigger') or '').strip()
        nid = d.get('node_id') or os.path.basename(f)[:-5]
        if '→' in trig or 'open raw' in trig.lower():
            form[nid] = 'instruction'
        elif re.match(r'\s*when\b', trig, re.I):
            form[nid] = 'legacy'
        else:
            form[nid] = 'other'
    return form


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', required=True, help='ISO split time, e.g. 2026-06-12T15:41:00+09:00')
    ap.add_argument('--logs', default=os.path.expanduser('~/.cobra/logs'))
    ap.add_argument('--nodes', default=os.path.expanduser('~/.cobra/store/nodes'))
    args = ap.parse_args()

    split = _parse_iso(args.since)
    tz = split.tzinfo or dt.timezone.utc
    form = _node_trigger_form(args.nodes) if os.path.isdir(args.nodes) else {}

    pre = {'events': 0, 'nodes': 0, 'trig': 0, 'days': set(), 'trig_form': {}}
    post = {'events': 0, 'nodes': 0, 'trig': 0, 'days': set(), 'trig_form': {}}

    for path in glob.glob(os.path.join(args.logs, '*', 'cobra_structured.jsonl')):
        for line in open(path, errors='ignore'):
            if '"memory.recall"' not in line:
                continue
            try:
                rec = json.loads(line)['record']
            except Exception:
                continue
            ex = rec.get('extra', {})
            if ex.get('event') != 'memory.recall':
                continue
            ts = rec.get('time', {}).get('timestamp')
            if ts is None:
                continue
            t = dt.datetime.fromtimestamp(ts, tz=tz)
            bucket = post if t >= split else pre
            bucket['events'] += 1
            bucket['nodes'] += ex.get('n_recalled', 0) or 0
            bucket['trig'] += ex.get('n_trigger_matched', 0) or 0
            bucket['days'].add(t.strftime('%Y-%m-%d'))
            for nid in ex.get('trigger_matched', []) or []:
                k = form.get(nid, 'unknown')
                bucket['trig_form'][k] = bucket['trig_form'].get(k, 0) + 1

    def show(label, b):
        ev, nodes, trig = b['events'], b['nodes'], b['trig']
        ndays = max(len(b['days']), 1)
        print(f'\n{label}:  events={ev}  days={len(b["days"])}')
        if ev:
            print(f'  recall events/day   : {ev / ndays:.1f}')
            print(f'  mean nodes/recall   : {nodes / ev:.2f}')
            print(f'  trigger-channel share: {100 * trig / max(nodes, 1):.1f}%  ({trig}/{nodes})')
            if b['trig_form']:
                tot = sum(b['trig_form'].values())
                forms = '  '.join(f'{k}={v} ({100*v/tot:.0f}%)' for k, v in sorted(b['trig_form'].items()))
                print(f'  trigger-matched by FORM: {forms}')

    print(f'Split at {split.isoformat()}')
    show('PRE  (legacy trigger form)', pre)
    show('POST (instruction trigger form)', post)
    if not pre['events'] and not post['events']:
        print('\nNo memory.recall events found — the journal lands only on retrievals '
              'AFTER this build is live (daemon reload). Re-run once a window has accrued.')


if __name__ == '__main__':
    main()
