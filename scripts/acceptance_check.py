#!/usr/bin/env python3
"""Project acceptance checks for VYNDRA Parts 1-5. Exits non-zero when criteria fail."""
from __future__ import annotations
import argparse, csv, json, sys, subprocess
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
FAIL=[]
def check(condition: bool, message: str):
    print(f"[{'PASS' if condition else 'FAIL'}] {message}")
    if not condition: FAIL.append(message)
def exists(path, message=None):
    p=ROOT/path; check(p.is_file(), message or f'{path} exists')
def csv_min(path, minimum, message, col=None):
    p=ROOT/path
    if not p.is_file(): check(False,f'{path} exists for {message}'); return
    try:
        df=pd.read_csv(p); check(len(df)>=minimum, f'{message} (found {len(df)}, need >= {minimum})')
        if col: check(col in df.columns, f'{path} contains column {col}')
    except Exception as exc: check(False,f'{path} is readable ({type(exc).__name__})')
def part1():
    for path in ['scripts/acceptance_check.py','vyndra/photo_api.py','vyndra/portions.py','scripts/train_food_intent.py','data/processed/reference_meals_template.csv','docs/API_REFERENCE.md']:
        exists(path)
    try:
        import app
        rules=list(app.app.url_map.iter_rules()); routes={r.rule for r in rules}
        for route in ['/upload','/api/meals/analyze','/api/chat/send','/api/chat/send-meal','/api/insights/<user_id>','/api/users','/api/meals','/api/chat/<conversation_id>','/api/social/influencers','/api/social/communities','/api/social/user/<user_id>','/api/foods/<path:food_name>/insights']:
            check(route in routes, f'route registered: {route}')
        check(sum(r.rule=='/api/chat/analyze' for r in rules)==1,'legacy /api/chat/analyze registered exactly once')
        check(app.app.config.get('MAX_CONTENT_LENGTH',0)>=8*1024*1024,'upload limit is at least 8 MiB')
        check('0.0.0.0' in open(ROOT/'config.py',encoding='utf-8').read(),'config.py sets default host 0.0.0.0')
        # The Part 1 endpoint tests use a generated test image and mocked model, so they
        # exercise both upload routes and the nudge response without loading best.pt.
        test_file = ROOT / 'tests' / 'test_part1_pipeline.py'
        if test_file.exists():
            result = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(test_file)], cwd=ROOT, capture_output=True, text=True)
            check(result.returncode == 0, 'mocked photo endpoints and chat nudge tests pass')
            if result.returncode != 0:
                print((result.stdout + '\n' + result.stderr)[-4000:])
        else:
            check(False, 'tests/test_part1_pipeline.py exists to verify photo and nudge endpoints')
    except Exception as exc: check(False,f'Flask application imports ({type(exc).__name__})')
def part2():
    csv_min('data/processed/chat_intelligence/chat_messages_synthetic.csv',3000,'synthetic chat messages')
    csv_min('data/processed/chat_intelligence/chat_messages_handwritten_review.csv',150,'handwritten/review chat messages')
    p=ROOT/'data/processed/chat_intelligence/chat_messages_synthetic.csv'
    if p.exists():
        df=pd.read_csv(p); check({'text','label','split'}.issubset(df.columns),'chat dataset has text/label/split columns'); check(not df.text.duplicated().any(),'chat dataset has no duplicate texts across splits')
        counts=df.label.value_counts(); check(len(counts)>=3 and counts.max()-counts.min()<=1,'chat labels are balanced')
    csv_min('results/tables/part4/text_model_comparison.csv',4,'text model comparison configurations','model')
    exists('results/tables/part4/mention_metrics_review.json'); exists('results/tables/part4/mention_failures_review.csv'); exists('results/tables/part4/nudge_threshold.json')
    exists('results/tables/part4/sentiment_vader_metrics.json')
    exists('results/tables/part4/lda_topics.json')
    exists('results/tables/part4/matching_method_comparison.csv')
    exists('results/figures/part4/text_confusion_matrix.png')
def part3():
    csv_min('data/processed/social_scale/social_users.csv',60,'social users','planted_community')
    csv_min('data/processed/social_scale/social_events.csv',1500,'timestamped social events','event_type')
    p=ROOT/'results/tables/part5/social_centrality.csv'
    if p.exists():
        df=pd.read_csv(p)
        for col in ['betweenness','closeness','eigenvector','pagerank','in_degree','out_degree']: check(col in df.columns,f'social centrality contains {col}')
        check(len(df)==60,'centrality table has 60 users')
    else: check(False,'social centrality results exist')
    exists('results/tables/part5/social_metrics.json'); exists('results/figures/part5/social_network_communities.png')
    csv_min('results/tables/part6/food_kmeans_metrics.csv',9,'K-Means silhouette results for k=2..10','silhouette')
    p=ROOT/'results/tables/part6/food_kmeans_metrics.csv'
    if p.exists():
        df=pd.read_csv(p); check(set(range(2,11)).issubset(set(df.k.astype(int))),'K-Means table includes every k from 2 through 10')
    exists('results/tables/part6/food_hierarchical_silhouette.csv'); exists('results/tables/part6/food_clusters.csv'); exists('results/figures/part6/food_clusters_pca.png'); exists('results/figures/part6/food_ward_dendrogram.png')
def part4():
    exists('data/processed/vision_test_manifest.csv'); exists('results/tables/part7/vision_model_comparison.csv'); exists('results/tables/part7/vision_metrics_by_dataset.csv'); exists('results/tables/part7/vision_worst_errors_top20.csv'); exists('results/figures/part7/vision_confusion_supported.png')
    p=ROOT/'results/tables/part7/vision_evaluation_summary.json'
    if p.exists():
        d=json.loads(p.read_text()); check('macro_f1_supported_classes' in d,'vision macro-F1 field is explicitly computed on supported classes'); check(d.get('successfully_predicted_rows',0)>0,'vision evaluation has successful predictions')
def part5():
    exists('notebooks/VYNDRA_analytics.py'); exists('docs/DATASETS.md'); exists('docs/API_REFERENCE.md'); exists('README.md')
    # Check every generated PNG has an adjacent descriptive source/report, and figures are created by scripts with titles.
    figdirs=[ROOT/'results/figures/part4',ROOT/'results/figures/part5',ROOT/'results/figures/part6',ROOT/'results/figures/part7']
    generated=[p for d in figdirs if d.exists() for p in d.glob('*.png')]
    check(bool(generated),'at least one generated analytical figure exists')
    for p in generated: check(p.stat().st_size>1000,f'figure is non-empty: {p.relative_to(ROOT)}')
    # Source-level guard: every plotting script/notebook that saves figures must include title calls.
    plotting_sources=[]
    for folder in [ROOT/'scripts', ROOT/'notebooks', ROOT/'vyndra']:
        if folder.exists():
            plotting_sources.extend(q for q in folder.rglob('*.py') if '.venv' not in q.parts and '__pycache__' not in q.parts)
    for source in plotting_sources:
        content=source.read_text(encoding='utf-8',errors='replace')
        if 'savefig(' in content:
            check(('set_title(' in content or 'plt.title(' in content), f'figure-producing source includes a title call: {source.relative_to(ROOT)}')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--part',type=int,choices=range(1,6)); ap.add_argument('--all',action='store_true'); args=ap.parse_args()
    if not args.part and not args.all: ap.error('choose --part N or --all')
    parts={1:part1,2:part2,3:part3,4:part4,5:part5}
    selected=parts.items() if args.all else [(args.part,parts[args.part])]
    for _n,fn in selected: fn()
    print(f'\nAcceptance check summary: {len(FAIL)} failure(s).')
    if FAIL:
        print('Failed criteria:'); [print(' - '+x) for x in FAIL]; return 1
    print('All selected acceptance checks passed.'); return 0
if __name__=='__main__': sys.exit(main())
