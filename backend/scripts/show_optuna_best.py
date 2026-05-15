"""Inspecciona Optuna study: best params + top 5 trials."""
import optuna

s = optuna.load_study(
    study_name='ppo_hpo_tfm',
    storage='sqlite:///hpo/optuna_study_ppo.db',
)

completed = [t for t in s.trials if t.state.name == 'COMPLETE']
pruned = [t for t in s.trials if t.state.name == 'PRUNED']
print('=' * 60)
print(f'Trials totales: {len(s.trials)}')
print(f'  Completed:    {len(completed)}')
print(f'  Pruned:       {len(pruned)}')
print('=' * 60)
print(f'Best Sharpe: {s.best_value:.4f}')
print(f'Best trial:  #{s.best_trial.number}')
print('Best params:')
for k, v in s.best_params.items():
    if isinstance(v, float):
        print(f'  {k:20s} = {v:.6f}')
    else:
        print(f'  {k:20s} = {v}')

print('=' * 60)
print('Top 5 trials por Sharpe:')
top5 = sorted(completed, key=lambda x: -x.value)[:5]
for t in top5:
    print(f'  Trial #{t.number:3d}  Sharpe={t.value:.4f}')

print('=' * 60)
print('Estadísticas Sharpe (completed only):')
import numpy as np
sharpes = np.array([t.value for t in completed])
print(f'  Media:  {sharpes.mean():.4f}')
print(f'  Std:    {sharpes.std():.4f}')
print(f'  Min:    {sharpes.min():.4f}')
print(f'  Max:    {sharpes.max():.4f}')
