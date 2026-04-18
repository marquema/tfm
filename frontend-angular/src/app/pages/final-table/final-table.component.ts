import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-final-table',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './final-table.component.html',
  styleUrl: './final-table.component.scss'
})
export class FinalTableComponent implements OnInit {
  loading = false;
  errorMessage = '';
  data: any = null;

  metricColumns = [
    { key: 'Sharpe Ratio',               label: 'Sharpe',      format: '1.2-3' },
    { key: 'Sortino Ratio',              label: 'Sortino',     format: '1.2-3' },
    { key: 'Retorno Total (%)',          label: 'Retorno (%)', format: '1.1-2' },
    { key: 'CAGR (%)',                   label: 'CAGR (%)',    format: '1.1-2' },
    { key: 'Volatilidad Anualizada (%)', label: 'Vol. (%)',    format: '1.1-2' },
    { key: 'Max Drawdown (%)',           label: 'MDD (%)',     format: '1.1-2' },
    { key: 'Valor Final ($)',            label: 'Valor Final', format: '1.0-0' },
  ];

  constructor(private api: ApiService, private cdr: ChangeDetectorRef) {}

  ngOnInit(): void {
    this.load();
  }

  load(): void {
    this.loading = true;
    this.errorMessage = '';
    this.api.getFinalTable().subscribe({
      next: (res) => {
        this.loading = false;
        if (res?.available) {
          this.data = res;
        } else {
          this.errorMessage = res?.error || 'No hay datos disponibles.';
          this.data = null;
        }
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.loading = false;
        this.errorMessage = err.error?.detail || `Error al cargar la tabla final (HTTP ${err.status}).`;
        this.data = null;
        this.cdr.detectChanges();
      }
    });
  }

  get strategies(): string[] {
    return this.data?.metrics ? Object.keys(this.data.metrics) : [];
  }

  metric(strategy: string, key: string): any {
    return this.data?.metrics?.[strategy]?.[key];
  }

  isBest(strategy: string, metricKey: string): boolean {
    const map: Record<string, string> = {
      'Sharpe Ratio': 'sharpe',
      'Retorno Total (%)': 'retorno',
      'Max Drawdown (%)': 'mdd',
      'Sortino Ratio': 'sortino',
    };
    const k = map[metricKey];
    if (!k) return false;
    return this.data?.best_by_metric?.[k]?.strategy === strategy;
  }

  isPPO(name: string): boolean {
    return name === 'IA_PPO';
  }

  comparison(baseline: string): any {
    return this.data?.ppo_vs_baselines?.[baseline];
  }

  get baselineNames(): string[] {
    return this.data?.ppo_vs_baselines ? Object.keys(this.data.ppo_vs_baselines) : [];
  }
}
