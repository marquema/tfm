import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterLink } from '@angular/router';
import { SimulationStore } from '../../../services/simulation.store';

interface StrategyMetrics {
  name: string;
  sharpe: number;
  sortino: number;
  retorno: number;
  cagr: number;
  volatilidad: number;
  mdd: number;
  valor_final: number;
}

@Component({
  selector: 'app-results',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './results.component.html',
  styleUrl: './results.component.scss'
})
export class ResultsComponent implements OnInit {
  raw: any = null;
  strategies: StrategyMetrics[] = [];
  testPeriod: any = null;
  tickers: string[] = [];
  hasData = false;
  errorMessage = '';

  constructor(
    private simulationStore: SimulationStore,
    private router: Router,
  ) {}

  ngOnInit(): void {
    this.raw = this.simulationStore.getResults();

    if (!this.raw || this.raw.error) {
      this.hasData = false;
      this.errorMessage = this.raw?.error || '';
      return;
    }

    // Parsear la estructura real del backend:
    // { metrics: { "IA_PPO": { "Sharpe Ratio": 0.6, ... }, "Equal_Weight_Mensual": {...} }, ... }
    if (this.raw.metrics && typeof this.raw.metrics === 'object') {
      this.strategies = Object.entries(this.raw.metrics).map(([name, m]: [string, any]) => ({
        name: this.formatName(name),
        sharpe: m['Sharpe Ratio'] ?? 0,
        sortino: m['Sortino Ratio'] ?? 0,
        retorno: m['Retorno Total (%)'] ?? 0,
        cagr: m['CAGR (%)'] ?? 0,
        volatilidad: m['Volatilidad Anualizada (%)'] ?? 0,
        mdd: m['Max Drawdown (%)'] ?? 0,
        valor_final: m['Valor Final ($)'] ?? 0,
      }));
      this.hasData = this.strategies.length > 0;
    }

    this.testPeriod = this.raw.test_period || null;
    this.tickers = this.raw.tickers || [];
  }

  private formatName(key: string): string {
    const names: Record<string, string> = {
      'IA_PPO': 'IA PPO (DRL)',
      'Equal_Weight_Mensual': 'Equal Weight',
      'Buy_and_Hold': 'Buy & Hold',
      'Cartera_60_40': 'Cartera 60/40',
      'Markowitz_MV': 'Markowitz MV',
      'Especulativo_HMM': 'Especulativo (GMM)',
    };
    return names[key] || key;
  }

  fmt(value: number, decimals: number = 2): string {
    if (value == null) return '—';
    return value.toFixed(decimals);
  }

  goBack(): void {
    this.router.navigate(['/investor/simulator']);
  }
}
