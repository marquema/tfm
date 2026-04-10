import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterLink } from '@angular/router';
import { SimulationStore } from '../../../services/simulation.store';
import { PlotlyChartComponent } from '../../../components/plotly-chart/plotly-chart.component';

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
  imports: [CommonModule, RouterLink, PlotlyChartComponent],
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

  // Chart data (6 gráficas del inversor)
  equityChartData: any[] = [];
  equityChartLayout: any = {};
  drawdownChartData: any[] = [];
  drawdownChartLayout: any = {};
  pieChartData: any[] = [];
  pieChartLayout: any = {};
  weightsChartData: any[] = [];
  weightsChartLayout: any = {};
  dailyReturnsData: any[] = [];
  dailyReturnsLayout: any = {};
  volatilityChartData: any[] = [];
  volatilityChartLayout: any = {};

  private readonly strategyColors: Record<string, string> = {
    'IA_PPO': '#00d4ff',
    'Equal_Weight_Mensual': '#f0a500',
    'Buy_and_Hold': '#7ed957',
    'Cartera_60_40': '#ff6b6b',
    'Markowitz_MV': '#c77dff',
    'Especulativo_HMM': '#ff9f1c',
  };

  private readonly darkLayout: any = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e0e0e0', family: 'Segoe UI, system-ui, sans-serif' },
    legend: {
      orientation: 'h' as const,
      y: -0.2,
      x: 0.5,
      xanchor: 'center' as const,
      font: { color: '#a0a0c0' },
    },
    margin: { l: 60, r: 20, t: 40, b: 60 },
    xaxis: {
      gridcolor: 'rgba(255,255,255,0.06)',
      zerolinecolor: 'rgba(255,255,255,0.1)',
    },
    yaxis: {
      gridcolor: 'rgba(255,255,255,0.06)',
      zerolinecolor: 'rgba(255,255,255,0.1)',
    },
  };

  constructor(
    private simulationStore: SimulationStore,
    private router: Router,
    private cdr: ChangeDetectorRef,
  ) {}

  ngOnInit(): void {
    this.raw = this.simulationStore.getResults();

    if (!this.raw || this.raw.error) {
      this.hasData = false;
      this.errorMessage = this.raw?.error || '';
      return;
    }

    // Parse backend structure
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

    if (this.hasData) {
      this.buildEquityChart();
      this.buildDrawdownChart();
      this.buildPieChart();
      this.buildWeightsChart();
      this.buildDailyReturnsChart();
      this.buildVolatilityChart();
      this.cdr.markForCheck();
    }
  }

  // ---- Chart builders ----

  private buildEquityChart(): void {
    const curves: Record<string, number[]> = this.raw.equity_curves || {};
    const dates: string[] = this.raw.dates || [];
    const ppoDates: string[] = this.raw.ppo_dates || [];

    this.equityChartData = Object.entries(curves).map(([key, values]) => {
      // PPO tiene un punto extra (capital inicial) → usa ppo_dates
      const xDates = key === 'IA_PPO' ? ppoDates : dates;
      return {
        y: values,
        x: xDates.slice(0, values.length),
        type: 'scatter',
        mode: 'lines',
        name: this.formatName(key),
        line: { color: this.strategyColors[key] || '#ffffff', width: key === 'IA_PPO' ? 3 : 1.5 },
        hovertemplate: '<b>%{fullData.name}</b><br>%{x}<br>$%{y:,.2f}<extra></extra>',
      };
    });

    this.equityChartLayout = {
      ...this.darkLayout,
      title: { text: 'Evolución del Capital', font: { color: '#ffffff', size: 16 } },
      xaxis: { ...this.darkLayout.xaxis, title: { text: 'Fecha', font: { color: '#a0a0c0' } }, type: 'date' },
      yaxis: { ...this.darkLayout.yaxis, title: { text: 'Valor ($)', font: { color: '#a0a0c0' } } },
      hovermode: 'x unified',
    };
  }

  private buildDrawdownChart(): void {
    const curves: Record<string, number[]> = this.raw.equity_curves || {};
    const dates: string[] = this.raw.dates || [];
    const ppoDates: string[] = this.raw.ppo_dates || [];

    this.drawdownChartData = Object.entries(curves).map(([key, values]) => {
      const dd = this.computeDrawdown(values);
      const xDates = key === 'IA_PPO' ? ppoDates : dates;
      return {
        y: dd,
        x: xDates.slice(0, dd.length),
        type: 'scatter',
        mode: 'lines',
        name: this.formatName(key),
        line: { color: this.strategyColors[key] || '#ffffff', width: 1.5 },
        fill: 'tozeroy',
        fillcolor: this.hexToRgba(this.strategyColors[key] || '#ffffff', 0.08),
        hovertemplate: '<b>%{fullData.name}</b><br>%{x}<br>%{y:.2f}%<extra></extra>',
      };
    });

    this.drawdownChartLayout = {
      ...this.darkLayout,
      title: { text: 'Drawdown — Peor Caída en Cada Momento', font: { color: '#ffffff', size: 16 } },
      xaxis: { ...this.darkLayout.xaxis, title: { text: 'Fecha', font: { color: '#a0a0c0' } }, type: 'date' },
      yaxis: {
        ...this.darkLayout.yaxis,
        title: { text: 'Drawdown (%)', font: { color: '#a0a0c0' } },
        ticksuffix: '%',
      },
      hovermode: 'x unified',
    };
  }

  private buildWeightsChart(): void {
    const weightsPpo: number[][] = this.raw.weights_ppo || [];
    const tickers: string[] = this.raw.tickers || [];
    const dates: string[] = this.raw.dates || [];

    if (!weightsPpo.length || !tickers.length) return;

    const tickerColors = [
      '#00d4ff', '#f0a500', '#7ed957', '#ff6b6b',
      '#c77dff', '#ff9f1c', '#36d7b7', '#e056fd',
      '#686de0', '#f9ca24', '#badc58', '#ff7979',
    ];

    this.weightsChartData = tickers.map((ticker, idx) => ({
      x: dates.slice(0, weightsPpo.length),
      y: weightsPpo.map((row: number[]) => (row[idx] ?? 0) * 100),
      type: 'scatter',
      mode: 'lines',
      name: ticker,
      stackgroup: 'weights',
      line: { width: 0.5, color: tickerColors[idx % tickerColors.length] },
      fillcolor: this.hexToRgba(tickerColors[idx % tickerColors.length], 0.7),
      hovertemplate: '<b>%{fullData.name}</b><br>%{x}<br>%{y:.1f}%<extra></extra>',
    }));

    this.weightsChartLayout = {
      ...this.darkLayout,
      title: { text: 'Evolución de Pesos PPO', font: { color: '#ffffff', size: 16 } },
      xaxis: { ...this.darkLayout.xaxis, title: { text: 'Fecha', font: { color: '#a0a0c0' } }, type: 'date' },
      yaxis: {
        ...this.darkLayout.yaxis,
        title: { text: 'Peso (%)', font: { color: '#a0a0c0' } },
        range: [0, 100],
        ticksuffix: '%',
      },
      hovermode: 'x unified',
    };
  }

  private buildPieChart(): void {
    const weightsPpo: number[][] = this.raw.weights_ppo || [];
    const tickers: string[] = this.raw.tickers || [];

    if (!weightsPpo.length || !tickers.length) return;

    // Últimos pesos del PPO = cartera final
    const lastWeights = weightsPpo[weightsPpo.length - 1];
    const tickerColors = [
      '#00d4ff', '#f0a500', '#7ed957', '#ff6b6b',
      '#c77dff', '#ff9f1c', '#36d7b7', '#e056fd',
      '#686de0', '#f9ca24', '#badc58', '#ff7979',
    ];

    this.pieChartData = [{
      values: lastWeights.map((w: number) => w * 100),
      labels: tickers,
      type: 'pie',
      hole: 0.4,
      marker: { colors: tickerColors.slice(0, tickers.length) },
      textinfo: 'label+percent',
      textfont: { color: '#e0e0e0', size: 12 },
      hovertemplate: '<b>%{label}</b><br>Peso: %{percent}<extra></extra>',
    }];

    this.pieChartLayout = {
      ...this.darkLayout,
      title: { text: 'Cartera Final del Agente PPO', font: { color: '#ffffff', size: 16 } },
      showlegend: true,
      legend: { ...this.darkLayout.legend, orientation: 'v' as const, x: 1.05, y: 0.5 },
    };
  }

  private buildDailyReturnsChart(): void {
    const curves: Record<string, number[]> = this.raw.equity_curves || {};
    const dates: string[] = this.raw.dates || [];
    const ppoCurve = curves['IA_PPO'];
    if (!ppoCurve || ppoCurve.length < 2) return;

    const returns: number[] = [];
    for (let i = 1; i < ppoCurve.length; i++) {
      returns.push(((ppoCurve[i] - ppoCurve[i - 1]) / ppoCurve[i - 1]) * 100);
    }

    const colors = returns.map(r => r >= 0 ? '#00e676' : '#ff5252');

    this.dailyReturnsData = [{
      x: dates.slice(1, returns.length + 1),
      y: returns,
      type: 'bar',
      marker: { color: colors },
      hovertemplate: '%{x}<br>Retorno: %{y:.2f}%<extra></extra>',
    }];

    this.dailyReturnsLayout = {
      ...this.darkLayout,
      title: { text: 'Retornos Diarios del PPO', font: { color: '#ffffff', size: 16 } },
      xaxis: { ...this.darkLayout.xaxis, title: { text: 'Fecha', font: { color: '#a0a0c0' } }, type: 'date' },
      yaxis: {
        ...this.darkLayout.yaxis,
        title: { text: 'Retorno (%)', font: { color: '#a0a0c0' } },
        ticksuffix: '%',
        zeroline: true,
        zerolinecolor: 'rgba(255,255,255,0.3)',
      },
      bargap: 0.1,
    };
  }

  private buildVolatilityChart(): void {
    const curves: Record<string, number[]> = this.raw.equity_curves || {};
    const dates: string[] = this.raw.dates || [];
    const ppoDates: string[] = this.raw.ppo_dates || [];

    this.volatilityChartData = Object.entries(curves).map(([key, values]) => {
      const returns: number[] = [];
      for (let i = 1; i < values.length; i++) {
        returns.push((values[i] - values[i - 1]) / values[i - 1]);
      }
      const window = 20;
      const rollingVol: (number | null)[] = [];
      for (let i = 0; i < returns.length; i++) {
        if (i < window - 1) {
          rollingVol.push(null);
        } else {
          const slice = returns.slice(i - window + 1, i + 1);
          const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
          const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / slice.length;
          rollingVol.push(Math.sqrt(variance) * Math.sqrt(252) * 100);
        }
      }

      const xDates = key === 'IA_PPO' ? ppoDates : dates;
      return {
        x: xDates.slice(1, rollingVol.length + 1),
        y: rollingVol,
        type: 'scatter',
        mode: 'lines',
        name: this.formatName(key),
        line: { color: this.strategyColors[key] || '#ffffff', width: 2 },
        hovertemplate: '<b>%{fullData.name}</b><br>%{x}<br>%{y:.1f}%<extra></extra>',
      };
    });

    this.volatilityChartLayout = {
      ...this.darkLayout,
      title: { text: 'Volatilidad Rolling (20 días, anualizada)', font: { color: '#ffffff', size: 16 } },
      xaxis: { ...this.darkLayout.xaxis, title: { text: 'Fecha', font: { color: '#a0a0c0' } }, type: 'date' },
      yaxis: {
        ...this.darkLayout.yaxis,
        title: { text: 'Volatilidad (%)', font: { color: '#a0a0c0' } },
        ticksuffix: '%',
      },
      hovermode: 'x unified',
    };
  }

  // ---- Helpers ----

  private computeDrawdown(values: number[]): number[] {
    let peak = values[0] || 0;
    return values.map((v: number) => {
      if (v > peak) peak = v;
      return peak > 0 ? -((peak - v) / peak) * 100 : 0;
    });
  }

  private hexToRgba(hex: string, alpha: number): string {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
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
    if (value == null) return '\u2014';
    return value.toFixed(decimals);
  }

  goBack(): void {
    this.router.navigate(['/investor/simulator']);
  }
}
