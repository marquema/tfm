import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterLink } from '@angular/router';
import { ApiService } from '../../services/api.service';
import { AuthStore } from '../../services/auth.store';
import { SimulationStore } from '../../services/simulation.store';
import { PlotlyChartComponent } from '../../components/plotly-chart/plotly-chart.component';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, RouterLink, PlotlyChartComponent],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.scss'
})
export class DashboardComponent implements OnInit {
  // Estado del sistema
  systemStatus: any = null;
  statusLoading = true;

  // Última simulación (si existe)
  lastSim: any = null;
  hasSimulation = false;
  ppoMetrics: any = null;
  bestStrategy = '';

  // Gráfica mini de equity (resumen)
  equityChartData: any[] = [];
  equityChartLayout: any = {};

  // Gráfica mini de pie (cartera final)
  pieChartData: any[] = [];
  pieChartLayout: any = {};

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
    font: { color: '#e0e0e0', family: 'Segoe UI, system-ui, sans-serif', size: 11 },
    legend: { orientation: 'h' as const, y: -0.15, x: 0.5, xanchor: 'center' as const, font: { color: '#a0a0c0', size: 10 } },
    margin: { l: 50, r: 15, t: 30, b: 50 },
    xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
    yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
  };

  constructor(
    public auth: AuthStore,
    private api: ApiService,
    private simStore: SimulationStore,
    private router: Router,
    private cdr: ChangeDetectorRef,
  ) {}

  ngOnInit(): void {
    this.loadSystemStatus();
    this.loadLastSimulation();
  }

  loadSystemStatus(): void {
    this.statusLoading = true;
    this.api.getEstado().subscribe({
      next: (data) => {
        this.systemStatus = data;
        this.statusLoading = false;
        this.cdr.detectChanges();
      },
      error: () => {
        this.statusLoading = false;
        this.cdr.detectChanges();
      },
    });
  }

  loadLastSimulation(): void {
    const sim = this.simStore.getResults();
    if (!sim || sim.error || !sim.metrics) {
      this.hasSimulation = false;
      return;
    }

    this.lastSim = sim;
    this.hasSimulation = true;

    // Extraer métricas del PPO
    this.ppoMetrics = sim.metrics['IA_PPO'] || null;

    // Encontrar la mejor estrategia por Sharpe
    let bestSharpe = -Infinity;
    for (const [name, m] of Object.entries(sim.metrics) as [string, any][]) {
      const sharpe = m['Sharpe Ratio'] || 0;
      if (sharpe > bestSharpe) {
        bestSharpe = sharpe;
        this.bestStrategy = this.formatName(name);
      }
    }

    this.buildEquityMini();
    this.buildPieMini();
  }

  private buildEquityMini(): void {
    const curves = this.lastSim.equity_curves || {};
    const dates = this.lastSim.dates || [];
    const ppoDates = this.lastSim.ppo_dates || [];

    this.equityChartData = Object.entries(curves).map(([key, values]: [string, any]) => {
      const xDates = key === 'IA_PPO' ? ppoDates : dates;
      return {
        y: values,
        x: xDates.slice(0, values.length),
        type: 'scatter',
        mode: 'lines',
        name: this.formatName(key),
        line: { color: this.strategyColors[key] || '#aaa', width: key === 'IA_PPO' ? 2.5 : 1.2 },
      };
    });

    this.equityChartLayout = {
      ...this.darkLayout,
      xaxis: { ...this.darkLayout.xaxis, type: 'date' },
      yaxis: { ...this.darkLayout.yaxis, title: { text: '$', font: { color: '#a0a0c0', size: 10 } } },
      hovermode: 'x unified',
      height: 300,
    };
  }

  private buildPieMini(): void {
    const weights = this.lastSim.weights_ppo || [];
    const tickers = this.lastSim.tickers || [];
    if (!weights.length || !tickers.length) return;

    const last = weights[weights.length - 1];
    const tickerColors = ['#00d4ff','#f0a500','#7ed957','#ff6b6b','#c77dff','#ff9f1c','#36d7b7','#e056fd','#686de0','#f9ca24','#badc58','#ff7979'];

    this.pieChartData = [{
      values: last.map((w: number) => w * 100),
      labels: tickers,
      type: 'pie',
      hole: 0.45,
      marker: { colors: tickerColors.slice(0, tickers.length) },
      textinfo: 'label+percent',
      textfont: { color: '#e0e0e0', size: 10 },
    }];

    this.pieChartLayout = {
      ...this.darkLayout,
      showlegend: false,
      height: 300,
      margin: { l: 10, r: 10, t: 10, b: 10 },
    };
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

  get systemReady(): boolean {
    return this.systemStatus?.fase1_datos && this.systemStatus?.fase3_modelo_acad;
  }

  goSimulate(): void {
    this.router.navigate(['/investor/simulator']);
  }

  goResults(): void {
    this.router.navigate(['/investor/results']);
  }
}
