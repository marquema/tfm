import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { forkJoin } from 'rxjs';
import { ApiService } from '../../services/api.service';
import { PlotlyChartComponent } from '../../components/plotly-chart/plotly-chart.component';

type Mode = 'wf' | 'ew';

@Component({
  selector: 'app-validation',
  standalone: true,
  imports: [CommonModule, PlotlyChartComponent],
  templateUrl: './validation.component.html',
  styleUrl: './validation.component.scss'
})
export class ValidationComponent implements OnInit {
  loading = false;
  errorMessage = '';

  wf: any = null;
  ew: any = null;

  mode: Mode = 'wf';

  sharpeChartData: any[] = [];
  sharpeChartLayout: any = {};
  returnChartData: any[] = [];
  returnChartLayout: any = {};

  constructor(private api: ApiService, private cdr: ChangeDetectorRef) {}

  ngOnInit(): void {
    this.load();
  }

  load(): void {
    this.loading = true;
    this.errorMessage = '';
    forkJoin({
      wf: this.api.getWalkForwardResults(),
      ew: this.api.getExpandingWindowResults(),
    }).subscribe({
      next: (res) => {
        this.loading = false;
        this.wf = res.wf?.available ? res.wf : null;
        this.ew = res.ew?.available ? res.ew : null;
        if (!this.wf && !this.ew) {
          this.errorMessage = 'No hay resultados de validación temporal. Lanza walk-forward o expanding window desde Admin.';
        }
        // Seleccionar modo por defecto según disponibilidad
        if (!this.wf && this.ew) this.mode = 'ew';
        this.rebuildCharts();
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.loading = false;
        this.errorMessage = err.error?.detail || `Error cargando validación (HTTP ${err.status}).`;
        this.cdr.detectChanges();
      }
    });
  }

  setMode(m: Mode): void {
    this.mode = m;
    this.rebuildCharts();
  }

  get active(): any {
    return this.mode === 'wf' ? this.wf : this.ew;
  }

  get activeLabel(): string {
    return this.mode === 'wf' ? 'Walk-Forward (rolling)' : 'Expanding Window';
  }

  private rebuildCharts(): void {
    const src = this.active;
    if (!src || !src.windows?.length) {
      this.sharpeChartData = [];
      this.returnChartData = [];
      return;
    }

    const labels = src.windows.map((w: any, i: number) =>
      w.test_start ? String(w.test_start).substring(0, 10) : `V${i + 1}`
    );
    const sharpes = src.windows.map((w: any) => w['Sharpe Ratio']);
    const returns = src.windows.map((w: any) => w['Retorno Total (%)']);

    this.sharpeChartData = [{
      x: labels,
      y: sharpes,
      type: 'bar',
      name: 'Sharpe',
      marker: { color: sharpes.map((v: number) => v >= 0 ? '#3b82f6' : '#ef4444') },
    }];
    this.sharpeChartLayout = {
      ...this.commonLayout,
      title: { text: `Sharpe por ventana — ${this.activeLabel}`, font: { color: '#e0e0e0' } },
      xaxis: { ...this.commonLayout.xaxis, title: 'Inicio periodo test' },
      yaxis: { ...this.commonLayout.yaxis, title: 'Sharpe Ratio', zeroline: true },
    };

    this.returnChartData = [{
      x: labels,
      y: returns,
      type: 'bar',
      name: 'Retorno (%)',
      marker: { color: returns.map((v: number) => v >= 0 ? '#22c55e' : '#ef4444') },
    }];
    this.returnChartLayout = {
      ...this.commonLayout,
      title: { text: `Retorno por ventana — ${this.activeLabel}`, font: { color: '#e0e0e0' } },
      xaxis: { ...this.commonLayout.xaxis, title: 'Inicio periodo test' },
      yaxis: { ...this.commonLayout.yaxis, title: 'Retorno Total (%)', zeroline: true },
    };
  }

  private readonly commonLayout: any = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e0e0e0', family: 'Segoe UI, system-ui, sans-serif' },
    showlegend: false,
    margin: { l: 60, r: 20, t: 50, b: 70 },
    xaxis: {
      gridcolor: 'rgba(255,255,255,0.06)',
      zerolinecolor: 'rgba(255,255,255,0.1)',
      tickangle: -30,
      tickfont: { size: 10 },
    },
    yaxis: {
      gridcolor: 'rgba(255,255,255,0.06)',
      zerolinecolor: 'rgba(255,255,255,0.3)',
    },
  };
}
