import {
  Component,
  Input,
  AfterViewInit,
  OnChanges,
  SimpleChanges,
  ElementRef,
  ViewChild,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
} from '@angular/core';

declare var Plotly: any;

// Dynamic import helper — resolved once and cached
let plotlyPromise: Promise<any> | null = null;
function loadPlotly(): Promise<any> {
  if (!plotlyPromise) {
    plotlyPromise = import('plotly.js-dist-min' as any);
  }
  return plotlyPromise;
}

@Component({
  selector: 'app-plotly-chart',
  standalone: true,
  template: `<div #chartContainer [id]="chartId" class="plotly-container"></div>`,
  styles: [
    `
      .plotly-container {
        width: 100%;
        min-height: 400px;
      }
    `,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PlotlyChartComponent implements AfterViewInit, OnChanges {
  @Input() data: any[] = [];
  @Input() layout: any = {};
  @Input() chartId = 'chart';

  @ViewChild('chartContainer', { static: true }) containerRef!: ElementRef<HTMLDivElement>;

  private plotlyLib: any = null;
  private rendered = false;

  constructor(private cdr: ChangeDetectorRef) {}

  async ngAfterViewInit(): Promise<void> {
    this.plotlyLib = await loadPlotly();
    // The dynamic import returns a module; Plotly is the default export
    if (this.plotlyLib.default) {
      this.plotlyLib = this.plotlyLib.default;
    }
    this.rendered = true;
    this.drawChart();
  }

  ngOnChanges(_changes: SimpleChanges): void {
    if (this.rendered) {
      this.drawChart();
    }
  }

  private drawChart(): void {
    if (!this.plotlyLib || !this.containerRef?.nativeElement) {
      return;
    }
    const config = { responsive: true, displayModeBar: false };
    this.plotlyLib.newPlot(this.containerRef.nativeElement, this.data, this.layout, config);
    this.cdr.markForCheck();
  }
}
