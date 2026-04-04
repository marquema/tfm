import { Component } from '@angular/core';

interface MetricCard {
  label: string;
  value: string;
  suffix: string;
  trend: 'up' | 'down' | 'neutral';
}

@Component({
  selector: 'app-dashboard',
  standalone: true,
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.scss'
})
export class DashboardComponent {
  metrics: MetricCard[] = [
    { label: 'Sharpe Ratio', value: '1.42', suffix: '', trend: 'up' },
    { label: 'Retorno Anualizado', value: '18.7', suffix: '%', trend: 'up' },
    { label: 'Max Drawdown', value: '-12.3', suffix: '%', trend: 'down' },
    { label: 'CAGR', value: '16.2', suffix: '%', trend: 'up' },
    { label: 'Valor Final', value: '142,350', suffix: '$', trend: 'neutral' }
  ];
}
