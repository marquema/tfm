import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { JsonPipe } from '@angular/common';
import { ApiService } from '../../services/api.service';

interface PhaseAction {
  id: string;
  label: string;
  description: string;
  loading: boolean;
  response: string;
  error: boolean;
}

@Component({
  selector: 'app-status',
  standalone: true,
  imports: [JsonPipe],
  templateUrl: './status.component.html',
  styleUrl: './status.component.scss'
})
export class StatusComponent implements OnInit {
  systemStatus: any = null;
  statusLoading = true;
  statusError = '';

  phases: PhaseAction[] = [
    {
      id: 'preparar',
      label: 'Preparar Datos',
      description: 'Descarga y prepara los datos de mercado (Fase 1)',
      loading: false,
      response: '',
      error: false
    },
    {
      id: 'entrenar',
      label: 'Entrenar Modelo',
      description: 'Entrena el agente DRL con PPO (Fase 3)',
      loading: false,
      response: '',
      error: false
    },
    {
      id: 'walkforward',
      label: 'Walk-Forward',
      description: 'Ejecuta validacion walk-forward (Fase 3)',
      loading: false,
      response: '',
      error: false
    },
    {
      id: 'especulativo',
      label: 'Agente Especulativo',
      description: 'Ajusta el agente especulativo (Fase 4)',
      loading: false,
      response: '',
      error: false
    }
  ];

  constructor(private apiService: ApiService, private cdr: ChangeDetectorRef) {}

  ngOnInit(): void {
    this.loadStatus();
  }

  loadStatus(): void {
    this.statusLoading = true;
    this.statusError = '';
    this.apiService.getEstado().subscribe({
      next: (data) => {
        this.systemStatus = data;
        this.statusLoading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.statusError = 'No se pudo conectar con el backend. Verifica que el servidor este activo en localhost:8000.';
        this.statusLoading = false;
        console.error('Error cargando estado:', err);
        this.cdr.detectChanges();
      }
    });
  }

  executePhase(phase: PhaseAction): void {
    phase.loading = true;
    phase.response = '';
    phase.error = false;

    let observable;

    switch (phase.id) {
      case 'preparar':
        observable = this.apiService.postPrepararDatos({});
        break;
      case 'entrenar':
        observable = this.apiService.postEntrenar(100000);
        break;
      case 'walkforward':
        observable = this.apiService.postWalkForward(50000);
        break;
      case 'especulativo':
        observable = this.apiService.postEspeculativo();
        break;
      default:
        return;
    }

    observable.subscribe({
      next: (data) => {
        phase.response = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
        phase.loading = false;
        phase.error = false;
      },
      error: (err) => {
        phase.response = `Error: ${err.message || 'Fallo en la ejecucion'}`;
        phase.loading = false;
        phase.error = true;
        console.error(`Error en fase ${phase.id}:`, err);
      }
    });
  }
}
