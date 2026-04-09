import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-status',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './status.component.html',
  styleUrl: './status.component.scss'
})
export class StatusComponent implements OnInit {
  // Estado de conexión
  connected = false;
  loading = true;
  error = '';

  // Fases del sistema
  phases: { name: string; description: string; ready: boolean }[] = [];

  constructor(private api: ApiService, private cdr: ChangeDetectorRef) {}

  ngOnInit(): void {
    this.checkStatus();
  }

  checkStatus(): void {
    this.loading = true;
    this.error = '';
    this.api.getEstado().subscribe({
      next: (data) => {
        this.loading = false;
        this.connected = true;
        this.phases = [
          {
            name: 'Datos de mercado',
            description: 'Features normalizadas y precios de cierre descargados y procesados.',
            ready: data.fase1_datos === true,
          },
          {
            name: 'Modelo PPO (DRL)',
            description: 'Agente de Deep Reinforcement Learning entrenado y listo para simular.',
            ready: data.fase3_modelo_acad === true,
          },
          {
            name: 'Agente especulativo (GMM)',
            description: 'Modelo no supervisado de detección de regímenes de mercado ajustado.',
            ready: data.fase4_especulativo === true,
          },
        ];
        this.cdr.detectChanges();
      },
      error: () => {
        this.loading = false;
        this.connected = false;
        this.error = 'No se pudo conectar con el backend. Verifica que el servidor esté activo en localhost:8000.';
        this.cdr.detectChanges();
      }
    });
  }

  get allReady(): boolean {
    return this.phases.length > 0 && this.phases.every(p => p.ready);
  }

  get readyCount(): number {
    return this.phases.filter(p => p.ready).length;
  }
}
