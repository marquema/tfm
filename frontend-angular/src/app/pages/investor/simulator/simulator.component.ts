import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { ApiService } from '../../../services/api.service';
import { SimulationStore } from '../../../services/simulation.store';

@Component({
  selector: 'app-simulator',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './simulator.component.html',
  styleUrl: './simulator.component.scss'
})
export class SimulatorComponent implements OnInit {
  capital = 10000;
  commission = 0.1;
  loading = false;
  errorMessage = '';
  successMessage = '';

  // Estado del sistema: verifica si hay modelo entrenado antes de permitir simular
  systemReady = false;
  systemChecking = true;
  systemError = '';
  availableStrategies: any[] = [];

  constructor(
    private api: ApiService,
    private simulationStore: SimulationStore,
    private router: Router,
    private cdr: ChangeDetectorRef,
  ) {}

  ngOnInit(): void {
    this.checkSystem();
  }

  /** Verifica si el sistema tiene modelos entrenados y datos preparados. */
  checkSystem(): void {
    this.systemChecking = true;
    this.api.getStrategies().subscribe({
      next: (data) => {
        this.systemChecking = false;
        this.availableStrategies = data.strategies || [];
        const ppo = this.availableStrategies.find((s: any) => s.id === 'ppo');
        this.systemReady = ppo?.available === true;
        if (!this.systemReady) {
          this.systemError = 'El modelo PPO no está entrenado. El administrador debe ejecutar primero el pipeline completo: Preparar Datos → Entrenar PPO.';
        }
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.systemChecking = false;
        this.systemError = 'No se pudo verificar el estado del sistema. Inténtalo más tarde.';
        this.cdr.detectChanges();
      }
    });
  }

  onSimulate(): void {
    this.errorMessage = '';
    this.successMessage = '';
    this.loading = true;

    this.api.postSimulate(this.capital, this.commission / 100).subscribe({
      next: (data) => {
        this.loading = false;
        // Verificar si el backend devolvió un error en el body (no como HTTP error)
        if (data.error) {
          this.errorMessage = data.error;
          this.cdr.detectChanges();
          return;
        }
        this.simulationStore.setResults(data);
        this.router.navigate(['/investor/results']);
      },
      error: (err) => {
        this.loading = false;
        this.errorMessage = err.error?.detail || err.error?.error || 'Error al ejecutar la simulación.';
        this.cdr.detectChanges();
      }
    });
  }
}
