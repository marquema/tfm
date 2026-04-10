import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-admin',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './admin.component.html',
  styleUrl: './admin.component.scss'
})
export class AdminComponent implements OnInit {

  // ─── Usuarios ──────────────────────────────────────────────────────────────
  users: any[] = [];
  usersLoading = false;
  usersError = '';

  // ─── Estado de operaciones ─────────────────────────────────────────────────
  // Solo las operaciones síncronas (screener, datos, especulativo) se auto-desbloquean.
  // Las de background (entrenar, walk-forward) requieren desbloqueo manual.
  activeOperation: string | null = null;

  // ─── Universo actual (tickers por defecto) ─────────────────────────────────
  defaultTickers: string[] = [];
  defaultTickersLoading = false;

  // ─── Parámetros de fechas (compartidos entre screener y preparar datos) ────
  startDate = '2019-01-01';
  endDate = '2026-04-01';

  // ─── Parámetros del screener ───────────────────────────────────────────────
  topN = 15;
  maxPerSector = 3;

  // ─── Resultados del screener ───────────────────────────────────────────────
  screenerCandidates: any[] = [];

  // ─── Resultados de preparar datos ──────────────────────────────────────────
  prepararResult: any = null;

  // ─── Parámetros de entrenamiento ───────────────────────────────────────────
  trainSteps = 100000;
  wfSteps = 50000;

  // ─── Log de mensajes ───────────────────────────────────────────────────────
  messages: { text: string; type: 'success' | 'error' | 'info'; time: Date }[] = [];

  constructor(private api: ApiService, private cdr: ChangeDetectorRef) {}

  ngOnInit(): void {
    this.loadUsers();
    this.loadDefaultUniverse();

    // Recuperar operación de background activa (persiste entre navegaciones)
    const saved = localStorage.getItem('admin_active_op');
    if (saved) {
      this.activeOperation = saved;
      this.log(`Operación en segundo plano: ${saved}. Pulsa "Desbloquear" cuando termine.`, 'info');
    }
  }

  get locked(): boolean {
    return this.activeOperation !== null;
  }

  // ─── Usuarios ──────────────────────────────────────────────────────────────

  loadUsers(): void {
    this.usersLoading = true;
    this.usersError = '';
    console.log('[Admin] Cargando usuarios...');
    this.api.getUsers().subscribe({
      next: (data) => {
        this.usersLoading = false;
        this.users = Array.isArray(data) ? data : [];
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.usersLoading = false;
        this.usersError = err.status === 401
          ? 'Sesión expirada. Vuelve a iniciar sesión.'
          : (err.error?.detail || `Error al cargar usuarios (HTTP ${err.status}).`);
        this.cdr.detectChanges();
      }
    });
  }

  loadDefaultUniverse(): void {
    this.defaultTickersLoading = true;
    this.api.getUniverso('core').subscribe({
      next: (data) => {
        this.defaultTickersLoading = false;
        if (Array.isArray(data)) {
          this.defaultTickers = data.map((d: any) => d.ticker || d);
        } else if (data?.tickers) {
          this.defaultTickers = data.tickers;
        }
        this.cdr.detectChanges();
      },
      error: () => {
        this.defaultTickersLoading = false;
        this.cdr.detectChanges();
      }
    });
  }

  deleteUser(email: string): void {
    if (!confirm(`¿Eliminar usuario ${email}?`)) return;
    this.api.deleteUser(email).subscribe({
      next: () => {
        this.log(`Usuario ${email} eliminado.`, 'success');
        this.loadUsers();
      },
      error: (err) => this.log(err.error?.detail || 'Error al eliminar usuario.', 'error')
    });
  }

  // ─── Screener (síncrono — se auto-desbloquea) ─────────────────────────────

  runScreener(): void {
    this.lock('Screener S&P 500');
    this.screenerCandidates = [];
    this.log('Ejecutando screener... Esto puede tardar 3-5 minutos.', 'info');

    this.api.postScreener({
      start_date: this.startDate,
      end_date: this.endDate,
      top_n: this.topN.toString(),
      max_per_sector: this.maxPerSector.toString(),
    }).subscribe({
      next: (res) => {
        this.unlock();
        this.screenerCandidates = res.details || [];
        const n = res.candidates?.length || 0;
        this.log(`Screener completado: ${n} candidatos seleccionados. Se usarán automáticamente en "Preparar Datos".`, 'success');
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.unlock();
        this.log(err.error?.detail || 'Error al ejecutar screener.', 'error');
        this.cdr.detectChanges();
      }
    });
  }

  // ─── Preparar datos (síncrono — se auto-desbloquea) ────────────────────────

  runPrepararDatos(): void {
    this.lock('Preparando datos');
    this.prepararResult = null;
    this.log('Descargando datos y generando features...', 'info');

    this.api.postPrepararDatos({
      tickers: null,
      start: this.startDate,
      end: this.endDate,
    }).subscribe({
      next: (res) => {
        this.unlock();
        this.prepararResult = res;
        const tickers = res.tickers || [];
        this.log(`Datos preparados: ${tickers.length} activos, ${res.n_days || '?'} días. Tickers: ${tickers.join(', ')}`, 'success');
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.unlock();
        this.log(err.error?.detail || 'Error al preparar datos.', 'error');
        this.cdr.detectChanges();
      }
    });
  }

  // ─── Entrenar PPO (background — requiere desbloqueo manual) ────────────────

  runEntrenar(): void {
    this.lockBackground('Entrenamiento PPO');
    this.log(`Entrenamiento PPO lanzado (${this.trainSteps.toLocaleString()} pasos). Corre en segundo plano — pulsa "Desbloquear" cuando veas que terminó en la terminal.`, 'info');

    this.api.postEntrenar(this.trainSteps).subscribe({
      next: (res) => {
        // El backend respondió inmediatamente — el entrenamiento corre en background.
        // Cambiamos el nombre de la operación para que el banner sea informativo.
        this.activeOperation = 'Entrenamiento PPO (segundo plano)';
        this.log(res.message || 'Entrenamiento lanzado en segundo plano. Pulsa "Desbloquear" cuando termine.', 'info');
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.unlock();
        this.log(err.error?.detail || 'Error al iniciar entrenamiento.', 'error');
      }
    });
  }

  // ─── Walk-Forward (background — requiere desbloqueo manual) ────────────────

  runWalkForward(): void {
    this.lockBackground('Walk-Forward');
    this.log(`Walk-Forward lanzado (${this.wfSteps.toLocaleString()} pasos/ventana). Corre en segundo plano.`, 'info');

    this.api.postWalkForward(this.wfSteps).subscribe({
      next: (res) => {
        this.activeOperation = 'Walk-Forward (segundo plano)';
        this.log(res.message || 'Walk-Forward lanzado en segundo plano. Pulsa "Desbloquear" cuando termine.', 'info');
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.unlock();
        this.log(err.error?.detail || 'Error en walk-forward.', 'error');
      }
    });
  }

  // ─── Especulativo (síncrono — se auto-desbloquea) ──────────────────────────

  runEspeculativo(): void {
    this.lock('Agente Especulativo');
    this.log('Ajustando agente especulativo (GMM + K-Means)...', 'info');

    this.api.postEspeculativo().subscribe({
      next: (res) => {
        this.unlock();
        this.log(`Especulativo ajustado. Retorno test: ${res.retorno_test || '?'}, Valor final: ${res.valor_final || '?'}`, 'success');
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.unlock();
        this.log(err.error?.detail || 'Error al ajustar especulativo.', 'error');
        this.cdr.detectChanges();
      }
    });
  }

  // ─── Desbloqueo manual ─────────────────────────────────────────────────────

  forceUnlock(): void {
    this.unlock();
    this.log('Botones desbloqueados manualmente.', 'info');
  }

  // ─── Helpers ───────────────────────────────────────────────────────────────

  private lock(name: string): void {
    this.activeOperation = name;
    // NO guardar en localStorage — se auto-desbloquea al terminar
  }

  private lockBackground(name: string): void {
    this.activeOperation = name;
    localStorage.setItem('admin_active_op', name);  // Persiste entre navegaciones
  }

  private unlock(): void {
    this.activeOperation = null;
    localStorage.removeItem('admin_active_op');
    this.cdr.detectChanges();
  }

  private log(text: string, type: 'success' | 'error' | 'info'): void {
    this.messages.unshift({ text, type, time: new Date() });
    if (this.messages.length > 10) this.messages.pop();
    this.cdr.detectChanges();
  }
}
