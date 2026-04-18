import { Component, OnInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
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
export class AdminComponent implements OnInit, OnDestroy {

  // Polling para detectar cuando un background task termina
  private pollTimer: any = null;
  private pollField: string = '';  // campo de /estado a vigilar

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
  lastScreenerMeta: { start: string; end: string; created_at: string } | null = null;

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
  ewSteps = 100000;
  saSteps = 200000;

  // ─── Perfiles de riesgo ────────────────────────────────────────────────────
  riskProfiles: any[] = [];
  selectedRiskProfile = 'balanced';

  // ─── Resultados de sensitivity analysis ───────────────────────────────────
  sensitivityData: any = null;
  sensitivityLoading = false;

  // ─── Log de mensajes ───────────────────────────────────────────────────────
  messages: { text: string; type: 'success' | 'error' | 'info'; time: Date }[] = [];

  constructor(private api: ApiService, private cdr: ChangeDetectorRef) {}

  ngOnInit(): void {
    this.loadUsers();
    this.loadDefaultUniverse();
    this.loadRiskProfiles();
    this.loadSensitivityResults();

    // Recuperar operación de background activa (persiste entre navegaciones)
    const saved = localStorage.getItem('admin_active_op');
    if (saved) {
      this.activeOperation = saved;
      this.log(`Operación en segundo plano: ${saved}. Pulsa "Desbloquear" cuando termine.`, 'info');
    }
  }

  loadRiskProfiles(): void {
    this.api.getRiskProfiles().subscribe({
      next: (data) => {
        this.riskProfiles = Array.isArray(data) ? data : [];
        this.cdr.detectChanges();
      },
      error: () => {
        this.riskProfiles = [];
        this.cdr.detectChanges();
      }
    });
  }

  get selectedProfileInfo(): any {
    return this.riskProfiles.find(p => p.id === this.selectedRiskProfile);
  }

  // ─── Sensitivity analysis ──────────────────────────────────────────────────

  loadSensitivityResults(): void {
    this.sensitivityLoading = true;
    this.api.getSensitivityResults().subscribe({
      next: (data) => {
        this.sensitivityLoading = false;
        this.sensitivityData = data?.available ? data : null;
        this.cdr.detectChanges();
      },
      error: () => {
        this.sensitivityLoading = false;
        this.sensitivityData = null;
        this.cdr.detectChanges();
      }
    });
  }

  runSensitivityAnalysis(): void {
    this.lockBackground('Análisis de Sensibilidad');
    this.log(
      `Análisis de sensibilidad lanzado (${this.saSteps.toLocaleString()} pasos × 4 configs). ` +
      `Tarda ~4× un entrenamiento normal.`,
      'info'
    );

    this.api.postSensitivityAnalysis(this.saSteps).subscribe({
      next: (res) => {
        this.activeOperation = 'Análisis de Sensibilidad (segundo plano)';
        this.log(res.message || 'Análisis lanzado. Se desbloqueará automáticamente.', 'info');
        this.startPolling('fase3_sa_done', () => this.loadSensitivityResults());
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.unlock();
        this.log(err.error?.detail || 'Error al lanzar análisis de sensibilidad.', 'error');
      }
    });
  }

  isBestConfig(configName: string, metric: 'sharpe' | 'retorno' | 'mdd'): boolean {
    return this.sensitivityData?.best_by_metric?.[metric]?.config === configName;
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

    // Intentamos primero el último screener persistido (con métricas reales).
    // Si no hay ninguno todavía, caemos al universo core como fallback.
    this.api.getLastScreener().subscribe({
      next: (data) => {
        this.defaultTickersLoading = false;
        if (data?.available && data.candidates?.length) {
          this.defaultTickers = data.candidates;
          this.screenerCandidates = data.details || [];
          this.lastScreenerMeta = {
            start: data.start_date,
            end: data.end_date,
            created_at: data.created_at,
          };
        } else {
          this.loadFallbackUniverse();
        }
        this.cdr.detectChanges();
      },
      error: () => {
        this.loadFallbackUniverse();
      }
    });
  }

  private loadFallbackUniverse(): void {
    this.api.getUniverso('core').subscribe({
      next: (data) => {
        this.defaultTickersLoading = false;
        if (Array.isArray(data)) {
          this.defaultTickers = data.map((d: any) => d.ticker || d);
        } else if (data?.tickers) {
          this.defaultTickers = data.tickers;
        }
        this.lastScreenerMeta = null;
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
    const profileName = this.selectedProfileInfo?.name || this.selectedRiskProfile;
    this.log(
      `Entrenamiento PPO lanzado (${this.trainSteps.toLocaleString()} pasos, perfil: ${profileName}). ` +
      `Corre en segundo plano — se desbloqueará automáticamente.`,
      'info'
    );

    this.api.postEntrenar(this.trainSteps, this.selectedRiskProfile).subscribe({
      next: (res) => {
        this.activeOperation = 'Entrenamiento PPO (segundo plano)';
        this.log(res.message || 'Entrenamiento lanzado. Se desbloqueará automáticamente al terminar.', 'info');
        this.startPolling('fase3_training_done');
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.unlock();
        this.log(err.error?.detail || 'Error al iniciar entrenamiento.', 'error');
      }
    });
  }

  // ─── Walk-Forward (background — auto-detecta finalización) ─────────────────

  runWalkForward(): void {
    this.lockBackground('Walk-Forward');
    this.log(`Walk-Forward lanzado (${this.wfSteps.toLocaleString()} pasos/ventana). Se desbloqueará automáticamente.`, 'info');

    this.api.postWalkForward(this.wfSteps).subscribe({
      next: (res) => {
        this.activeOperation = 'Walk-Forward (segundo plano)';
        this.log(res.message || 'Walk-Forward lanzado. Se desbloqueará automáticamente al terminar.', 'info');
        this.startPolling('fase3_wf_done');
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.unlock();
        this.log(err.error?.detail || 'Error en walk-forward.', 'error');
      }
    });
  }

  // ─── Expanding Window (background — auto-detecta finalización) ────────────

  runExpandingWindow(): void {
    this.lockBackground('Expanding Window');
    this.log(
      `Expanding Window lanzado (${this.ewSteps.toLocaleString()} pasos/ventana, ` +
      `min 504 días train, 63 días test). Se desbloqueará automáticamente.`,
      'info'
    );

    this.api.postExpandingWindow(this.ewSteps).subscribe({
      next: (res) => {
        this.activeOperation = 'Expanding Window (segundo plano)';
        this.log(res.message || 'Expanding Window lanzado.', 'info');
        this.startPolling('fase3_ew_done');
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.unlock();
        this.log(err.error?.detail || 'Error en expanding window.', 'error');
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
    this.stopPolling();
    this.unlock();
    this.log('Botones desbloqueados manualmente.', 'info');
  }

  ngOnDestroy(): void {
    this.stopPolling();
  }

  // ─── Polling para background tasks ─────────────────────────────────────────

  /**
   * Inicia un polling cada 15s a GET /estado para detectar cuando un
   * background task termina. Cuando el campo vigilado pasa a true,
   * desbloquea automáticamente sin intervención del usuario.
   */
  private startPolling(field: string, onComplete?: () => void): void {
    this.stopPolling();
    this.pollField = field;

    // Guardar referencia al estado actual para detectar cambios
    const initialStatus = this.pollField;

    this.pollTimer = setInterval(() => {
      this.api.getEstado().subscribe({
        next: (data) => {
          if (data[initialStatus] === true) {
            this.stopPolling();
            this.unlock();
            this.log(`Operación completada: ${this.activeOperation || initialStatus}`, 'success');
            if (onComplete) onComplete();
            this.cdr.detectChanges();
          }
        },
        error: () => {
          // Backend no responde — no hacer nada, seguir intentando
        }
      });
    }, 15000); // Cada 15 segundos
  }

  private stopPolling(): void {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  // ─── Helpers ───────────────────────────────────────────────────────────────

  private lock(name: string): void {
    this.activeOperation = name;
  }

  private lockBackground(name: string): void {
    this.activeOperation = name;
    localStorage.setItem('admin_active_op', name);
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
