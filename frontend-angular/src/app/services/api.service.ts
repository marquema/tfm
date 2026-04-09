import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { timeout } from 'rxjs/operators';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private baseUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  // ─── Helpers ───────────────────────────────────────────────────────────────

  private authHeaders(): { headers: { Authorization: string } } {
    const token = localStorage.getItem('token') || '';
    return { headers: { Authorization: 'Bearer ' + token } };
  }

  // ─── Auth (público) ────────────────────────────────────────────────────────

  /** POST /auth/login — Iniciar sesión (OAuth2 form data) */
  login(email: string, password: string): Observable<any> {
    const body = new HttpParams()
      .set('username', email)
      .set('password', password);
    return this.http.post(`${this.baseUrl}/auth/login`, body.toString(), {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    });
  }

  /** POST /auth/register — Registrar nuevo usuario */
  register(email: string, password: string, fullName: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/auth/register`, {
      email,
      password,
      full_name: fullName
    });
  }

  /** GET /auth/me — Datos del usuario autenticado */
  getMe(): Observable<any> {
    return this.http.get(`${this.baseUrl}/auth/me`, this.authHeaders());
  }

  /** GET /auth/users — Lista de usuarios (admin) */
  getUsers(): Observable<any> {
    return this.http.get(`${this.baseUrl}/auth/users`, this.authHeaders());
  }

  /** DELETE /auth/users/:email — Eliminar usuario (admin) */
  deleteUser(email: string): Observable<any> {
    return this.http.delete(`${this.baseUrl}/auth/users/${email}`, this.authHeaders());
  }

  // ─── Público ───────────────────────────────────────────────────────────────

  /** GET /estado — Estado general del sistema */
  getEstado(): Observable<any> {
    return this.http.get(`${this.baseUrl}/estado`);
  }

  /** GET /universo?level=core — Universo de activos */
  getUniverso(level: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/universo`, { params: { level } });
  }

  // ─── Inversor (requiere JWT) ───────────────────────────────────────────────

  /** GET /investor/strategies — Estrategias disponibles */
  getStrategies(): Observable<any> {
    return this.http.get(`${this.baseUrl}/investor/strategies`, this.authHeaders());
  }

  /** POST /investor/simulate — Simular portfolio */
  postSimulate(capital: number, commission: number): Observable<any> {
    return this.http.post(
      `${this.baseUrl}/investor/simulate`,
      { capital, commission },
      this.authHeaders()
    );
  }

  // ─── Admin (requiere JWT con role=admin) ───────────────────────────────────

  /** POST /admin/fase1/screener — Ejecutar screener de activos */
  postScreener(params: any): Observable<any> {
    const defaults: Record<string, string> = {
      start_date: '2020-01-01',
      end_date: '2026-04-01',
      top_n: '15',
      max_per_sector: '3'
    };
    const merged = { ...defaults, ...params };
    return this.http.post(`${this.baseUrl}/admin/fase1/screener`, null, {
      ...this.authHeaders(),
      params: merged
    }).pipe(timeout(600000));
  }

  /** POST /admin/fase1/preparar-datos — Preparar datos de mercado */
  postPrepararDatos(config: any = {}): Observable<any> {
    return this.http.post(`${this.baseUrl}/admin/fase1/preparar-datos`, config, this.authHeaders())
      .pipe(timeout(600000));
  }

  /** GET /admin/fase2/validar-datos — Validar datos disponibles */
  getValidarDatos(): Observable<any> {
    return this.http.get(`${this.baseUrl}/admin/fase2/validar-datos`, this.authHeaders());
  }

  /** POST /admin/fase3/entrenar-academico — Entrenar modelo DRL */
  postEntrenar(steps: number): Observable<any> {
    return this.http.post(
      `${this.baseUrl}/admin/fase3/entrenar-academico`,
      null,
      { ...this.authHeaders(), params: { steps: steps.toString() } }
    ).pipe(timeout(600000));
  }

  /** POST /admin/fase3/walk-forward — Walk-forward validation */
  postWalkForward(steps: number): Observable<any> {
    return this.http.post(
      `${this.baseUrl}/admin/fase3/walk-forward`,
      null,
      { ...this.authHeaders(), params: { steps_por_ventana: steps.toString() } }
    ).pipe(timeout(600000));
  }

  /** POST /admin/fase4/ajustar-especulativo — Ajustar agente especulativo */
  postEspeculativo(): Observable<any> {
    return this.http.post(`${this.baseUrl}/admin/fase4/ajustar-especulativo`, null, this.authHeaders())
      .pipe(timeout(600000));
  }
}
