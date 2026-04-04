import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private baseUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  /** GET /estado - Estado general del sistema */
  getEstado(): Observable<any> {
    return this.http.get(`${this.baseUrl}/estado`);
  }

  /** GET /universo?level=core - Universo de activos */
  getUniverso(level: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/universo`, {
      params: { level }
    });
  }

  /** POST /fase1/preparar-datos - Preparar datos de mercado */
  postPrepararDatos(config: any): Observable<any> {
    return this.http.post(`${this.baseUrl}/fase1/preparar-datos`, config);
  }

  /** POST /fase1/screener - Ejecutar screener de activos */
  postScreener(params: any): Observable<any> {
    return this.http.post(`${this.baseUrl}/fase1/screener`, params);
  }

  /** POST /fase3/entrenar-academico - Entrenar modelo DRL */
  postEntrenar(steps: number): Observable<any> {
    return this.http.post(`${this.baseUrl}/fase3/entrenar-academico`, { steps });
  }

  /** POST /fase3/walk-forward - Walk-forward validation */
  postWalkForward(steps: number): Observable<any> {
    return this.http.post(`${this.baseUrl}/fase3/walk-forward`, { steps });
  }

  /** POST /fase4/ajustar-especulativo - Ajustar agente especulativo */
  postEspeculativo(): Observable<any> {
    return this.http.post(`${this.baseUrl}/fase4/ajustar-especulativo`, {});
  }

  /** GET /fase2/validar-datos - Validar datos disponibles */
  getValidarDatos(): Observable<any> {
    return this.http.get(`${this.baseUrl}/fase2/validar-datos`);
  }
}
