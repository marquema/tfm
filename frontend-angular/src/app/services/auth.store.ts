import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class AuthStore {
  token: string | null = null;
  role: string | null = null;
  email: string | null = null;
  fullName: string | null = null;

  // true mientras se verifica el token al arrancar la app
  validating = false;
  // true cuando la validación ha terminado (éxito o fallo)
  initialized = false;

  constructor(private router: Router, private http: HttpClient) {
    this.token = localStorage.getItem('token');
    this.role = localStorage.getItem('role');
    this.email = localStorage.getItem('email');
    this.fullName = localStorage.getItem('fullName');
  }

  /**
   * Verifica contra el backend si el token almacenado sigue siendo válido.
   * Se llama una vez al arrancar la app (desde app.component ngOnInit).
   * Si el token es inválido o ha expirado, limpia la sesión.
   */
  validateSession(): void {
    if (!this.token) {
      this.initialized = true;
      return;
    }

    this.validating = true;
    this.http.get(`${environment.apiUrl}/auth/me`, {
      headers: { Authorization: `Bearer ${this.token}` }
    }).subscribe({
      next: (data: any) => {
        // Token válido — actualizar datos por si cambiaron en el backend
        this.role = data.role;
        this.email = data.email;
        this.fullName = data.full_name;
        localStorage.setItem('role', data.role);
        localStorage.setItem('email', data.email);
        localStorage.setItem('fullName', data.full_name || '');
        this.validating = false;
        this.initialized = true;
      },
      error: () => {
        // Token expirado o inválido — limpiar sesión
        console.warn('[Auth] Token expirado o inválido. Cerrando sesión.');
        this.clearSession();
        this.validating = false;
        this.initialized = true;
      }
    });
  }

  isLoggedIn(): boolean {
    return !!this.token;
  }

  isAdmin(): boolean {
    return this.role === 'admin';
  }

  login(token: string, role: string, email: string, fullName: string): void {
    this.token = token;
    this.role = role;
    this.email = email;
    this.fullName = fullName;
    this.initialized = true;
    localStorage.setItem('token', token);
    localStorage.setItem('role', role);
    localStorage.setItem('email', email);
    localStorage.setItem('fullName', fullName);
  }

  logout(): void {
    this.clearSession();
    this.router.navigate(['/login']);
  }

  getToken(): string | null {
    return this.token;
  }

  private clearSession(): void {
    this.token = null;
    this.role = null;
    this.email = null;
    this.fullName = null;
    localStorage.removeItem('token');
    localStorage.removeItem('role');
    localStorage.removeItem('email');
    localStorage.removeItem('fullName');
  }
}
