import { Injectable } from '@angular/core';
import { Router } from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class AuthStore {
  token: string | null = null;
  role: string | null = null;
  email: string | null = null;
  fullName: string | null = null;

  constructor(private router: Router) {
    this.token = localStorage.getItem('token');
    this.role = localStorage.getItem('role');
    this.email = localStorage.getItem('email');
    this.fullName = localStorage.getItem('fullName');
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
    localStorage.setItem('token', token);
    localStorage.setItem('role', role);
    localStorage.setItem('email', email);
    localStorage.setItem('fullName', fullName);
  }

  logout(): void {
    this.token = null;
    this.role = null;
    this.email = null;
    this.fullName = null;
    localStorage.removeItem('token');
    localStorage.removeItem('role');
    localStorage.removeItem('email');
    localStorage.removeItem('fullName');
    this.router.navigate(['/login']);
  }

  getToken(): string | null {
    return this.token;
  }
}
