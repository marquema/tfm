import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { ApiService } from '../../services/api.service';
import { AuthStore } from '../../services/auth.store';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss'
})
export class LoginComponent {
  // Login form
  email = '';
  password = '';

  // Register form
  registerEmail = '';
  registerPassword = '';
  registerName = '';

  // UI state
  showRegister = false;
  loading = false;
  errorMessage = '';
  successMessage = '';

  constructor(
    private apiService: ApiService,
    private authStore: AuthStore,
    private router: Router
  ) {}

  onSubmit(): void {
    this.errorMessage = '';
    this.loading = true;

    this.apiService.login(this.email, this.password).subscribe({
      next: (response) => {
        this.loading = false;
        this.authStore.login(
          response.access_token,
          response.role || 'investor',
          response.email || this.email,
          response.full_name || ''
        );
        if (response.role === 'admin') {
          this.router.navigate(['/admin']);
        } else {
          this.router.navigate(['/dashboard']);
        }
      },
      error: (err) => {
        this.loading = false;
        this.errorMessage =
          err.error?.detail || 'Error al iniciar sesion. Verifica tus credenciales.';
      }
    });
  }

  onRegister(): void {
    this.errorMessage = '';
    this.successMessage = '';
    this.loading = true;

    this.apiService
      .register(this.registerEmail, this.registerPassword, this.registerName)
      .subscribe({
        next: () => {
          this.loading = false;
          this.successMessage = 'Registro exitoso. Iniciando sesion...';
          // Auto-login despues de registrar
          this.email = this.registerEmail;
          this.password = this.registerPassword;
          this.onSubmit();
        },
        error: (err) => {
          this.loading = false;
          this.errorMessage =
            err.error?.detail || 'Error al registrarse. Intenta de nuevo.';
        }
      });
  }

  toggleForm(): void {
    this.showRegister = !this.showRegister;
    this.errorMessage = '';
    this.successMessage = '';
  }
}
