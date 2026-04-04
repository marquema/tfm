import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss'
})
export class LoginComponent {
  email = '';
  password = '';

  onSubmit(): void {
    // Funcionalidad de autenticacion pendiente de implementar
    console.log('Login intentado con:', this.email);
  }
}
