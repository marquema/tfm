import { Component, OnInit } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { AuthStore } from './services/auth.store';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App implements OnInit {
  readonly currentYear = new Date().getFullYear();
  sidebarCollapsed = false;

  constructor(public authStore: AuthStore) {}

  ngOnInit(): void {
    // Verificar si el token almacenado sigue siendo válido al arrancar.
    // Si el token expiró, limpia la sesión y el usuario verá el login.
    this.authStore.validateSession();
  }

  toggleSidebar(): void {
    this.sidebarCollapsed = !this.sidebarCollapsed;
  }

  logout(): void {
    this.authStore.logout();
  }
}
