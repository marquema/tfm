import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'dashboard',
    pathMatch: 'full'
  },
  {
    path: 'dashboard',
    loadComponent: () =>
      import('./pages/dashboard/dashboard.component').then(m => m.DashboardComponent)
  },
  {
    path: 'universo',
    loadComponent: () =>
      import('./pages/universe/universe.component').then(m => m.UniverseComponent)
  },
  {
    path: 'estado',
    loadComponent: () =>
      import('./pages/status/status.component').then(m => m.StatusComponent)
  },
  {
    path: 'login',
    loadComponent: () =>
      import('./pages/login/login.component').then(m => m.LoginComponent)
  },
  {
    path: '**',
    redirectTo: 'dashboard'
  }
];
