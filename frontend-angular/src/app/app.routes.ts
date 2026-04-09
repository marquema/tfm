import { Routes } from '@angular/router';
import { authGuard } from './guards/auth.guard';
import { adminGuard } from './guards/auth.guard';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'dashboard',
    pathMatch: 'full'
  },
  {
    path: 'login',
    loadComponent: () =>
      import('./pages/login/login.component').then(m => m.LoginComponent)
  },
  {
    path: 'dashboard',
    loadComponent: () =>
      import('./pages/dashboard/dashboard.component').then(m => m.DashboardComponent),
    canActivate: [authGuard]
  },
  {
    path: 'universo',
    loadComponent: () =>
      import('./pages/universe/universe.component').then(m => m.UniverseComponent),
    canActivate: [authGuard]
  },
  {
    path: 'estado',
    loadComponent: () =>
      import('./pages/status/status.component').then(m => m.StatusComponent)
  },
  {
    path: 'investor/simulator',
    loadComponent: () =>
      import('./pages/investor/simulator/simulator.component').then(m => m.SimulatorComponent),
    canActivate: [authGuard]
  },
  {
    path: 'investor/results',
    loadComponent: () =>
      import('./pages/investor/results/results.component').then(m => m.ResultsComponent),
    canActivate: [authGuard]
  },
  {
    path: 'admin',
    loadComponent: () =>
      import('./pages/admin/admin.component').then(m => m.AdminComponent),
    canActivate: [adminGuard]
  },
  {
    path: '**',
    redirectTo: 'dashboard'
  }
];
