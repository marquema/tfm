import { Component, OnInit } from '@angular/core';
import { ApiService } from '../../services/api.service';

interface Asset {
  ticker: string;
  nombre: string;
  categoria: string;
  sector: string;
  instrumento: string;
  descripcion: string;
}

@Component({
  selector: 'app-universe',
  standalone: true,
  templateUrl: './universe.component.html',
  styleUrl: './universe.component.scss'
})
export class UniverseComponent implements OnInit {
  assets: Asset[] = [];
  loading = false;
  errorMessage = '';

  constructor(private apiService: ApiService) {}

  ngOnInit(): void {
    this.loadUniverse();
  }

  loadUniverse(): void {
    this.loading = true;
    this.errorMessage = '';
    this.apiService.getUniverso('core').subscribe({
      next: (data) => {
        this.assets = data.activos ?? data ?? [];
        this.loading = false;
      },
      error: (err) => {
        this.errorMessage = 'Error al cargar el universo de activos. Verifica que el backend este activo.';
        this.loading = false;
        console.error('Error cargando universo:', err);
      }
    });
  }
}
