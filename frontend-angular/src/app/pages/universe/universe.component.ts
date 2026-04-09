import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-universe',
  standalone: true,
  templateUrl: './universe.component.html',
  styleUrl: './universe.component.scss'
})
export class UniverseComponent implements OnInit {
  assets: any[] = [];
  loading = false;
  errorMessage = '';

  constructor(private api: ApiService, private cdr: ChangeDetectorRef) {}

  ngOnInit(): void {
    this.loadUniverse();
  }

  loadUniverse(): void {
    this.loading = true;
    this.errorMessage = '';
    this.api.getUniverso('core').subscribe({
      next: (data) => {
        this.loading = false;
        this.assets = Array.isArray(data) ? data : [];
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.loading = false;
        this.errorMessage = 'Error al cargar el universo de activos. Verifica que el backend esté activo.';
        this.cdr.detectChanges();
      }
    });
  }
}
