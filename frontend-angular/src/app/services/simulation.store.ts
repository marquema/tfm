import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class SimulationStore {
  results: any | null = null;

  setResults(data: any): void {
    this.results = data;
  }

  getResults(): any | null {
    return this.results;
  }
}
