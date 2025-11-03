import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Document {
  filename: string;
  path: string;
  last_modified?: string;
}

@Injectable({
  providedIn: 'root'
})
export class KnowledgeBaseService {
  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) { }

  queryKnowledgeBase(query: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/query`, { query });
  }

  getDocuments(): Observable<Document[]> {
    return this.http.get<Document[]>(`${this.apiUrl}/documents`);
  }
/*
  getViewUrl(filename: string): string {
    return `${this.apiUrl}/view/${filename}`;
  }
*/
  getDownloadUrl(filename: string): string {
    return `${this.apiUrl}/download/${filename}`;
  }
}