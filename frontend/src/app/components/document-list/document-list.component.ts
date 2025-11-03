import { Component, OnInit } from '@angular/core';
import { KnowledgeBaseService, Document } from '../../services/knowledge-base.service';
import { interval } from 'rxjs';
import { switchMap } from 'rxjs/operators';

@Component({
  selector: 'app-document-list',
  templateUrl: './document-list.component.html',
  styleUrls: ['./document-list.component.css']
})
export class DocumentListComponent implements OnInit {
  documents: Document[] = [];
  isLoading: boolean = true;
  localStorageItems: {key: string, value: string}[] = [];

  constructor(private knowledgeBaseService: KnowledgeBaseService) {}

  ngOnInit(): void {
    // Load documents initially
    this.loadDocuments();
    
    // Load localStorage items
    this.loadLocalStorageItems();
    
    // Refresh document list every 10 seconds
    interval(10000).pipe(
      switchMap(() => this.knowledgeBaseService.getDocuments())
    ).subscribe({
      next: (documents) => {
        this.documents = documents;
      },
      error: (error) => {
        console.error('Error refreshing documents:', error);
      }
    });
  }

  loadDocuments(): void {
    this.isLoading = true;
    this.knowledgeBaseService.getDocuments().subscribe({
      next: (documents) => {
        this.documents = documents;
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Error loading documents:', error);
        this.isLoading = false;
      }
    });
  }
/*
  getViewUrl(filename: string): string {
    return this.knowledgeBaseService.getViewUrl(filename);
  } */

  getDownloadUrl(filename: string): string {
    return this.knowledgeBaseService.getDownloadUrl(filename);
  }

  loadLocalStorageItems(): void {
    this.localStorageItems = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key) {
        const value = localStorage.getItem(key) || '';
        this.localStorageItems.push({ key, value });
      }
    }
  }
}