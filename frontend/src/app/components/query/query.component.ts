import { Component } from '@angular/core';
import { KnowledgeBaseService } from '../../services/knowledge-base.service';

@Component({
  selector: 'app-query',
  templateUrl: './query.component.html',
  styleUrls: ['./query.component.css']
})
export class QueryComponent {
  query: string = '';
  isLoading: boolean = false;
  errorMessage: string = '';
  messages: { role: 'user' | 'assistant'; content: string; sources?: string[] }[] = [];

  constructor(private knowledgeBaseService: KnowledgeBaseService) {}

  onKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.submitQuery();
    }
  }

  formatContent(content: string): string {
    return content.replace(/\n/g, '<br>');
  }

  submitQuery() {
    if (!this.query.trim()) {
      this.errorMessage = 'Please enter a query';
      return;
    }

    this.isLoading = true;
    this.errorMessage = '';
    const userMessage = this.query;
    this.messages.push({ role: 'user', content: userMessage });
    this.query = '';

    this.knowledgeBaseService.queryKnowledgeBase(userMessage).subscribe({
      next: (response) => {
        let answerText = '';
        let sources: string[] = [];
        if (response?.answer && response.answer.trim()) {
          answerText = response.answer;
          if (response.source_documents) {
            sources = response.source_documents;
          } else {
            const sourceMatches = answerText.match(/\(Source: .*?id:(.*?)\)/g);
            if (sourceMatches) {
              sources = [...new Set(sourceMatches.map((match: string) => {
                const idMatch = match.match(/id:(.*?)\)/);
                return idMatch ? idMatch[1].trim() : '';
              }))].filter((id: string) => id);
            }
          }
        } else {
          answerText = 'No relevant information found. Please rephrase or try another query.';
        }
        this.messages.push({ role: 'assistant', content: answerText, sources });
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Error querying knowledge base:', error);
        this.errorMessage = 'An error occurred while processing your query. Please try again.';
        this.isLoading = false;
      }
    });
  }
}