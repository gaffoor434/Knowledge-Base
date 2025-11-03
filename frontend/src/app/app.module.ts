import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { RouterModule, Routes } from '@angular/router';

import { AppComponent } from './app.component';
import { QueryComponent } from './components/query/query.component';
import { DocumentListComponent } from './components/document-list/document-list.component';

const routes: Routes = [
  { path: '', component: QueryComponent },
  { path: 'chat', component: QueryComponent },
  { path: 'documents', component: DocumentListComponent },
  { path: '**', redirectTo: '' }
];

@NgModule({
  declarations: [
    AppComponent,
    QueryComponent,
    DocumentListComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    FormsModule,
    RouterModule.forRoot(routes)
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }