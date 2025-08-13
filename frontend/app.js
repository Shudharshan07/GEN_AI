class PDFQAApp {
    constructor() {
        this.baseUrl = 'http://localhost:8000';
        this.selectedFiles = [];
        this.chatHistory = [];
        this.documentsUploaded = false;
        
        this.initializeEventListeners();
        this.checkBackendHealth();
        this.loadStats();
    }

    initializeEventListeners() {
        // File upload
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const processBtn = document.getElementById('processBtn');
        const questionInput = document.getElementById('questionInput');
        const askBtn = document.getElementById('askBtn');

        // Drag and drop
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', this.handleDragOver.bind(this));
        dropZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
        dropZone.addEventListener('drop', this.handleDrop.bind(this));

        // File selection
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));

        // Process documents
        processBtn.addEventListener('click', this.processDocuments.bind(this));

        // Ask question
        askBtn.addEventListener('click', this.askQuestion.bind(this));
        questionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.askQuestion();
            }
        });
    }

    handleDragOver(e) {
        e.preventDefault();
        document.getElementById('dropZone').classList.add('drag-over');
    }

    handleDragLeave(e) {
        e.preventDefault();
        document.getElementById('dropZone').classList.remove('drag-over');
    }

    handleDrop(e) {
        e.preventDefault();
        document.getElementById('dropZone').classList.remove('drag-over');
        
        const files = Array.from(e.dataTransfer.files).filter(file => 
            file.type === 'application/pdf'
        );
        
        this.updateSelectedFiles(files);
    }

    handleFileSelect(e) {
        const files = Array.from(e.target.files);
        this.updateSelectedFiles(files);
    }

    updateSelectedFiles(files) {
        // Limit to 2 files
        if (files.length > 2) {
            this.showAlert('Maximum 2 PDF files allowed', 'error');
            files = files.slice(0, 2);
        }

        this.selectedFiles = files;
        this.renderUploadedFiles();
        
        const processBtn = document.getElementById('processBtn');
        processBtn.disabled = this.selectedFiles.length === 0;
    }

    renderUploadedFiles() {
        const container = document.getElementById('uploadedFiles');
        
        if (this.selectedFiles.length === 0) {
            container.innerHTML = '';
            return;
        }

        container.innerHTML = this.selectedFiles.map((file, index) => `
            <div class="file-item">
                <div class="file-info">
                    <span class="file-icon">📄</span>
                    <span>${file.name}</span>
                    <small>(${this.formatFileSize(file.size)})</small>
                </div>
                <button class="remove-file" onclick="app.removeFile(${index})">×</button>
            </div>
        `).join('');
    }

    removeFile(index) {
        this.selectedFiles.splice(index, 1);
        this.renderUploadedFiles();
        
        const processBtn = document.getElementById('processBtn');
        processBtn.disabled = this.selectedFiles.length === 0;
    }

    async processDocuments() {
        if (this.selectedFiles.length === 0) return;

        const processBtn = document.getElementById('processBtn');
        processBtn.disabled = true;
        processBtn.innerHTML = '<div class="spinner"></div> Processing...';

        try {
            const formData = new FormData();
            this.selectedFiles.forEach(file => {
                formData.append('files', file);
            });

            const response = await fetch(`${this.baseUrl}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Upload failed');
            }

            const result = await response.json();
            
            this.showAlert(`✅ Successfully processed ${result.files_processed.length} files with ${result.total_chunks} chunks`, 'success');
            
            this.documentsUploaded = true;
            this.enableQuestionInput();
            this.loadStats();
            this.clearChatHistory();

            // Clear uploaded files
            this.selectedFiles = [];
            this.renderUploadedFiles();

        } catch (error) {
            console.error('Upload error:', error);
            this.showAlert(`❌ Error: ${error.message}`, 'error');
        } finally {
            processBtn.disabled = false;
            processBtn.innerHTML = '🔄 Process Documents';
        }
    }

    enableQuestionInput() {
        const questionInput = document.getElementById('questionInput');
        const askBtn = document.getElementById('askBtn');
        
        questionInput.disabled = false;
        askBtn.disabled = false;
        
        // Clear the no-documents message
        const chatHistory = document.getElementById('chatHistory');
        const noDocsMessage = chatHistory.querySelector('.no-documents');
        if (noDocsMessage) {
            chatHistory.innerHTML = '';
        }
    }

    async askQuestion() {
        const questionInput = document.getElementById('questionInput');
        const question = questionInput.value.trim();
        
        if (!question) {
            this.showAlert('Please enter a question', 'error');
            return;
        }

        const askBtn = document.getElementById('askBtn');
        askBtn.disabled = true;
        askBtn.innerHTML = '<div class="spinner"></div> Thinking...';

        // Add question to chat immediately
        this.addMessageToChat(question, null);

        try {
            const response = await fetch(`${this.baseUrl}/ask`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: question,
                    top_k: 5,
                    model: "gemini-2.0-flash-exp"
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Question processing failed');
            }

            const result = await response.json();
            
            // Update the last message with the answer
            this.updateLastMessageWithAnswer(result);
            
            questionInput.value = '';

        } catch (error) {
            console.error('Question error:', error);
            this.updateLastMessageWithAnswer({
                answer: `❌ Error: ${error.message}`,
                sources: [],
                confidence: 0,
                processing_time: 0,
                model_used: 'error'
            });
        } finally {
            askBtn.disabled = false;
            askBtn.innerHTML = '🔍 Ask Question';
        }
    }

    addMessageToChat(question, answer) {
        const chatHistory = document.getElementById('chatHistory');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message';
        messageDiv.innerHTML = `
            <div class="question">Q: ${question}</div>
            <div class="answer" id="answer-${Date.now()}">
                <div class="loading">
                    <div class="spinner"></div>
                    <span>Generating answer...</span>
                </div>
            </div>
        `;
        
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        return messageDiv;
    }

    updateLastMessageWithAnswer(result) {
        const chatHistory = document.getElementById('chatHistory');
        const lastMessage = chatHistory.lastElementChild;
        const answerDiv = lastMessage.querySelector('.answer');
        
        const confidenceClass = result.confidence > 0.7 ? 'high' : 
                                result.confidence > 0.4 ? 'medium' : 'low';
        
        answerDiv.innerHTML = `
            ${result.answer}
            <div class="answer-meta">
                <span class="confidence ${confidenceClass}">
                    Confidence: ${(result.confidence * 100).toFixed(1)}%
                </span>
                <span>⏱️ ${result.processing_time}s</span>
                <span>🤖 ${result.model_used}</span>
                <span>📄 ${result.sources.length} sources</span>
            </div>
        `;

        // Add sources
        if (result.sources && result.sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources';
            sourcesDiv.innerHTML = `
                <h4>📚 Sources:</h4>
                ${result.sources.map(source => `
                    <div class="source-item">
                        <div class="source-header">
                            [Source ${source.source_id}] ${source.filename}, Page ${source.page}
                            <small>(Relevance: ${(source.relevance_score * 100).toFixed(1)}%)</small>
                        </div>
                        <div class="source-preview">${source.preview}</div>
                    </div>
                `).join('')}
            `;
            answerDiv.appendChild(sourcesDiv);
        }
        
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    clearChatHistory() {
        const chatHistory = document.getElementById('chatHistory');
        chatHistory.innerHTML = '';
    }

    async loadStats() {
        try {
            const response = await fetch(`${this.baseUrl}/stats`);
            if (!response.ok) return;
            
            const stats = await response.json();
            this.renderStats(stats);
        } catch (error) {
            console.error('Stats error:', error);
        }
    }

    renderStats(stats) {
        const statsCard = document.getElementById('statsCard');
        const statsGrid = document.getElementById('statsGrid');
        
        if (stats.total_chunks === 0) {
            statsCard.style.display = 'none';
            return;
        }

        statsCard.style.display = 'block';
        
        statsGrid.innerHTML = `
            <div class="stat-item">
                <div class="stat-value">${stats.total_chunks}</div>
                <div class="stat-label">Chunks</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${stats.total_documents}</div>
                <div class="stat-label">Documents</div>
            </div>
        `;
        
        if (stats.documents && stats.documents.length > 0) {
            const docsInfo = stats.documents.map(doc => 
                `${doc.filename} (${doc.total_pages}p, ${doc.total_chunks}c)`
            ).join('<br>');
            
            statsGrid.innerHTML += `
                <div class="stat-item" style="grid-column: span 2;">
                    <div class="stat-label">Loaded Documents</div>
                    <div style="font-size: 0.9rem; margin-top: 5px;">${docsInfo}</div>
                </div>
            `;
        }
    }

    async checkBackendHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            if (response.ok) {
                const health = await response.json();
                if (health.total_chunks > 0) {
                    this.documentsUploaded = true;
                    this.enableQuestionInput();
                }
            }
        } catch (error) {
            this.showAlert('⚠️ Backend not available. Make sure the FastAPI server is running on port 8000.', 'error');
        }
    }

    showAlert(message, type) {
        // Remove existing alerts
        const existingAlerts = document.querySelectorAll('.alert');
        existingAlerts.forEach(alert => alert.remove());
        
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert ${type}`;
        alertDiv.textContent = message;
        
        const container = document.querySelector('.container');
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Initialize the app
const app = new PDFQAApp();