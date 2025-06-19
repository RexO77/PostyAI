// Global State
let selectedTone = 'Professional';
let currentPost = null;
let postHistory = [];
let isGenerating = false;

// Initialize App
document.addEventListener('DOMContentLoaded', async function() {
    await initializeApp();
});

// Handle page visibility changes for better performance
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        // Page is hidden - could pause non-critical operations
    } else {
        // Page is visible - resume normal operations
    }
});

async function initializeApp() {
    console.log('🚀 Initializing PostyAI...');
    
    try {
        // Load initial data
        await loadTopics();
        await loadHistory();
        
        // Setup event listeners
        setupEventListeners();
        
        // Update statistics
        updateStats();
        
        // Show welcome state
        showState('welcome');
        
        console.log('✅ PostyAI initialized successfully');
    } catch (error) {
        console.error('❌ Failed to initialize PostyAI:', error);
        showNotification('Failed to initialize application. Please refresh the page.', 'error');
    }
}

// Load topics from API
async function loadTopics() {
    try {
        const response = await fetch('/api/tags');
        if (!response.ok) throw new Error('Failed to load topics');
        
        const data = await response.json();
        const topicSelect = document.getElementById('topic');
        
        // Clear existing options except the placeholder
        topicSelect.innerHTML = '<option value="">Choose a topic...</option>';
        
        // Add topics from the tags array
        data.tags.forEach(topic => {
            const option = document.createElement('option');
            option.value = topic;
            option.textContent = topic;
            topicSelect.appendChild(option);
        });
        
        console.log(`📋 Loaded ${data.tags.length} topics`);
    } catch (error) {
        console.error('Failed to load topics:', error);
        // Add default topics as fallback
        const defaultTopics = ['Technology', 'Business', 'Marketing', 'AI & ML', 'DevOps', 'Health', 'Education'];
        const topicSelect = document.getElementById('topic');
        
        defaultTopics.forEach(topic => {
            const option = document.createElement('option');
            option.value = topic;
            option.textContent = topic;
            topicSelect.appendChild(option);
        });
    }
}

// Load post history
async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        if (!response.ok) throw new Error('Failed to load history');
        
        postHistory = await response.json();
        renderHistory();
        
        console.log(`📚 Loaded ${postHistory.length} historical posts`);
    } catch (error) {
        console.error('Failed to load history:', error);
        postHistory = [];
        renderHistory();
    }
}

// Setup event listeners
function setupEventListeners() {
    // Form submission
    const form = document.getElementById('contentForm');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (isGenerating) return;
        
        const formData = new FormData(form);
        const topic = formData.get('topic');
        const length = formData.get('length');
        const language = formData.get('language');
        const tone = selectedTone;
        
        await generatePost(topic, length, language, tone);
    });
    
    // Tone selection
    document.querySelectorAll('.tone-option').forEach(option => {
        option.addEventListener('click', () => {
            // Remove selected class from all options
            document.querySelectorAll('.tone-option').forEach(opt => opt.classList.remove('selected'));
            
            // Add selected class to clicked option
            option.classList.add('selected');
            selectedTone = option.dataset.tone;
        });
    });
    
    // Action buttons
    document.addEventListener('click', (e) => {
        if (e.target.matches('[data-action]')) {
            const action = e.target.dataset.action;
            
            switch (action) {
                case 'copy':
                    copyToClipboard();
                    break;
                case 'regenerate':
                    regeneratePost();
                    break;
                case 'export':
                    exportPost();
                    break;
                case 'predict':
                    predictEngagement();
                    break;
                case 'analyze':
                    analyzeText();
                    break;
                case 'model-info':
                    showModelInfo();
                    break;
            }
        }
    });
}

// Generate post
async function generatePost(topic, length, language, tone) {
    if (isGenerating) return;
    
    isGenerating = true;
    showState('loading');
    
    const generateBtn = document.querySelector('.generate-btn');
    generateBtn.disabled = true;
    generateBtn.innerHTML = '<div class="spinner"></div> Generating...';
    
    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                tag: topic,
                length,
                language,
                tone
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        if (!result.success || !result.post) {
            throw new Error('Invalid response from server');
        }
        
        currentPost = {
            id: result.post.id,
            content: result.post.content,
            params: result.post.params,
            stats: result.post.stats,
            timestamp: result.post.timestamp,
            generation_time: result.post.generation_time
        };
        
        // Add to history
        postHistory.unshift(currentPost);
        if (postHistory.length > 50) {
            postHistory = postHistory.slice(0, 50);
        }
        
        // Save to localStorage
        localStorage.setItem('postHistory', JSON.stringify(postHistory));
        
        displayPost(currentPost);
        renderHistory();
        updateStats();
        
        showNotification('Post generated successfully!', 'success');
        
    } catch (error) {
        console.error('Generation failed:', error);
        showNotification(`Failed to generate post: ${error.message}`, 'error');
        showState('welcome');
    } finally {
        isGenerating = false;
        generateBtn.disabled = false;
        generateBtn.innerHTML = '<i data-lucide="zap"></i> Generate Post';
        
        // Re-initialize Lucide icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    }
}

// Display post
function displayPost(post) {
    showState('result');
    
    const contentDisplay = document.querySelector('.content-display');
    const resultState = contentDisplay.querySelector('.result-state');
    
    if (!resultState) {
        const newResultState = document.createElement('div');
        newResultState.className = 'content-state result-state';
        newResultState.innerHTML = `
            <div class="post-content">${post.content}</div>
            <div class="analytics">
                <div class="analytic-item">
                    <div class="analytic-value">${post.analytics?.engagement || 'N/A'}</div>
                    <div class="analytic-label">Engagement</div>
                </div>
                <div class="analytic-item">
                    <div class="analytic-value">${post.analytics?.readability || 'N/A'}</div>
                    <div class="analytic-label">Readability</div>
                </div>
                <div class="analytic-item">
                    <div class="analytic-value">${post.analytics?.sentiment || 'N/A'}</div>
                    <div class="analytic-label">Sentiment</div>
                </div>
                <div class="analytic-item">
                    <div class="analytic-value">${post.analytics?.word_count || 'N/A'}</div>
                    <div class="analytic-label">Words</div>
                </div>
                <div class="analytic-item">
                    <div class="analytic-value">${post.analytics?.char_count || 'N/A'}</div>
                    <div class="analytic-label">Characters</div>
                </div>
                <div class="analytic-item">
                    <div class="analytic-value">${post.analytics?.estimated_read_time || 'N/A'}</div>
                    <div class="analytic-label">Read Time</div>
                </div>
            </div>
        `;
        contentDisplay.querySelector('.content-states').appendChild(newResultState);
    } else {
        resultState.querySelector('.post-content').textContent = post.content;
        
        // Update analytics
        const analytics = resultState.querySelector('.analytics');
        if (analytics && post.stats) {
            analytics.innerHTML = `
                <div class="analytic-item">
                    <div class="analytic-value">${post.stats.word_count || 'N/A'}</div>
                    <div class="analytic-label">Words</div>
                </div>
                <div class="analytic-item">
                    <div class="analytic-value">${post.stats.reading_time || 'N/A'}</div>
                    <div class="analytic-label">Min Read</div>
                </div>
                <div class="analytic-item">
                    <div class="analytic-value">${post.stats.engagement_score || 'N/A'}/10</div>
                    <div class="analytic-label">Engagement</div>
                </div>
                <div class="analytic-item">
                    <div class="analytic-value">${post.generation_time || 'N/A'}s</div>
                    <div class="analytic-label">Generated</div>
                </div>
            `;
        }
    }
}

// Show state
function showState(state) {
    const states = document.querySelectorAll('.content-state');
    states.forEach(s => s.classList.add('hidden'));
    
    const targetState = document.querySelector(`.${state}-state`);
    if (targetState) {
        targetState.classList.remove('hidden');
    }
    
    // Show/hide action buttons based on state
    const actionButtons = document.querySelectorAll('.content-actions .action-btn');
    if (state === 'result') {
        // Show all action buttons when displaying results
        actionButtons.forEach(btn => btn.style.display = 'flex');
    } else {
        // Hide action buttons for other states
        actionButtons.forEach(btn => btn.style.display = 'none');
    }
}

// Copy to clipboard
async function copyToClipboard() {
    if (!currentPost) return;
    
    try {
        await navigator.clipboard.writeText(currentPost.content);
        showNotification('Post copied to clipboard!', 'success');
    } catch (error) {
        console.error('Failed to copy:', error);
        showNotification('Unable to copy to clipboard. Please select and copy the text manually.', 'warning');
    }
}

// Regenerate post
async function regeneratePost() {
    if (!currentPost || isGenerating) return;
    
    const params = currentPost.params;
    if (!params) return;
    
    await generatePost(params.tag, params.length, params.language, params.tone);
}

// Export post
async function exportPost() {
    if (!currentPost) return;
    
    try {
        const response = await fetch('/api/export', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                post: currentPost
            })
        });
        
        if (!response.ok) throw new Error('Export failed');
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `post-${currentPost.id.substring(0, 8)}.txt`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showNotification('Post exported successfully!', 'success');
    } catch (error) {
        console.error('Export failed:', error);
        showNotification('Unable to download file. Please copy the post text manually.', 'warning');
    }
}

// Update statistics
function updateStats() {
    const totalPosts = postHistory.length;
    const totalTopics = new Set(postHistory.map(p => p.topic)).size;
    const avgTime = postHistory.length > 0 ? '2.4s' : '0s';
    
    document.querySelector('.stats').innerHTML = `
        <div class="stat-item">
            <span class="stat-value">${totalPosts}</span>
            <span>Posts</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">${totalTopics}</span>
            <span>Topics</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">${avgTime}</span>
            <span>Avg Time</span>
        </div>
    `;
}

// Render history
function renderHistory() {
    const historyList = document.querySelector('.history-list');
    
    if (postHistory.length === 0) {
        historyList.innerHTML = `
            <div style="padding: 40px; text-align: center; color: var(--text-muted);">
                <p>No posts generated yet</p>
                <p style="font-size: 12px; margin-top: 8px;">Your generated posts will appear here</p>
            </div>
        `;
        return;
    }
    
    historyList.innerHTML = postHistory.map(post => `
        <div class="history-item" onclick="loadHistoryPost('${post.id}')">
            <div class="history-meta">
                <span class="history-tag">${post.params?.tag || 'Unknown'}</span>
                <span class="history-time">${new Date(post.timestamp).toLocaleTimeString()}</span>
            </div>
            <div class="history-preview">${(post.content || '').substring(0, 80)}...</div>
        </div>
    `).join('');
}

// Load history post
function loadHistoryPost(postId) {
    const post = postHistory.find(p => p.id === postId);
    if (post) {
        currentPost = post;
        displayPost(post);
        
        // Update form values
        document.getElementById('topic').value = post.params?.tag || '';
        document.getElementById('length').value = post.params?.length || '';
        document.getElementById('language').value = post.params?.language || '';
        
        // Update tone selection
        document.querySelectorAll('.tone-option').forEach(opt => opt.classList.remove('selected'));
        const toneOption = document.querySelector(`[data-tone="${post.params?.tone}"]`);
        if (toneOption) {
            toneOption.classList.add('selected');
            selectedTone = post.params.tone;
        }
    }
}

// Clear history
function clearHistory() {
    postHistory = [];
    localStorage.removeItem('postHistory');
    renderHistory();
    updateStats();
    showNotification('History cleared successfully!', 'success');
}

// Batch generation
function showBatchModal() {
    const modal = document.getElementById('batchModal');
    modal.classList.add('show');
}

function closeBatchModal() {
    const modal = document.getElementById('batchModal');
    modal.classList.remove('show');
}

async function startBatchGeneration() {
    const count = parseInt(document.getElementById('batchCount').value);
    if (count < 1 || count > 10) {
        showNotification('Please enter a number between 1 and 10', 'warning');
        return;
    }
    
    closeBatchModal();
    showNotification(`Starting batch generation of ${count} posts...`, 'success');
    
    // Implementation would go here
    // This is a placeholder for the actual batch generation logic
}

// Show notification
function showNotification(message, type = 'success') {
    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.notification');
    existingNotifications.forEach(notification => notification.remove());
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    
    const icon = type === 'success' ? '✓' : type === 'error' ? '✗' : '⚠';
    
    notification.innerHTML = `
        <div class="notification-icon">${icon}</div>
        <div class="notification-text">${message}</div>
    `;
    
    document.body.appendChild(notification);
    
    // Trigger show animation
    setTimeout(() => notification.classList.add('show'), 100);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

// Close modal on outside click
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('modal')) {
        e.target.classList.remove('show');
    }
});

// ============================================================================
// ML FUNCTIONALITY
// ============================================================================

// Predict engagement for current post
async function predictEngagement() {
    if (!currentPost) {
        showNotification('No post available for prediction', 'warning');
        return;
    }
    
    try {
        const response = await fetch('/api/ml/predict-engagement', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                content: currentPost.content,
                tags: [currentPost.params?.tag],
                language: currentPost.params?.language
            })
        });
        
        if (!response.ok) throw new Error('Prediction failed');
        
        const result = await response.json();
        displayMLPrediction(result);
        showNotification('Engagement prediction completed!', 'success');
        
    } catch (error) {
        console.error('Prediction failed:', error);
        showNotification('Engagement prediction is temporarily unavailable. Your post is still ready to use!', 'warning');
    }
}

// Display ML prediction results
function displayMLPrediction(result) {
    const resultState = document.querySelector('.result-state');
    if (!resultState) return;
    
    // Remove existing ML panel
    const existingPanel = resultState.querySelector('.ml-analysis-panel');
    if (existingPanel) {
        existingPanel.remove();
    }
    
    const engagement = result.predicted_engagement || 0;
    const confidence = result.confidence_interval || [0, 0];
    const modelUsed = result.model_used || 'Unknown';
    const featuresUsed = result.features_used || 0;
    
    const mlPanel = document.createElement('div');
    mlPanel.className = 'ml-analysis-panel';
    mlPanel.innerHTML = `
        <div class="ml-section">
            <div class="ml-section-title">🎯 Engagement Prediction</div>
            <div class="engagement-prediction">
                <div class="engagement-main">
                    <span class="engagement-value">${Math.round(engagement)}</span>
                    <span class="engagement-unit">interactions</span>
                </div>
                <div class="engagement-confidence">Range: ${Math.round(confidence[0])} - ${Math.round(confidence[1])}</div>
                <div class="ml-model-info">Model: ${modelUsed} | Features: ${featuresUsed}</div>
            </div>
        </div>
        
        <div class="ml-section">
            <div class="ml-section-title">📊 Analysis</div>
            <div class="factors-list">
                <div class="factor-item">
                    <span class="factor-name">Predicted Performance</span>
                    <span class="factor-value">${engagement < 100 ? 'Low' : engagement < 300 ? 'Good' : engagement < 600 ? 'High' : 'Excellent'}</span>
                </div>
                <div class="factor-item">
                    <span class="factor-name">Confidence Level</span>
                    <span class="factor-value">${Math.round(((confidence[1] - confidence[0]) / engagement * 100)) || 40}% range</span>
                </div>
            </div>
        </div>
        
        <div class="ml-actions">
            <button class="ml-action-btn" onclick="analyzeText()">📈 Detailed Analysis</button>
            <button class="ml-action-btn" onclick="showModelInfo()">ℹ️ Model Info</button>
        </div>
    `;
    
    resultState.appendChild(mlPanel);
}

// Analyze text with detailed insights
async function analyzeText() {
    if (!currentPost) {
        showNotification('No post available for analysis', 'warning');
        return;
    }
    
    try {
        const response = await fetch('/api/ml/analyze-text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                content: currentPost.content
            })
        });
        
        if (!response.ok) throw new Error('Analysis failed');
        
        const analysis = await response.json();
        displayTextAnalysis(analysis);
        showNotification('Text analysis completed!', 'success');
        
    } catch (error) {
        console.error('Analysis failed:', error);
        showNotification('Text analysis is temporarily unavailable. Your post is still ready to use!', 'warning');
    }
}

// Display detailed text analysis
function displayTextAnalysis(analysis) {
    const resultState = document.querySelector('.result-state');
    if (!resultState) return;
    
    // Find or create ML panel
    let mlPanel = resultState.querySelector('.ml-analysis-panel');
    if (!mlPanel) {
        mlPanel = document.createElement('div');
        mlPanel.className = 'ml-analysis-panel';
        resultState.appendChild(mlPanel);
    }
    
    // Add text analysis section
    const textAnalysisSection = document.createElement('div');
    textAnalysisSection.className = 'ml-section';
    textAnalysisSection.innerHTML = `
        <div class="ml-section-title">🔍 Text Analysis</div>
        <div class="text-analysis-grid">
            <div class="analysis-item">
                <div class="analysis-value">${analysis.sentiment_score}</div>
                <div class="analysis-label">Sentiment</div>
            </div>
            <div class="analysis-item">
                <div class="analysis-value">${analysis.readability_score}</div>
                <div class="analysis-label">Readability</div>
            </div>
            <div class="analysis-item">
                <div class="analysis-value">${analysis.complexity_score}</div>
                <div class="analysis-label">Complexity</div>
            </div>
            <div class="analysis-item">
                <div class="analysis-value">${analysis.emotion_score}</div>
                <div class="analysis-label">Emotion</div>
            </div>
        </div>
    `;
    
    // Remove existing text analysis section if present
    const existingTextAnalysis = mlPanel.querySelector('.text-analysis-grid');
    if (existingTextAnalysis) {
        existingTextAnalysis.closest('.ml-section').remove();
    }
    
    mlPanel.appendChild(textAnalysisSection);
}

// Show model information modal
async function showModelInfo() {
    const modal = document.getElementById('modelInfoModal');
    modal.classList.add('show');
    await loadModelInfo();
}

// Close model info modal
function closeModelInfoModal() {
    const modal = document.getElementById('modelInfoModal');
    modal.classList.remove('show');
}

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch('/api/ml/model-info');
        if (!response.ok) throw new Error('Failed to load model info');
        
        const info = await response.json();
        displayModelInfo(info);
        
    } catch (error) {
        console.error('Failed to load model info:', error);
        document.getElementById('modelInfoContent').innerHTML = `
            <p style="color: var(--text-muted); text-align: center;">
                Failed to load model information
            </p>
        `;
    }
}

// Display model information
function displayModelInfo(info) {
    const content = document.getElementById('modelInfoContent');
    content.innerHTML = `
        <div style="margin-bottom: 20px;">
            <h4 style="color: var(--text-primary); margin-bottom: 8px;">Model Performance</h4>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                <div class="analysis-item">
                    <div class="analysis-value">${info.accuracy}%</div>
                    <div class="analysis-label">Accuracy</div>
                </div>
                <div class="analysis-item">
                    <div class="analysis-value">${info.precision}%</div>
                    <div class="analysis-label">Precision</div>
                </div>
            </div>
        </div>
        
        <div style="margin-bottom: 20px;">
            <h4 style="color: var(--text-primary); margin-bottom: 8px;">Training Data</h4>
            <p style="color: var(--text-secondary); font-size: 14px;">
                ${info.training_samples} samples trained on ${info.features} features
            </p>
        </div>
        
        <div>
            <h4 style="color: var(--text-primary); margin-bottom: 8px;">Model Details</h4>
            <p style="color: var(--text-secondary); font-size: 14px;">
                Algorithm: ${info.algorithm}<br>
                Last Updated: ${new Date(info.last_updated).toLocaleDateString()}<br>
                Version: ${info.version}
            </p>
        </div>
    `;
}

// Refresh model info
async function refreshModelInfo() {
    await loadModelInfo();
    showNotification('Model information refreshed!', 'success');
}

// Override the displayPost function to include ML features
const originalDisplayPost = displayPost;
displayPost = function(post) {
    originalDisplayPost(post);
    
    // Add ML prediction automatically for new posts
    if (post && post.content) {
        setTimeout(() => {
            predictEngagement();
        }, 1000);
    }
};

// Global error handler for unhandled promise rejections
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showNotification('An unexpected error occurred. Please try again.', 'error');
    event.preventDefault();
});

// Global error handler for JavaScript errors
window.addEventListener('error', function(event) {
    console.error('JavaScript error:', event.error);
    showNotification('An unexpected error occurred. Please refresh the page.', 'error');
});
