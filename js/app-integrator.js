/**
 * MR Bot Neural Language Teacher - Application Integrator
 * 
 * This module integrates all components of the neural language learning system:
 * - Neural Language Model
 * - Neural Data Manager
 * - Neural Conversation Engine
 * - UI Components
 */

class NeuralAppIntegrator {
    constructor() {
        this.initialized = false;
        this.components = {
            languageModel: null,
            dataManager: null,
            conversationEngine: null
        };
        
        // UI elements
        this.ui = {
            chatContainer: null,
            userInput: null,
            sendButton: null,
            languageToggle: null,
            feedbackButtons: null,
            correctionInput: null,
            loadingIndicator: null,
            statusDisplay: null
        };
        
        // App state
        this.state = {
            isProcessing: false,
            currentLanguage: 'en',
            lastUserMessage: null,
            lastBotResponse: null,
            isCorrectionMode: false
        };
    }

    /**
     * Initialize the application
     */
    async initialize() {
        console.log("Initializing Neural MR Bot Application...");
        
        try {
            // Load TensorFlow.js
            await this.loadTensorFlow();
            
            // Initialize components
            await this.initializeComponents();
            
            // Set up UI
            this.setupUI();
            
            // Set up event listeners
            this.setupEventListeners();
            
            this.initialized = true;
            console.log("Neural MR Bot Application initialized successfully");
            
            // Add welcome message
            this.addBotMessage(this.getWelcomeMessage());
            
            return true;
        } catch (error) {
            console.error("Error initializing application:", error);
            this.addBotMessage("Error initializing the application. Please refresh the page and try again.");
            return false;
        }
    }

    /**
     * Load TensorFlow.js library
     */
    async loadTensorFlow() {
        console.log("Loading TensorFlow.js...");
        
        return new Promise((resolve, reject) => {
            // Check if TensorFlow.js is already loaded
            if (typeof tf !== 'undefined') {
                console.log("TensorFlow.js already loaded");
                resolve();
                return;
            }
            
            // Create script element
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.18.0/dist/tf.min.js';
            script.async = true;
            
            script.onload = () => {
                console.log("TensorFlow.js loaded successfully");
                resolve();
            };
            
            script.onerror = () => {
                const error = new Error("Failed to load TensorFlow.js");
                console.error(error);
                reject(error);
            };
            
            document.head.appendChild(script);
        });
    }

    /**
     * Initialize all components
     */
    async initializeComponents() {
        console.log("Initializing components...");
        
        // Initialize data manager
        this.components.dataManager = new NeuralDataManager();
        await this.components.dataManager.initialize();
        
        // Initialize language model
        this.components.languageModel = new NeuralLanguageModel();
        await this.components.languageModel.initialize(this.components.dataManager);
        
        // Initialize conversation engine
        this.components.conversationEngine = new NeuralConversationEngine();
        await this.components.conversationEngine.initialize(
            this.components.languageModel,
            this.components.dataManager
        );
        
        console.log("All components initialized");
    }

    /**
     * Set up UI elements
     */
    setupUI() {
        console.log("Setting up UI...");
        
        // Get UI elements
        this.ui.chatContainer = document.getElementById('chat-container');
        this.ui.userInput = document.getElementById('user-input');
        this.ui.sendButton = document.getElementById('send-button');
        this.ui.languageToggle = document.getElementById('language-toggle');
        this.ui.loadingIndicator = document.getElementById('loading-indicator');
        this.ui.statusDisplay = document.getElementById('status-display');
        
        // Create feedback UI elements
        this.createFeedbackUI();
        
        // Update status display
        this.updateStatusDisplay();
    }

    /**
     * Create feedback UI elements
     */
    createFeedbackUI() {
        // Create feedback container
        const feedbackContainer = document.createElement('div');
        feedbackContainer.id = 'feedback-container';
        feedbackContainer.className = 'hidden p-4 bg-gray-100 rounded-lg mt-4';
        
        // Create feedback buttons
        const buttonContainer = document.createElement('div');
        buttonContainer.className = 'flex space-x-2 mb-2';
        
        const correctButton = document.createElement('button');
        correctButton.id = 'correct-button';
        correctButton.className = 'bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600';
        correctButton.textContent = 'Correct';
        
        const incorrectButton = document.createElement('button');
        incorrectButton.id = 'incorrect-button';
        incorrectButton.className = 'bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600';
        incorrectButton.textContent = 'Incorrect';
        
        buttonContainer.appendChild(correctButton);
        buttonContainer.appendChild(incorrectButton);
        
        // Create correction input
        const correctionContainer = document.createElement('div');
        correctionContainer.id = 'correction-container';
        correctionContainer.className = 'hidden mt-2';
        
        const correctionInput = document.createElement('textarea');
        correctionInput.id = 'correction-input';
        correctionInput.className = 'w-full p-2 border rounded';
        correctionInput.placeholder = 'Enter the correct response...';
        
        const submitCorrectionButton = document.createElement('button');
        submitCorrectionButton.id = 'submit-correction';
        submitCorrectionButton.className = 'bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 mt-2';
        submitCorrectionButton.textContent = 'Submit Correction';
        
        correctionContainer.appendChild(correctionInput);
        correctionContainer.appendChild(submitCorrectionButton);
        
        // Add to feedback container
        feedbackContainer.appendChild(buttonContainer);
        feedbackContainer.appendChild(correctionContainer);
        
        // Add to document
        document.body.appendChild(feedbackContainer);
        
        // Store UI elements
        this.ui.feedbackButtons = {
            container: feedbackContainer,
            correctButton: correctButton,
            incorrectButton: incorrectButton
        };
        
        this.ui.correctionInput = {
            container: correctionContainer,
            input: correctionInput,
            submitButton: submitCorrectionButton
        };
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        console.log("Setting up event listeners...");
        
        // Send button click
        this.ui.sendButton.addEventListener('click', () => {
            this.handleUserInput();
        });
        
        // Enter key press in input field
        this.ui.userInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                this.handleUserInput();
            }
        });
        
        // Language toggle
        if (this.ui.languageToggle) {
            this.ui.languageToggle.addEventListener('change', (event) => {
                this.state.currentLanguage = event.target.checked ? 'bn' : 'en';
                this.components.conversationEngine.setTargetLanguage(this.state.currentLanguage);
                
                // Update UI language
                document.documentElement.lang = this.state.currentLanguage;
                this.updatePlaceholder();
            });
        }
        
        // Feedback buttons
        this.ui.feedbackButtons.correctButton.addEventListener('click', () => {
            this.hideFeedbackUI();
            this.showSuccessMessage("Thank you for your feedback!");
        });
        
        this.ui.feedbackButtons.incorrectButton.addEventListener('click', () => {
            this.showCorrectionInput();
        });
        
        // Submit correction button
        this.ui.correctionInput.submitButton.addEventListener('click', () => {
            this.submitCorrection();
        });
    }

    /**
     * Handle user input
     */
    async handleUserInput() {
        if (this.state.isProcessing) return;
        
        const userInput = this.ui.userInput.value.trim();
        if (!userInput) return;
        
        // Clear input field
        this.ui.userInput.value = '';
        
        // Add user message to chat
        this.addUserMessage(userInput);
        
        // Store last user message
        this.state.lastUserMessage = userInput;
        
        // Show loading indicator
        this.showLoading(true);
        this.state.isProcessing = true;
        
        try {
            // Process user input
            const response = await this.components.conversationEngine.processInput(userInput);
            
            // Store last bot response
            this.state.lastBotResponse = response.text;
            
            // Add bot message to chat
            this.addBotMessage(response.text);
            
            // Show feedback UI
            this.showFeedbackUI();
            
            // Update status display
            this.updateStatusDisplay();
        } catch (error) {
            console.error("Error processing user input:", error);
            this.addBotMessage("I'm having trouble processing your request. Please try again.");
        } finally {
            // Hide loading indicator
            this.showLoading(false);
            this.state.isProcessing = false;
        }
    }

    /**
     * Add user message to chat
     * @param {string} message - User message
     */
    addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'flex justify-end mb-4';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'bg-blue-500 text-white rounded-lg py-2 px-4 max-w-[70%]';
        messageContent.textContent = message;
        
        messageElement.appendChild(messageContent);
        this.ui.chatContainer.appendChild(messageElement);
        
        // Scroll to bottom
        this.scrollToBottom();
    }

    /**
     * Add bot message to chat
     * @param {string} message - Bot message
     */
    addBotMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'flex mb-4';
        
        const avatarElement = document.createElement('div');
        avatarElement.className = 'w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center mr-2';
        avatarElement.textContent = 'MR';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'bg-gray-200 rounded-lg py-2 px-4 max-w-[70%]';
        
        // Convert markdown to HTML (simple version)
        const formattedMessage = this.formatMessage(message);
        messageContent.innerHTML = formattedMessage;
        
        messageElement.appendChild(avatarElement);
        messageElement.appendChild(messageContent);
        this.ui.chatContainer.appendChild(messageElement);
        
        // Scroll to bottom
        this.scrollToBottom();
    }

    /**
     * Format message with markdown-like syntax
     * @param {string} message - Raw message
     * @returns {string} - Formatted HTML
     */
    formatMessage(message) {
        // Replace markdown-like syntax with HTML
        let formatted = message
            // Headers
            .replace(/^# (.*$)/gm, '<h1 class="text-xl font-bold mb-2">$1</h1>')
            .replace(/^## (.*$)/gm, '<h2 class="text-lg font-bold mb-1">$1</h2>')
            .replace(/^### (.*$)/gm, '<h3 class="text-md font-bold mb-1">$1</h3>')
            // Bold
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Italic
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Lists
            .replace(/^\- (.*$)/gm, '<li>$1</li>')
            // Line breaks
            .replace(/\n/g, '<br>');
        
        // Wrap lists in ul tags
        if (formatted.includes('<li>')) {
            formatted = formatted.replace(/(<li>.*?<\/li>)/gs, '<ul class="list-disc ml-4">$1</ul>');
        }
        
        return formatted;
    }

    /**
     * Scroll chat container to bottom
     */
    scrollToBottom() {
        this.ui.chatContainer.scrollTop = this.ui.chatContainer.scrollHeight;
    }

    /**
     * Show or hide loading indicator
     * @param {boolean} show - Whether to show loading
     */
    showLoading(show) {
        if (this.ui.loadingIndicator) {
            this.ui.loadingIndicator.style.display = show ? 'block' : 'none';
        }
        
        // Disable/enable send button
        if (this.ui.sendButton) {
            this.ui.sendButton.disabled = show;
        }
    }

    /**
     * Show feedback UI
     */
    showFeedbackUI() {
        this.ui.feedbackButtons.container.classList.remove('hidden');
    }

    /**
     * Hide feedback UI
     */
    hideFeedbackUI() {
        this.ui.feedbackButtons.container.classList.add('hidden');
        this.ui.correctionInput.container.classList.add('hidden');
    }

    /**
     * Show correction input
     */
    showCorrectionInput() {
        this.ui.correctionInput.container.classList.remove('hidden');
        this.ui.correctionInput.input.value = this.state.lastBotResponse || '';
        this.ui.correctionInput.input.focus();
    }

    /**
     * Submit correction
     */
    async submitCorrection() {
        const correctionText = this.ui.correctionInput.input.value.trim();
        if (!correctionText) return;
        
        try {
            // Submit feedback
            await this.components.conversationEngine.submitFeedback(
                this.state.lastUserMessage,
                this.state.lastBotResponse,
                correctionText
            );
            
            // Show success message
            this.showSuccessMessage("Thank you for your correction! I'll learn from it.");
            
            // Hide correction UI
            this.hideFeedbackUI();
            
            // Update last bot message in chat
            this.updateLastBotMessage(correctionText);
        } catch (error) {
            console.error("Error submitting correction:", error);
            this.showErrorMessage("Failed to submit correction. Please try again.");
        }
    }

    /**
     * Update the last bot message in the chat
     * @param {string} newText - New message text
     */
    updateLastBotMessage(newText) {
        const botMessages = this.ui.chatContainer.querySelectorAll('.flex:not(.justify-end)');
        if (botMessages.length > 0) {
            const lastBotMessage = botMessages[botMessages.length - 1];
            const messageContent = lastBotMessage.querySelector('div:nth-child(2)');
            if (messageContent) {
                messageContent.innerHTML = this.formatMessage(newText);
                messageContent.classList.add('bg-yellow-100');
                setTimeout(() => {
                    messageContent.classList.remove('bg-yellow-100');
                }, 2000);
            }
        }
    }

    /**
     * Show success message
     * @param {string} message - Success message
     */
    showSuccessMessage(message) {
        const successElement = document.createElement('div');
        successElement.className = 'bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative mb-4';
        successElement.textContent = message;
        
        this.ui.chatContainer.appendChild(successElement);
        this.scrollToBottom();
        
        // Remove after 3 seconds
        setTimeout(() => {
            successElement.remove();
        }, 3000);
    }

    /**
     * Show error message
     * @param {string} message - Error message
     */
    showErrorMessage(message) {
        const errorElement = document.createElement('div');
        errorElement.className = 'bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4';
        errorElement.textContent = message;
        
        this.ui.chatContainer.appendChild(errorElement);
        this.scrollToBottom();
        
        // Remove after 3 seconds
        setTimeout(() => {
            errorElement.remove();
        }, 3000);
    }

    /**
     * Update input placeholder based on language
     */
    updatePlaceholder() {
        if (this.ui.userInput) {
            this.ui.userInput.placeholder = this.state.currentLanguage === 'en' 
                ? 'Type your message...' 
                : 'আপনার বার্তা টাইপ করুন...';
        }
    }

    /**
     * Update status display
     */
    updateStatusDisplay() {
        if (!this.ui.statusDisplay) return;
        
        const trainingMetrics = this.components.languageModel.trainingMetrics;
        const userProfile = this.components.dataManager.getUserProfile();
        
        const statusHTML = `
            <div class="text-sm">
                <div><strong>Learning progress:</strong> ${trainingMetrics.totalExamples} examples</div>
                <div><strong>Accuracy:</strong> ${(trainingMetrics.accuracy * 100).toFixed(1)}%</div>
                <div><strong>Language:</strong> ${userProfile.targetLanguage === 'en' ? 'English' : 'Bengali'}</div>
                <div><strong>Level:</strong> ${userProfile.proficiencyLevel}</div>
            </div>
        `;
        
        this.ui.statusDisplay.innerHTML = statusHTML;
    }

    /**
     * Get welcome message
     * @returns {string} - Welcome message
     */
    getWelcomeMessage() {
        return `
# Welcome to MR Bot Neural Language Teacher!

I'm a self-learning AI designed to help you practice languages. I can:

- Have conversations in **English** and **Bengali**
- Learn from our interactions
- Improve based on your corrections
- Adapt to your learning needs

**Try saying:**
- "Hello, how are you?"
- "Let's practice Bengali"
- "Teach me some vocabulary"
- "What can you help me with?"

Type a message to get started!
`;
    }

    /**
     * Export trained model
     * @param {string} format - Export format
     */
    async exportModel(format = 'json') {
        try {
            const exportData = await this.components.languageModel.exportModel(format);
            
            if (!exportData) {
                throw new Error("Failed to export model");
            }
            
            // For JSON format, create a downloadable file
            if (format === 'json') {
                const dataStr = JSON.stringify(exportData);
                const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
                
                const exportFileDefaultName = 'mr-bot-model.json';
                
                const linkElement = document.createElement('a');
                linkElement.setAttribute('href', dataUri);
                linkElement.setAttribute('download', exportFileDefaultName);
                linkElement.click();
                
                return true;
            }
            
            return exportData;
        } catch (error) {
            console.error("Error exporting model:", error);
            return null;
        }
    }

    /**
     * Import trained model
     * @param {Object} importData - Import data
     */
    async importModel(importData) {
        try {
            const result = await this.components.languageModel.importModel(importData);
            
            if (result) {
                this.showSuccessMessage("Model imported successfully!");
                this.updateStatusDisplay();
            } else {
                this.showErrorMessage("Failed to import model");
            }
            
            return result;
        } catch (error) {
            console.error("Error importing model:", error);
            this.showErrorMessage("Error importing model: " + error.message);
            return false;
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create and initialize app
    window.app = new NeuralAppIntegrator();
    window.app.initialize().catch(error => {
        console.error("Failed to initialize app:", error);
    });
});