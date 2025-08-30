/**
 * MR Bot Neural Language Teacher - Enhanced Application Integrator
 * 
 * This module integrates all components of the neural language learning system:
 * - Neural Language Model (Transformer-based)
 * - Enhanced Neural Data Manager
 * - Neural Conversation Engine
 * - UI Components with multi-language support and speech features
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
            languageSelector: null,  // Changed from languageToggle to languageSelector
            feedbackButtons: null,
            correctionInput: null,
            loadingIndicator: null,
            statusDisplay: null,
            micButton: null,         // New: microphone button
            speakerButton: null      // New: speaker button
        };
        
        // App state
        this.state = {
            isProcessing: false,
            currentLanguage: 'en',
            lastUserMessage: null,
            lastBotResponse: null,
            isCorrectionMode: false,
            isListening: false,      // New: speech recognition state
            isSpeaking: false,       // New: speech synthesis state
            supportedLanguages: [],  // New: list of supported languages
            offlineMode: false       // New: offline mode indicator
        };
        
        // Speech recognition and synthesis
        this.speech = {
            recognition: null,
            synthesis: null,
            voices: []
        };
    }

    /**
     * Initialize the application
     */
    async initialize() {
        console.log("Initializing Enhanced Neural MR Bot Application...");

       // try {
            // *** START: নতুন লাইন যোগ করুন ***
            // TensorFlow.js-কে WebGL ব্যবহার না করে শুধুমাত্র CPU ব্যবহার করতে বাধ্য করা হচ্ছে
            // এটি Shader Compilation Error ঠিক করবে।
            await tf.setBackend('cpu');
            console.log("TensorFlow.js backend is set to CPU.");
            // *** END: নতুন লাইন যোগ করুন ***

            // Check for offline mode
            this.state.offlineMode = !navigator.onLine;

            // Load TensorFlow.js (যদিও আমরা এটি ব্যবহার না-ও করতে পারি, এটি থাকা ভালো)
            await this.loadTensorFlow();

            // Initialize components
            await this.initializeComponents();

            // Set up UI
            this.setupUI();

            // Set up event listeners
            this.setupEventListeners();

            // Initialize speech recognition and synthesis
            this.initializeSpeech();

            this.initialized = true;
            console.log("Enhanced Neural MR Bot Application initialized successfully");

            // Add welcome message
            this.addBotMessage(this.getWelcomeMessage());

            return true;
       /* } catch (error) {
            console.error("Error initializing application:", error);
            this.addBotMessage("Error initializing the application. Please refresh the page and try again.");
            return false;
        }*/
    }
    /*
    async initialize() {
        console.log("Initializing Enhanced Neural MR Bot Application...");
        
        try {
            // Check for offline mode
            this.state.offlineMode = !navigator.onLine;
            
            // Load TensorFlow.js
            await this.loadTensorFlow();
            
            // Initialize components
            await this.initializeComponents();
            
            // Set up UI
            this.setupUI();
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Initialize speech recognition and synthesis
            this.initializeSpeech();
            
            this.initialized = true;
            console.log("Enhanced Neural MR Bot Application initialized successfully");
            
            // Add welcome message
            this.addBotMessage(this.getWelcomeMessage());
            
            return true;
        } catch (error) {
            console.error("Error initializing application:", error);
            this.addBotMessage("Error initializing the application. Please refresh the page and try again.");
            return false;
        }
    }
    */
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
            
            // Try to load from local file first (for offline support)
            const loadFromLocal = () => {
                const script = document.createElement('script');
                script.src = 'js/lib/tf.min.js';
                script.async = true;
                
                script.onload = () => {
                    console.log("TensorFlow.js loaded from local file");
                    resolve();
                };
                
                script.onerror = () => {
                    console.error("Failed to load TensorFlow.js from local file, trying CDN...");
                    loadFromCDN();
                };
                
                document.head.appendChild(script);
            };
            
            // Fallback to CDN
            const loadFromCDN = () => {
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.18.0/dist/tf.min.js';
                script.async = true;
                
                script.onload = () => {
                    console.log("TensorFlow.js loaded from CDN");
                    resolve();
                };
                
                script.onerror = () => {
                    const error = new Error("Failed to load TensorFlow.js");
                    console.error(error);
                    reject(error);
                };
                
                document.head.appendChild(script);
            };
            
            // Start with local file if in offline mode, otherwise try CDN first
            if (this.state.offlineMode) {
                loadFromLocal();
            } else {
                loadFromCDN();
            }
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
        
        // Get supported languages
        this.state.supportedLanguages = this.components.dataManager.getSupportedLanguages();
        
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
        this.ui.loadingIndicator = document.getElementById('loading-indicator');
        this.ui.statusDisplay = document.getElementById('status-display');
        
        // Set up language selector (replacing toggle)
        this.setupLanguageSelector();
        
        // Set up speech UI elements
        this.setupSpeechUI();
        
        // Create feedback UI elements
        this.createFeedbackUI();
        
        // Update status display
        this.updateStatusDisplay();
    }

    /**
     * Set up language selector dropdown
     */
    setupLanguageSelector() {
        // Find the language toggle container
        const languageToggleContainer = document.getElementById('language-toggle-container');
        
        if (languageToggleContainer) {
            // Remove existing toggle
            languageToggleContainer.innerHTML = '';
            
            // Create language selector dropdown
            const languageSelector = document.createElement('select');
            languageSelector.id = 'language-selector';
            languageSelector.className = 'bg-white border rounded px-2 py-1 text-sm';
            
            // Add options for all supported languages
            this.state.supportedLanguages.forEach(langCode => {
                const option = document.createElement('option');
                option.value = langCode;
                option.textContent = this.getLanguageName(langCode);
                languageSelector.appendChild(option);
            });
            
            // Set current language
            languageSelector.value = this.state.currentLanguage;
            
            // Add label
            const label = document.createElement('label');
            label.htmlFor = 'language-selector';
            label.className = 'mr-2 text-sm';
            label.textContent = 'Language:';
            
            // Add to container
            languageToggleContainer.appendChild(label);
            languageToggleContainer.appendChild(languageSelector);
            
            // Store reference
            this.ui.languageSelector = languageSelector;
        } else {
            console.error("Language toggle container not found");
        }
    }

    /**
     * Set up speech UI elements
     */
    setupSpeechUI() {
        // Create container for speech buttons
        const speechContainer = document.createElement('div');
        speechContainer.id = 'speech-container';
        speechContainer.className = 'flex items-center space-x-2';
        
        // Create microphone button
        const micButton = document.createElement('button');
        micButton.id = 'mic-button';
        micButton.className = 'bg-primary hover:bg-secondary text-white rounded-full p-2 transition-colors';
        micButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
        `;
        micButton.title = "Speak your message";
        
        // Create speaker button
        const speakerButton = document.createElement('button');
        speakerButton.id = 'speaker-button';
        speakerButton.className = 'bg-primary hover:bg-secondary text-white rounded-full p-2 transition-colors';
        speakerButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
            </svg>
        `;
        speakerButton.title = "Listen to the response";
        
        // Add buttons to container
        speechContainer.appendChild(micButton);
        speechContainer.appendChild(speakerButton);
        
        // Find the input area to add the speech container
        const inputArea = document.querySelector('.border-t.p-4.flex');
        if (inputArea) {
            // Insert before the send button
            inputArea.insertBefore(speechContainer, this.ui.sendButton);
        }
        
        // Store references
        this.ui.micButton = micButton;
        this.ui.speakerButton = speakerButton;
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
     * Initialize speech recognition and synthesis
     */
    initializeSpeech() {
        // Check if browser supports speech recognition
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            this.speech.recognition = new SpeechRecognition();
            this.speech.recognition.continuous = false;
            this.speech.recognition.interimResults = false;
            
            // Set language based on current selection
            this.speech.recognition.lang = this.getBCP47LanguageCode(this.state.currentLanguage);
            
            // Handle recognition results
            this.speech.recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                this.ui.userInput.value = transcript;
                this.handleUserInput();
            };
            
            // Handle recognition end
            this.speech.recognition.onend = () => {
                this.state.isListening = false;
                this.ui.micButton.classList.remove('bg-red-500');
                this.ui.micButton.classList.add('bg-primary');
            };
            
            // Handle recognition errors
            this.speech.recognition.onerror = (event) => {
                console.error("Speech recognition error:", event.error);
                this.showErrorMessage(`Speech recognition error: ${event.error}`);
                this.state.isListening = false;
                this.ui.micButton.classList.remove('bg-red-500');
                this.ui.micButton.classList.add('bg-primary');
            };
        } else {
            console.warn("Speech recognition not supported in this browser");
            this.ui.micButton.disabled = true;
            this.ui.micButton.title = "Speech recognition not supported in this browser";
        }
        
        // Check if browser supports speech synthesis
        if ('speechSynthesis' in window) {
            this.speech.synthesis = window.speechSynthesis;
            
            // Load available voices
            this.loadVoices();
            
            // Some browsers need a delay to load voices
            if (speechSynthesis.onvoiceschanged !== undefined) {
                speechSynthesis.onvoiceschanged = this.loadVoices.bind(this);
            }
        } else {
            console.warn("Speech synthesis not supported in this browser");
            this.ui.speakerButton.disabled = true;
            this.ui.speakerButton.title = "Speech synthesis not supported in this browser";
        }
    }
    
    /**
     * Load available voices for speech synthesis
     */
    loadVoices() {
        if (this.speech.synthesis) {
            this.speech.voices = this.speech.synthesis.getVoices();
            console.log(`Loaded ${this.speech.voices.length} voices for speech synthesis`);
        }
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
        
        // Language selector change
        if (this.ui.languageSelector) {
            this.ui.languageSelector.addEventListener('change', (event) => {
                this.state.currentLanguage = event.target.value;
                this.components.conversationEngine.setTargetLanguage(this.state.currentLanguage);
                
                // Update UI language
                document.documentElement.lang = this.state.currentLanguage;
                this.updatePlaceholder();
                
                // Update speech recognition language
                if (this.speech.recognition) {
                    this.speech.recognition.lang = this.getBCP47LanguageCode(this.state.currentLanguage);
                }
                
                // Update user profile
                this.components.dataManager.updateUserProfile({
                    targetLanguage: this.state.currentLanguage
                });
            });
        }
        
        // Microphone button click
        if (this.ui.micButton) {
            this.ui.micButton.addEventListener('click', () => {
                this.toggleSpeechRecognition();
            });
        }
        
        // Speaker button click
        if (this.ui.speakerButton) {
            this.ui.speakerButton.addEventListener('click', () => {
                if (this.state.lastBotResponse) {
                    this.speakText(this.state.lastBotResponse);
                }
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
        
        // Online/offline events
        window.addEventListener('online', () => {
            this.state.offlineMode = false;
            this.showSuccessMessage("You are back online!");
        });
        
        window.addEventListener('offline', () => {
            this.state.offlineMode = true;
            this.showErrorMessage("You are offline. Some features may be limited.");
        });
    }

    /**
     * Toggle speech recognition
     */
    toggleSpeechRecognition() {
        if (!this.speech.recognition) return;
        
        if (this.state.isListening) {
            // Stop listening
            this.speech.recognition.stop();
            this.state.isListening = false;
            this.ui.micButton.classList.remove('bg-red-500');
            this.ui.micButton.classList.add('bg-primary');
        } else {
            // Start listening
            try {
                this.speech.recognition.start();
                this.state.isListening = true;
                this.ui.micButton.classList.remove('bg-primary');
                this.ui.micButton.classList.add('bg-red-500');
            } catch (error) {
                console.error("Error starting speech recognition:", error);
                this.showErrorMessage("Could not start speech recognition");
            }
        }
    }

    /**
     * Speak text using speech synthesis
     * @param {string} text - Text to speak
     */
    speakText(text) {
        if (!this.speech.synthesis) return;

    // কথা বলা অবস্থায় থাকলে, আগেরটি থামিয়ে নতুনটির জন্য প্রস্তুত হও
    if (this.speech.synthesis.speaking) {
        this.speech.synthesis.cancel();
    }
        // Create utterance
        const utterance = new SpeechSynthesisUtterance(text);
        
        // Set language
        utterance.lang = this.getBCP47LanguageCode(this.state.currentLanguage);
        
        // Find appropriate voice
        const voices = this.speech.voices;
        const langCode = this.state.currentLanguage;
        const langPrefix = langCode.split('-')[0];
        
        // Try to find a voice that matches the language exactly
        let voice = voices.find(v => v.lang.toLowerCase().startsWith(langPrefix));
        
        // If no exact match, try to find a voice with the same language code
        if (!voice) {
            voice = voices.find(v => v.lang.toLowerCase().startsWith(langPrefix));
        }
        
        // If still no match, use the default voice
        if (voice) {
            utterance.voice = voice;
        }
        
        // Set other properties
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        
        // Add events
        utterance.onstart = () => {
            this.state.isSpeaking = true;
            this.ui.speakerButton.classList.remove('bg-primary');
            this.ui.speakerButton.classList.add('bg-green-500');
        };
        
        utterance.onend = () => {
            this.state.isSpeaking = false;
            this.ui.speakerButton.classList.remove('bg-green-500');
            this.ui.speakerButton.classList.add('bg-primary');
        };
        
        utterance.onerror = (event) => {
            console.error("Speech synthesis error:", event);
            this.state.isSpeaking = false;
            this.ui.speakerButton.classList.remove('bg-green-500');
            this.ui.speakerButton.classList.add('bg-primary');
        };
        
        // Speak
        this.speech.synthesis.speak(utterance);
    }

    /**
     * Get BCP-47 language code for speech recognition/synthesis
     * @param {string} langCode - Language code (e.g., 'en', 'bn')
     * @returns {string} - BCP-47 language code
     */
    getBCP47LanguageCode(langCode) {
        // Map of language codes to BCP-47 language codes
        const langMap = {
            'en': 'en-US',
            'bn': 'bn-BD',
            'hi': 'hi-IN',
            'es': 'es-ES',
            'fr': 'fr-FR',
            'de': 'de-DE',
            'ja': 'ja-JP',
            'zh': 'zh-CN',
            'ru': 'ru-RU',
            'ar': 'ar-SA'
        };
        
        return langMap[langCode] || langCode;
    }

    /**
     * Get language name from language code
     * @param {string} langCode - Language code
     * @returns {string} - Language name
     */
    getLanguageName(langCode) {
        const langNames = {
            'en': 'English',
            'bn': 'Bengali (বাংলা)',
            'hi': 'Hindi (हिन्दी)',
            'es': 'Spanish (Español)',
            'fr': 'French (Français)',
            'de': 'German (Deutsch)',
            'ja': 'Japanese (日本語)',
            'zh': 'Chinese (中文)',
            'ru': 'Russian (Русский)',
            'ar': 'Arabic (العربية)'
        };
        
        return langNames[langCode] || langCode;
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
        messageElement.className = 'flex justify-end mb-4 message-animation';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'user-message rounded-lg py-2 px-4 max-w-[70%]';
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
        // Check if chat container exists, if not try to get it or create a fallback
        if (!this.ui.chatContainer) {
            this.ui.chatContainer = document.getElementById('chat-container');
            
            // If still not found, create a fallback container
            if (!this.ui.chatContainer) {
                console.warn("Chat container not found, creating fallback");
                this.ui.chatContainer = document.createElement('div');
                this.ui.chatContainer.id = 'chat-container';
                this.ui.chatContainer.className = 'h-96 overflow-y-auto p-4 space-y-4';
                
                // Try to append to main or body
                const main = document.querySelector('main');
                if (main) {
                    main.appendChild(this.ui.chatContainer);
                } else {
                    document.body.appendChild(this.ui.chatContainer);
                }
            }
        }
        
        const messageElement = document.createElement('div');
        messageElement.className = 'flex mb-4 message-animation';
        
        const avatarElement = document.createElement('div');
        avatarElement.className = 'w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center mr-2';
        avatarElement.textContent = 'MR';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'bot-message rounded-lg py-2 px-4 max-w-[70%]';
        
        // Convert markdown to HTML (simple version)
        const formattedMessage = this.formatMessage(message);
        messageContent.innerHTML = formattedMessage;
        
        messageElement.appendChild(avatarElement);
        messageElement.appendChild(messageContent);
        this.ui.chatContainer.appendChild(messageElement);
        
        // Scroll to bottom
        this.scrollToBottom();
        
        // Add speaker button to message
        if (this.speech.synthesis) {
            const speakButton = document.createElement('button');
            speakButton.className = 'ml-2 text-gray-500 hover:text-primary';
            speakButton.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                </svg>
            `;
            speakButton.title = "Listen to this message";
            speakButton.addEventListener('click', () => {
                this.speakText(message);
            });
            
            messageElement.appendChild(speakButton);
        }
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
            .replace(/^- (.*$)/gm, '<li>$1</li>')
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
            // Submit feedback with metadata
            await this.components.conversationEngine.submitFeedback(
                this.state.lastUserMessage,
                this.state.lastBotResponse,
                correctionText,
                {
                    language: this.state.currentLanguage,
                    difficulty: this.components.dataManager.getUserProfile().proficiencyLevel
                }
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
            const placeholders = {
                'en': 'Type your message...',
                'bn': 'আপনার বার্তা টাইপ করুন...',
                'hi': 'अपना संदेश टाइप करें...',
                'es': 'Escribe tu mensaje...',
                'fr': 'Tapez votre message...',
                'de': 'Geben Sie Ihre Nachricht ein...',
                'ja': 'メッセージを入力...',
                'zh': '输入您的消息...',
                'ru': 'Введите ваше сообщение...',
                'ar': 'اكتب رسالتك...'
            };
            
            this.ui.userInput.placeholder = placeholders[this.state.currentLanguage] || 'Type your message...';
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
                <div><strong>Language:</strong> ${this.getLanguageName(userProfile.targetLanguage)}</div>
                <div><strong>Level:</strong> ${userProfile.proficiencyLevel}</div>
                <div><strong>Mode:</strong> ${this.state.offlineMode ? 'Offline' : 'Online'}</div>
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

- Have conversations in **multiple languages**
- Learn from our interactions
- Improve based on your corrections
- Adapt to your learning needs
- **NEW:** Voice interaction - click the microphone to speak and the speaker to listen

**Try saying:**
- "Hello, how are you?"
- "Let's practice a different language"
- "Teach me some vocabulary"
- "What can you help me with?"

Type a message or click the microphone to get started!
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
