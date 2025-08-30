/**
 * Neural MR Bot Language Teacher - Demo Script
 * 
 * This script demonstrates the functionality of the neural language model
 * by simulating conversations and training.
 */

// Sample training data
const sampleTrainingData = {
    conversations: [
        // English conversations
        { input: "Hello, how are you?", response: "I'm doing well, thank you! How can I help you learn languages today?" },
        { input: "What languages can you teach?", response: "I can help you learn English and Bengali (বাংলা). Which one would you like to practice?" },
        { input: "I want to learn Bengali", response: "Great choice! Bengali is a beautiful language. Let's start with some basic phrases." },
        { input: "How do you say hello in Bengali?", response: "Hello in Bengali is 'হ্যালো' (hyalo) or more traditionally 'নমস্কার' (nomoshkar)." },
        { input: "Thank you", response: "You're welcome! In Bengali, 'thank you' is 'ধন্যবাদ' (dhonnobad)." },
        
        // Bengali conversations
        { input: "আমার নাম জন", response: "আপনার সাথে দেখা করে ভালো লাগলো, জন! আমি আপনাকে ভাষা শিখতে সাহায্য করতে পারি।" },
        { input: "তুমি কেমন আছো?", response: "আমি ভালো আছি, ধন্যবাদ! আপনি কেমন আছেন?" },
        { input: "আমি ইংরেজি শিখতে চাই", response: "নিশ্চয়ই! আমি আপনাকে ইংরেজি শিখতে সাহায্য করতে পারি। কোন বিষয় দিয়ে শুরু করতে চান?" },
        
        // Grammar examples
        { input: "What is the past tense of 'go'?", response: "The past tense of 'go' is 'went'. For example: 'I went to the store yesterday.'" },
        { input: "How do I form questions in English?", response: "To form questions in English, you typically invert the subject and auxiliary verb. For example: 'You are happy' becomes 'Are you happy?'" },
        
        // Vocabulary examples
        { input: "What are some fruits in Bengali?", response: "Here are some fruits in Bengali: আপেল (apple) = apple, কলা (kola) = banana, আম (aam) = mango, আঙ্গুর (angur) = grape" },
        { input: "Teach me colors in English", response: "Here are some colors in English: red, blue, green, yellow, purple, orange, black, white, brown, pink" }
    ],
    
    texts: {
        en: [
            "Learning a new language opens up new opportunities and provides a new perspective. It allows you to connect with different people and cultures.",
            "English is a West Germanic language that was first spoken in early medieval England. It is now the most widely used language in the world.",
            "Grammar is the set of structural rules governing the composition of clauses, phrases, and words in a natural language."
        ],
        bn: [
            "বাংলা ভাষা (বাঙলা, বাঙ্গলা) একটি ইন্দো-আর্য ভাষা, যা দক্ষিণ এশিয়ার বাঙালি জাতির প্রধান কথ্য ও লেখ্য ভাষা।",
            "বাংলাদেশের জাতীয় ভাষা এবং ভারতের পশ্চিমবঙ্গ, ত্রিপুরা, আসামের বরাক উপত্যকার প্রধান ভাষা বাংলা।",
            "বাংলা বিশ্বের অন্যতম জনপ্রিয় ভাষা এবং মাতৃভাষীর সংখ্যা অনুসারে বিশ্বের ৭ম বৃহত্তম ভাষা।"
        ]
    }
};

/**
 * Demo class to showcase the neural language model functionality
 */
class NeuralModelDemo {
    constructor() {
        this.dataManager = null;
        this.languageModel = null;
        this.conversationEngine = null;
        this.initialized = false;
        
        // Demo state
        this.demoState = {
            currentStep: 0,
            totalSteps: 5,
            trainingComplete: false
        };
        
        // UI elements
        this.ui = {
            demoContainer: null,
            progressBar: null,
            statusText: null,
            demoButton: null,
            outputContainer: null
        };
    }
    
    /**
     * Initialize the demo
     */
    async initialize() {
        console.log("Initializing Neural Model Demo...");
        
        try {
            // Set up UI
            this.setupUI();
            
            // Initialize components
            await this.initializeComponents();
            
            this.initialized = true;
            this.updateStatus("Demo initialized. Click 'Start Demo' to begin.");
            
            return true;
        } catch (error) {
            console.error("Error initializing demo:", error);
            this.updateStatus("Error initializing demo: " + error.message, true);
            return false;
        }
    }
    
    /**
     * Set up UI elements
     */
    setupUI() {
        // Create demo container
        this.ui.demoContainer = document.createElement('div');
        this.ui.demoContainer.className = 'bg-white rounded-lg shadow-md p-4 mb-4';
        this.ui.demoContainer.innerHTML = `
            <h2 class="text-xl font-bold mb-4 text-gray-800">Neural Model Demo</h2>
            
            <div class="mb-4">
                <div class="flex justify-between mb-1">
                    <span class="text-sm text-gray-700">Progress</span>
                    <span class="text-sm text-gray-700" id="demo-progress-text">0%</span>
                </div>
                <div class="w-full bg-gray-200 rounded-full h-2.5">
                    <div id="demo-progress-bar" class="bg-primary h-2.5 rounded-full" style="width: 0%"></div>
                </div>
            </div>
            
            <div id="demo-status" class="p-3 bg-gray-100 rounded mb-4 text-sm">
                Ready to start demo.
            </div>
            
            <button id="demo-button" class="bg-primary hover:bg-secondary text-white py-2 px-4 rounded">
                Start Demo
            </button>
            
            <div id="demo-output" class="mt-4 border-t pt-4 hidden">
                <h3 class="font-bold mb-2">Demo Results</h3>
                <div id="demo-results" class="text-sm"></div>
            </div>
        `;
        
        // Add to document before the footer
        const footer = document.querySelector('footer');
        if (footer) {
            document.body.insertBefore(this.ui.demoContainer, footer);
        } else {
            document.body.appendChild(this.ui.demoContainer);
        }
        
        // Get UI elements
        this.ui.progressBar = document.getElementById('demo-progress-bar');
        this.ui.progressText = document.getElementById('demo-progress-text');
        this.ui.statusText = document.getElementById('demo-status');
        this.ui.demoButton = document.getElementById('demo-button');
        this.ui.outputContainer = document.getElementById('demo-output');
        this.ui.results = document.getElementById('demo-results');
        
        // Add event listener to demo button
        this.ui.demoButton.addEventListener('click', () => {
            if (this.demoState.trainingComplete) {
                this.resetDemo();
            } else {
                this.runDemo();
            }
        });
    }
    
    /**
     * Initialize components
     */
    async initializeComponents() {
        // Initialize data manager
        this.dataManager = new NeuralDataManager();
        await this.dataManager.initialize();
        
        // Initialize language model
        this.languageModel = new NeuralLanguageModel();
        await this.languageModel.initialize(this.dataManager);
        
        // Initialize conversation engine
        this.conversationEngine = new NeuralConversationEngine();
        await this.conversationEngine.initialize(
            this.languageModel,
            this.dataManager
        );
    }
    
    /**
     * Run the demo
     */
    async runDemo() {
        if (!this.initialized) {
            this.updateStatus("Demo not initialized. Please refresh the page.", true);
            return;
        }
        
        this.ui.demoButton.disabled = true;
        this.ui.demoButton.textContent = "Running Demo...";
        
        try {
            // Step 1: Load sample data
            await this.runDemoStep("Loading sample training data...", async () => {
                // Add conversations
                for (const conv of sampleTrainingData.conversations) {
                    await this.dataManager.addConversation(conv.input, conv.response);
                }
                
                // Add texts
                for (const lang in sampleTrainingData.texts) {
                    for (const text of sampleTrainingData.texts[lang]) {
                        await this.dataManager.addTrainingText(text, lang);
                    }
                }
                
                return `Loaded ${sampleTrainingData.conversations.length} conversations and ${
                    sampleTrainingData.texts.en.length + sampleTrainingData.texts.bn.length
                } texts.`;
            });
            
            // Step 2: Train tokenizer
            await this.runDemoStep("Training tokenizer...", async () => {
                const trainingData = await this.dataManager.getTrainingData();
                this.languageModel.tokenizer.fit(trainingData);
                
                return `Tokenizer trained with vocabulary size: ${this.languageModel.tokenizer.vocabSize}`;
            });
            
            // Step 3: Train model on conversations
            await this.runDemoStep("Training model on conversations...", async () => {
                const pairs = await this.dataManager.getConversationPairs();
                
                // Use a smaller subset for the demo to make it faster
                const trainingPairs = pairs.slice(0, 10);
                
                await this.languageModel.trainModel(trainingPairs, { epochs: 2 });
                
                return `Model trained on ${trainingPairs.length} conversation pairs.`;
            });
            
            // Step 4: Generate responses
            await this.runDemoStep("Testing model with sample inputs...", async () => {
                const testInputs = [
                    "Hello",
                    "How are you?",
                    "What languages do you speak?",
                    "Teach me something in Bengali"
                ];
                
                let results = "";
                
                for (const input of testInputs) {
                    const response = await this.languageModel.generateResponse(input);
                    results += `<div class="mb-2"><strong>Input:</strong> ${input}<br><strong>Response:</strong> ${response}</div>`;
                }
                
                return results;
            });
            
            // Step 5: Show model metrics
            await this.runDemoStep("Analyzing model performance...", async () => {
                const metrics = this.languageModel.trainingMetrics;
                
                return `
                    <div class="grid grid-cols-2 gap-2">
                        <div><strong>Training sessions:</strong> ${metrics.totalTrainingSessions}</div>
                        <div><strong>Examples learned:</strong> ${metrics.totalExamples}</div>
                        <div><strong>Last training:</strong> ${metrics.lastTrainingTime || 'Never'}</div>
                        <div><strong>Accuracy:</strong> ${(metrics.accuracy * 100).toFixed(2)}%</div>
                        <div><strong>Loss:</strong> ${metrics.loss.toFixed(4)}</div>
                    </div>
                `;
            });
            
            // Demo complete
            this.demoState.trainingComplete = true;
            this.ui.demoButton.disabled = false;
            this.ui.demoButton.textContent = "Reset Demo";
            this.updateStatus("Demo completed successfully! You can now test the model in the chat above.", false);
            
        } catch (error) {
            console.error("Error running demo:", error);
            this.updateStatus("Error running demo: " + error.message, true);
            this.ui.demoButton.disabled = false;
            this.ui.demoButton.textContent = "Retry Demo";
        }
    }
    
    /**
     * Run a single demo step
     * @param {string} statusText - Status text to display
     * @param {Function} stepFunction - Function to execute for this step
     */
    async runDemoStep(statusText, stepFunction) {
        this.updateStatus(statusText);
        
        // Update progress
        const progress = (this.demoState.currentStep / this.demoState.totalSteps) * 100;
        this.updateProgress(progress);
        
        // Run step function
        const result = await stepFunction();
        
        // Add result to output
        this.addResult(statusText, result);
        
        // Increment step
        this.demoState.currentStep++;
        
        // Update progress again
        const newProgress = (this.demoState.currentStep / this.demoState.totalSteps) * 100;
        this.updateProgress(newProgress);
    }
    
    /**
     * Update status text
     * @param {string} text - Status text
     * @param {boolean} isError - Whether this is an error message
     */
    updateStatus(text, isError = false) {
        if (this.ui.statusText) {
            this.ui.statusText.textContent = text;
            
            if (isError) {
                this.ui.statusText.classList.add('bg-red-100', 'text-red-700');
                this.ui.statusText.classList.remove('bg-gray-100', 'text-gray-700');
            } else {
                this.ui.statusText.classList.remove('bg-red-100', 'text-red-700');
                this.ui.statusText.classList.add('bg-gray-100', 'text-gray-700');
            }
        }
    }
    
    /**
     * Update progress bar
     * @param {number} percent - Progress percentage
     */
    updateProgress(percent) {
        if (this.ui.progressBar) {
            this.ui.progressBar.style.width = `${percent}%`;
        }
        
        if (this.ui.progressText) {
            this.ui.progressText.textContent = `${Math.round(percent)}%`;
        }
    }
    
    /**
     * Add result to output container
     * @param {string} title - Result title
     * @param {string} content - Result content
     */
    addResult(title, content) {
        if (this.ui.results) {
            // Show output container
            this.ui.outputContainer.classList.remove('hidden');
            
            // Create result element
            const resultElement = document.createElement('div');
            resultElement.className = 'mb-4 pb-4 border-b border-gray-200';
            resultElement.innerHTML = `
                <h4 class="font-bold text-primary mb-2">${title}</h4>
                <div class="pl-2 border-l-2 border-gray-300">${content}</div>
            `;
            
            // Add to results container
            this.ui.results.appendChild(resultElement);
        }
    }
    
    /**
     * Reset the demo
     */
    resetDemo() {
        // Reset state
        this.demoState.currentStep = 0;
        this.demoState.trainingComplete = false;
        
        // Reset UI
        this.updateProgress(0);
        this.updateStatus("Ready to start demo.");
        this.ui.demoButton.textContent = "Start Demo";
        
        // Clear results
        if (this.ui.results) {
            this.ui.results.innerHTML = '';
        }
        
        // Hide output container
        this.ui.outputContainer.classList.add('hidden');
        
        // Reset components
        this.dataManager.clearAllData();
        this.initializeComponents();
    }
}

// Initialize demo when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Check if app is already initialized
    if (window.app) {
        // Create and initialize demo
        window.demo = new NeuralModelDemo();
        window.demo.initialize().catch(error => {
            console.error("Failed to initialize demo:", error);
        });
    } else {
        console.error("App not initialized. Demo cannot start.");
    }
});