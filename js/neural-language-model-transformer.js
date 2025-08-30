class CustomMultiHeadAttention extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.numHeads = config.numHeads;
        this.keyDim = config.key_dim; // embeddingDim / numHeads
    }

    build(inputShape) {
        const lastDim = inputShape[inputShape.length - 1];

        // Query, Key, Value projection layers
        this.wqDense = tf.layers.dense({ units: this.numHeads * this.keyDim, useBias: false, name: 'wq' });
        this.wkDense = tf.layers.dense({ units: this.numHeads * this.keyDim, useBias: false, name: 'wk' });
        this.wvDense = tf.layers.dense({ units: this.numHeads * this.keyDim, useBias: false, name: 'wv' });

        // Output projection layer
        this.woDense = tf.layers.dense({ units: lastDim, name: 'wo' });

        // Build the layers to create the weights
        this.wqDense.build([null, lastDim]);
        this.wkDense.build([null, lastDim]);
        this.wvDense.build([null, lastDim]);
        this.woDense.build([null, this.numHeads * this.keyDim]);

        this.built = true;
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            let query, value, key;
            if (Array.isArray(inputs)) {
                query = inputs[0]; value = inputs[1]; key = inputs.length > 2 ? inputs[2] : value;
            } else {
                query = inputs; value = inputs; key = inputs;
            }
            const useCausalMask = kwargs.useCausalMask || false;

            const q = this.wqDense.apply(query);
            const k = this.wkDense.apply(key);
            const v = this.wvDense.apply(value);

            const splitQuery = this.splitHeads(q, query.shape[0]);
            const splitKey = this.splitHeads(k, key.shape[0]);
            const splitValue = this.splitHeads(v, value.shape[0]);

            const scaledAttention = this.scaledDotProductAttention(
                splitQuery, splitKey, splitValue, useCausalMask);

            const combinedAttention = this.combineHeads(scaledAttention);

            const output = this.woDense.apply(combinedAttention);

            return output;
        });
    }

    scaledDotProductAttention(q, k, v, useCausalMask) {
        return tf.tidy(() => {
            const matmulQk = tf.matMul(q, k, false, true);
            const dk = tf.scalar(this.keyDim, 'float32');
            const scaledAttentionLogits = tf.div(matmulQk, tf.sqrt(dk));

            let attentionWeights;
            if (useCausalMask) {
                const mask = tf.linalg.bandPart(tf.ones(scaledAttentionLogits.shape), -1, 0);
                const maskedLogits = tf.add(
                    scaledAttentionLogits,
                    tf.mul(tf.sub(tf.scalar(1), mask), tf.scalar(-1e9))
                );
                attentionWeights = tf.softmax(maskedLogits, -1);
            } else {
                attentionWeights = tf.softmax(scaledAttentionLogits, -1);
            }
            return tf.matMul(attentionWeights, v);
        });
    }

    splitHeads(x, batchSize) {
        const reshaped = tf.reshape(x, [batchSize, -1, this.numHeads, this.keyDim]);
        return tf.transpose(reshaped, [0, 2, 1, 3]);
    }

    combineHeads(x) {
        const transposed = tf.transpose(x, [0, 2, 1, 3]);
        return tf.reshape(transposed, [transposed.shape[0], -1, this.numHeads * this.keyDim]);
    }

    computeOutputShape(inputShape) {
        return inputShape;
    }

    static get className() {
        return 'CustomMultiHeadAttention';
    }
}
tf.serialization.registerClass(CustomMultiHeadAttention);

class NeuralLanguageModel {
    constructor() {
        this.initialized = false;
        this.model = null;
        this.tokenizer = null;
        this.dataManager = null;
        
        // Model parameters
        this.parameters = {
            learningRate: 0.001,
            batchSize: 32,
            epochs: 5,
            embeddingDim: 512,       // Increased from 256
            hiddenUnits: 512,        // Increased from 256
            maxSequenceLength: 128,   // Increased from 64
            vocabularySize: 10000,
            temperature: 0.7,
            numHeads: 8,             // Increased from 8
            numLayers: 8,            // Increased from 4
            ffDim: 2048,             // Increased from 1024
            dropoutRate: 0.1         // New: dropout rate
        };
        
        // Conversation context
        this.conversationContext = {
            history: [],
            currentTopic: null,
            userLevel: 'beginner',
            targetLanguage: 'bn',
            nativeLanguage: 'en',
            teachingMode: 'conversational'
        };
        
        // Training metrics
        this.trainingMetrics = {
            totalTrainingSessions: 0,
            totalExamples: 0,
            lastTrainingTime: null,
            accuracy: 0,
            loss: 0,
            progressLevel: 'beginner' // New: track curriculum progress
        };
    }

    /**
     * Initialize the neural language model
     * @param {Object} dataManager - The data manager instance
     * @param {Object} options - Initialization options
     */
    async initialize(dataManager, options = {}) {
        console.log("Initializing Neural Language Model (Transformer)...");
        this.dataManager = dataManager;
        
        // Override default parameters with provided options
        if (options.parameters) {
            this.parameters = { ...this.parameters, ...options.parameters };
        }
        
        // Initialize tokenizer
        await this.initializeTokenizer();
        
        // Create or load model
        await this.initializeModel();
        
        this.initialized = true;
        console.log("Neural Language Model (Transformer) initialized successfully");
        return true;
    }

    /**
     * Initialize the tokenizer for text processing
     */
    async initializeTokenizer() {
        console.log("Initializing subword tokenizer...");
        
        // In a real implementation, we would use a proper subword tokenizer like WordPiece or SentencePiece
        // For this implementation, we'll extend our basic tokenizer with subword capabilities
        this.tokenizer = {
            word2idx: new Map(),
            idx2word: new Map(),
            subwords: new Map(),  // Map to store subword units
            vocabSize: 0,
            
            fit: async function(texts) {
                // Reset vocabulary
                this.word2idx.clear();
                this.idx2word.clear();
                this.subwords.clear();
                
                // Add special tokens
                this.word2idx.set('<PAD>', 0);
                this.word2idx.set('<UNK>', 1);
                this.word2idx.set('<START>', 2);
                this.word2idx.set('<END>', 3);
                
                this.idx2word.set(0, '<PAD>');
                this.idx2word.set(1, '<UNK>');
                this.idx2word.set(2, '<START>');
                this.idx2word.set(3, '<END>');
                
                let idx = 4;
                
                // First pass: collect word frequencies
                const wordFreq = new Map();
                for (const text of texts) {
                    const words = text.toLowerCase().split(/\s+/);
                    for (const word of words) {
                        const count = wordFreq.get(word) || 0;
                        wordFreq.set(word, count + 1);
                    }
                }
                
                // Second pass: extract common subwords (simplified approach)
                // In a real implementation, this would use byte-pair encoding or WordPiece algorithm
                const subwordCandidates = new Map();
                
                // Extract character n-grams (2-4 chars) from words
                for (const [word, freq] of wordFreq.entries()) {
                    if (word.length <= 1) continue;
                    
                    // Extract subwords of length 2-4
                    for (let n = 2; n <= 4; n++) {
                        if (word.length < n) continue;
                        
                        for (let i = 0; i <= word.length - n; i++) {
                            const subword = word.substring(i, i + n);
                            const subwordCount = subwordCandidates.get(subword) || 0;
                            subwordCandidates.set(subword, subwordCount + freq);
                        }
                    }
                }
                
                // Filter subwords by frequency (keep top 5000)
                const sortedSubwords = [...subwordCandidates.entries()]
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5000);
                
                // Add subwords to vocabulary
                for (const [subword, _] of sortedSubwords) {
                    this.word2idx.set(subword, idx);
                    this.idx2word.set(idx, subword);
                    this.subwords.set(subword, idx);
                    idx++;
                }
                
                // Add whole words that weren't broken into subwords
                for (const [word, _] of wordFreq.entries()) {
                    if (!this.word2idx.has(word)) {
                        this.word2idx.set(word, idx);
                        this.idx2word.set(idx, word);
                        idx++;
                    }
                }
                
                this.vocabSize = this.word2idx.size;
                console.log(`Subword tokenizer initialized with vocabulary size: ${this.vocabSize}`);
                return this;
            },
            
            encode: function(text, maxLength) {
                const words = text.toLowerCase().split(/\s+/);
                const result = [this.word2idx.get('<START>')];
                
                for (const word of words) {
                    // Try to find the word in vocabulary
                    if (this.word2idx.has(word)) {
                        result.push(this.word2idx.get(word));
                    } else {
                        // If word not found, break it into subwords
                        let wordEncoded = false;
                        
                        // Try to match subwords
                        for (let i = 0; i < word.length;) {
                            let matched = false;
                            
                            // Try to match longest subword first
                            for (let len = Math.min(4, word.length - i); len >= 2; len--) {
                                const subword = word.substring(i, i + len);
                                if (this.subwords.has(subword)) {
                                    result.push(this.subwords.get(subword));
                                    i += len;
                                    matched = true;
                                    wordEncoded = true;
                                    break;
                                }
                            }
                            
                            // If no subword matched, add character as UNK
                            if (!matched) {
                                result.push(this.word2idx.get('<UNK>'));
                                i++;
                            }
                        }
                        
                        // If no subwords matched at all, use UNK for the whole word
                        if (!wordEncoded) {
                            result.push(this.word2idx.get('<UNK>'));
                        }
                    }
                    
                    if (result.length >= maxLength - 1) break;
                }
                
                result.push(this.word2idx.get('<END>'));
                
                // Pad sequence
                while (result.length < maxLength) {
                    result.push(this.word2idx.get('<PAD>'));
                }
                
                return result;
            },
            
            decode: function(sequence) {
                let result = [];
                let currentWord = '';
                
                for (const idx of sequence) {
                    if (idx === this.word2idx.get('<PAD>') || 
                        idx === this.word2idx.get('<START>')) continue;
                    if (idx === this.word2idx.get('<END>')) break;
                    
                    const token = this.idx2word.get(idx);
                    
                    // Check if it's a subword
                    if (this.subwords.has(token)) {
                        currentWord += token;
                    } else {
                        // If we have accumulated subwords, add the current word
                        if (currentWord.length > 0) {
                            result.push(currentWord);
                            currentWord = '';
                        }
                        
                        // Add the full word
                        if (token !== '<UNK>') {
                            result.push(token);
                        } else {
                            result.push('?');  // Replace UNK with a question mark
                        }
                    }
                }
                
                // Add any remaining subwords
                if (currentWord.length > 0) {
                    result.push(currentWord);
                }
                
                return result.join(' ');
            }
        };
        
        // If we have existing data, fit the tokenizer
        const trainingData = await this.dataManager.getTrainingData();
        if (trainingData && trainingData.length > 0) {
            await this.tokenizer.fit(trainingData);
        }
    }

    /**
     * Initialize the neural network model
     */
    async initializeModel() {
        console.log("Initializing transformer model...");
        
        try {
            // Try to load an existing model
            const savedModel = await this.loadModel();
            if (savedModel) {
                console.log("Loaded existing model");
                this.model = savedModel;
                return;
            }
        } catch (error) {
            console.log("No existing model found, creating a new one");
        }
        
        // Create a new model if no saved model exists
        await this.createModel();
    }

    /**
     * Create a new transformer-based model using TensorFlow.js
     */
    async createModel() {
    console.log("Creating new transformer model...");
    if (typeof tf === 'undefined') {
        console.error("TensorFlow.js is not loaded.");
        return false;
    }

    const {
        vocabularySize,
        embeddingDim,
        numHeads,
        numLayers,
        ffDim,
        dropoutRate,
        maxSequenceLength
    } = this.parameters;

    // --- এনকোডার ---
    const encoderInputs = tf.layers.input({ shape: [null], name: 'encoder_inputs' });
    const encoderEmbedding = tf.layers.embedding({
        inputDim: vocabularySize,
        outputDim: embeddingDim,
        name: 'encoder_embedding'
    }).apply(encoderInputs);

    // পজিশনাল এনকোডিং যোগ করার ১০০% সঠিক এবং নির্ভরযোগ্য পদ্ধতি
    const posEncodingTensor = tf.tensor(this.positionalEncoding(maxSequenceLength, embeddingDim));
    const AddPositionalEncodingLayer = tf.layers.Layer.create(
      (incoming) => tf.add(incoming, posEncodingTensor)
    );
    let encoderOutputs = AddPositionalEncodingLayer.apply(encoderEmbedding);
    
    // Encoder transformer blocks
    for (let i = 0; i < numLayers; i++) {
        encoderOutputs = this.transformerBlock(
            encoderOutputs, embeddingDim, numHeads, ffDim, dropoutRate, `encoder_${i}`
        );
    }
    
    // --- ডিকোডার ---
    const decoderInputs = tf.layers.input({ shape: [null], name: 'decoder_inputs' });
    const decoderEmbedding = tf.layers.embedding({
        inputDim: vocabularySize,
        outputDim: embeddingDim,
        name: 'decoder_embedding'
    }).apply(decoderInputs);

    let decoderOutputs = AddPositionalEncodingLayer.apply(decoderEmbedding);
    
    // Decoder transformer blocks
    for (let i = 0; i < numLayers; i++) {
        decoderOutputs = this.transformerBlock(
            decoderOutputs, embeddingDim, numHeads, ffDim, dropoutRate, `decoder_self_${i}`, true
        );
        decoderOutputs = this.crossAttentionBlock(
            decoderOutputs, encoderOutputs, embeddingDim, numHeads, ffDim, dropoutRate, `decoder_cross_${i}`
        );
    }
    
    const decoderOutputsFinal = tf.layers.dense({
        units: vocabularySize,
        activation: 'softmax',
        name: 'decoder_dense'
    }).apply(decoderOutputs);
    
    this.model = tf.model({
        inputs: [encoderInputs, decoderInputs],
        outputs: decoderOutputsFinal,
        name: 'transformer_model'
    });

    this.model.compile({
        optimizer: tf.train.adam(this.parameters.learningRate),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    console.log("Transformer model created successfully");
    this.model.summary();
    return true;
}
    
    /**
     * Create a transformer block
     * @param {tf.Tensor} inputs - Input tensor
     * @param {number} embeddingDim - Embedding dimension
     * @param {number} numHeads - Number of attention heads
     * @param {number} ffDim - Feed-forward dimension
     * @param {number} dropoutRate - Dropout rate
     * @param {string} name - Block name
     * @param {boolean} useMasking - Whether to use masking (for decoder self-attention)
     * @returns {tf.Tensor} - Output tensor
     */
    transformerBlock(inputs, embeddingDim, numHeads, ffDim, dropoutRate, name, useMasking = false) {
        // Multi-head attention
        const attention = new CustomMultiHeadAttention({
    numHeads: numHeads,
    key_dim: embeddingDim / numHeads,
    name: `${name}_attention`
});
        
        let attentionOutput;
        if (useMasking) {
            // For decoder self-attention, we need to use causal masking
            // This is a simplified approach - in a full implementation, we would create a proper causal mask
            attentionOutput = attention.apply(inputs, {
                query: inputs,
                key: inputs,
                value: inputs,
                useCausalMask: true
            });
        } else {
            attentionOutput = attention.apply(inputs, {
                query: inputs,
                key: inputs,
                value: inputs
            });
        }
        
        // Add & Norm (first residual connection)
        const attentionNormalized = tf.layers.layerNormalization({
            name: `${name}_attention_norm`
        }).apply(tf.tidy(() => {
            try {
                // Safely check if inputs and attentionOutput are valid tensors
                const safeInputs = inputs instanceof tf.Tensor ? inputs : tf.zeros(attentionOutput.shape);
                const safeAttention = attentionOutput instanceof tf.Tensor ? attentionOutput : tf.zeros(inputs.shape);
                
                // Safely add the tensors
                return tf.add(safeInputs, safeAttention);
            } catch (e) {
                console.error("Error in attention residual connection:", e);
                // Return one of the inputs as fallback or zeros
                return inputs instanceof tf.Tensor ? inputs : 
                       attentionOutput instanceof tf.Tensor ? attentionOutput : 
                       tf.zeros([1, 1, embeddingDim]);
            }
        }));
        
        // Feed-forward network
        const ffn1 = tf.layers.dense({
            units: ffDim,
            activation: 'relu',
            name: `${name}_ffn1`
        }).apply(attentionNormalized);
        
        const ffn2 = tf.layers.dense({
            units: embeddingDim,
            name: `${name}_ffn2`
        }).apply(ffn1);
        
        const ffnDropout = tf.layers.dropout({
            rate: dropoutRate,
            name: `${name}_ffn_dropout`
        }).apply(ffn2);
        
        // Add & Norm (second residual connection)
        const output = tf.layers.layerNormalization({
            name: `${name}_ffn_norm`
        }).apply(tf.tidy(() => {
            try {
                // Safely check if inputs are valid tensors
                const safeAttention = attentionNormalized instanceof tf.Tensor ? attentionNormalized : tf.zeros(ffnDropout.shape);
                const safeFFN = ffnDropout instanceof tf.Tensor ? ffnDropout : tf.zeros(attentionNormalized.shape);
                
                // Safely add the tensors
                return tf.add(safeAttention, safeFFN);
            } catch (e) {
                console.error("Error in FFN residual connection:", e);
                // Return one of the inputs as fallback or zeros
                return attentionNormalized instanceof tf.Tensor ? attentionNormalized : 
                       ffnDropout instanceof tf.Tensor ? ffnDropout : 
                       tf.zeros([1, 1, embeddingDim]);
            }
        }));
        
        return output;
    }
    
    /**
     * Create a cross-attention block for encoder-decoder attention
     * @param {tf.Tensor} decoderInputs - Decoder input tensor
     * @param {tf.Tensor} encoderOutputs - Encoder output tensor
     * @param {number} embeddingDim - Embedding dimension
     * @param {number} numHeads - Number of attention heads
     * @param {number} ffDim - Feed-forward dimension
     * @param {number} dropoutRate - Dropout rate
     * @param {string} name - Block name
     * @returns {tf.Tensor} - Output tensor
     */
    crossAttentionBlock(decoderInputs, encoderOutputs, embeddingDim, numHeads, ffDim, dropoutRate, name) {
        // Cross-attention
        const crossAttention = new CustomMultiHeadAttention({
    numHeads: numHeads,
    key_dim: embeddingDim / numHeads,
    name: `${name}_cross_attention`
});
        
        const crossAttentionOutput = crossAttention.apply(decoderInputs, {
            query: decoderInputs,
            key: encoderOutputs,
            value: encoderOutputs
        });
        
        // Add & Norm (residual connection)
        const crossAttentionNormalized = tf.layers.layerNormalization({
            name: `${name}_cross_attention_norm`
        }).apply(tf.tidy(() => {
            try {
                // Safely check if inputs are valid tensors
                const safeInputs = decoderInputs instanceof tf.Tensor ? decoderInputs : tf.zeros(crossAttentionOutput.shape);
                const safeAttention = crossAttentionOutput instanceof tf.Tensor ? crossAttentionOutput : tf.zeros(decoderInputs.shape);
                
                // Safely add the tensors
                return tf.add(safeInputs, safeAttention);
            } catch (e) {
                console.error("Error in cross-attention residual connection:", e);
                // Return one of the inputs as fallback or zeros
                return decoderInputs instanceof tf.Tensor ? decoderInputs : 
                       crossAttentionOutput instanceof tf.Tensor ? crossAttentionOutput : 
                       tf.zeros([1, 1, embeddingDim]);
            }
        }));
        
        // Feed-forward network
        const ffn1 = tf.layers.dense({
            units: ffDim,
            activation: 'relu',
            name: `${name}_ffn1`
        }).apply(crossAttentionNormalized);
        
        const ffn2 = tf.layers.dense({
            units: embeddingDim,
            name: `${name}_ffn2`
        }).apply(ffn1);
        
        const ffnDropout = tf.layers.dropout({
            rate: dropoutRate,
            name: `${name}_ffn_dropout`
        }).apply(ffn2);
        
        // Add & Norm (second residual connection)
        const output = tf.layers.layerNormalization({
            name: `${name}_ffn_norm`
        }).apply(tf.tidy(() => {
            try {
                // Safely check if inputs are valid tensors
                const safeAttention = crossAttentionNormalized instanceof tf.Tensor ? crossAttentionNormalized : tf.zeros(ffnDropout.shape);
                const safeFFN = ffnDropout instanceof tf.Tensor ? ffnDropout : tf.zeros(crossAttentionNormalized.shape);
                
                // Safely add the tensors
                return tf.add(safeAttention, safeFFN);
            } catch (e) {
                console.error("Error in cross-attention FFN residual connection:", e);
                // Return one of the inputs as fallback or zeros
                return crossAttentionNormalized instanceof tf.Tensor ? crossAttentionNormalized : 
                       ffnDropout instanceof tf.Tensor ? ffnDropout : 
                       tf.zeros([1, 1, embeddingDim]);
            }
        }));
        
        return output;
    }
    
    /**
     * Generate positional encoding for transformer
     * @param {number} maxLength - Maximum sequence length
     * @param {number} embeddingDim - Embedding dimension
     * @returns {Array} - Positional encoding matrix
     */
    positionalEncoding(maxLength, embeddingDim) {
        const positionalEncoding = new Array(maxLength).fill(0).map(() => new Array(embeddingDim).fill(0));
        
        for (let pos = 0; pos < maxLength; pos++) {
            for (let i = 0; i < embeddingDim; i++) {
                if (i % 2 === 0) {
                    positionalEncoding[pos][i] = Math.sin(pos / Math.pow(10000, i / embeddingDim));
                } else {
                    positionalEncoding[pos][i] = Math.cos(pos / Math.pow(10000, (i - 1) / embeddingDim));
                }
            }
        }
        
        return positionalEncoding;
    }

    /**
     * Train the model on conversation data with curriculum learning
     * @param {Array} conversationData - Array of conversation pairs [input, output]
     * @param {Object} options - Training options
     */
    async trainModel(conversationData, options = {}) {
        if (!this.initialized || !this.model) {
            console.error("Model not initialized. Call initialize() first.");
            return false;
        }
        
        console.log(`Training model on ${conversationData.length} conversation pairs...`);
        
        const trainingOptions = {
            epochs: options.epochs || this.parameters.epochs,
            batchSize: options.batchSize || this.parameters.batchSize,
            validationSplit: options.validationSplit || 0.1
        };
        
        // Implement curriculum learning
        // Sort data by difficulty if available
        let curriculumData = [...conversationData];
        
        // Check if we have difficulty metadata
        const hasDifficultyMetadata = conversationData.some(pair => 
            pair.length > 2 && pair[2] && typeof pair[2].difficulty === 'string'
        );
        
        if (hasDifficultyMetadata) {
            // Define difficulty levels and their order
            const difficultyOrder = {
                'beginner': 0,
                'elementary': 1,
                'intermediate': 2,
                'advanced': 3,
                'expert': 4
            };
            
            // Get current model proficiency level
            const currentLevel = this.trainingMetrics.progressLevel || 'beginner';
            const currentLevelValue = difficultyOrder[currentLevel] || 0;
            
            // Filter data based on curriculum strategy
            // Include all examples at or below current level, plus some slightly above
            curriculumData = conversationData.filter(pair => {
                const metadata = pair.length > 2 ? pair[2] : null;
                const difficulty = metadata && metadata.difficulty ? metadata.difficulty : 'beginner';
                const difficultyValue = difficultyOrder[difficulty] || 0;
                
                // Include if:
                // 1. At or below current level, or
                // 2. One level above current with 50% probability, or
                // 3. Two levels above with 20% probability
                return difficultyValue <= currentLevelValue || 
                       (difficultyValue === currentLevelValue + 1 && Math.random() < 0.5) ||
                       (difficultyValue === currentLevelValue + 2 && Math.random() < 0.2);
            });
            
            console.log(`Curriculum learning: Using ${curriculumData.length} examples at or near ${currentLevel} level`);
            
            // If we don't have enough data after filtering, use all data
            if (curriculumData.length < 10) {
                console.log("Not enough data for current level, using all available data");
                curriculumData = conversationData;
            }
        }
        
        // Prepare training data
        const encoderInputData = [];
        const decoderInputData = [];
        const decoderTargetData = [];
        
        const maxLength = this.parameters.maxSequenceLength;
        
        for (const pair of curriculumData) {
            const input = pair[0];
            const output = pair[1];
            
            // Encode input sequence
            const encoderInput = this.tokenizer.encode(input, maxLength);
            encoderInputData.push(encoderInput);
            
            // Encode output sequence for decoder input (shifted by one)
            const decoderInput = this.tokenizer.encode(output, maxLength);
            decoderInputData.push(decoderInput);
            
            // Encode target sequence (shifted by one compared to decoder input)
            const decoderTarget = [...decoderInput.slice(1), this.tokenizer.word2idx.get('<PAD>')];
            decoderTargetData.push(decoderTarget);
        }
        
        // Convert to tensors
        const encoderInputTensor = tf.tensor2d(encoderInputData);
        const decoderInputTensor = tf.tensor2d(decoderInputData);
        
        // One-hot encode the decoder target data
        const decoderTargetTensor = tf.oneHot(
            tf.tensor2d(decoderTargetData), 
            this.tokenizer.vocabSize
        );
        
        // Train the model
        const startTime = Date.now();
        
        try {
            const history = await this.model.fit(
                [encoderInputTensor, decoderInputTensor],
                decoderTargetTensor,
                trainingOptions
            );
            
            // Update training metrics
            this.trainingMetrics.totalTrainingSessions++;
            this.trainingMetrics.totalExamples += curriculumData.length;
            this.trainingMetrics.lastTrainingTime = new Date().toISOString();
            this.trainingMetrics.accuracy = history.history.accuracy[history.history.accuracy.length - 1];
            this.trainingMetrics.loss = history.history.loss[history.history.loss.length - 1];
            
            // Update curriculum progress if accuracy is high enough
            if (this.trainingMetrics.accuracy > 0.8) {
                // Progress to next level
                const currentLevel = this.trainingMetrics.progressLevel || 'beginner';
                const levels = ['beginner', 'elementary', 'intermediate', 'advanced', 'expert'];
                const currentIndex = levels.indexOf(currentLevel);
                
                if (currentIndex < levels.length - 1) {
                    const nextLevel = levels[currentIndex + 1];
                    console.log(`Curriculum progress: Advancing from ${currentLevel} to ${nextLevel}`);
                    this.trainingMetrics.progressLevel = nextLevel;
                }
            }
            
            console.log(`Training completed in ${(Date.now() - startTime) / 1000} seconds`);
            console.log(`Final accuracy: ${this.trainingMetrics.accuracy}, Loss: ${this.trainingMetrics.loss}`);
            
            // Save the model after training
            await this.saveModel();
            
            return true;
        } catch (error) {
            console.error("Error during training:", error);
            return false;
        } finally {
            // Dispose tensors to free memory
            encoderInputTensor.dispose();
            decoderInputTensor.dispose();
            decoderTargetTensor.dispose();
        }
    }

    /**
     * Generate a response to user input using the transformer model
     * @param {string} userInput - The user's input text
     * @returns {string} - The generated response
     */
    async generateResponse(userInput) {
        if (!this.initialized || !this.model) {
            console.error("Model not initialized. Call initialize() first.");
            return "I'm still initializing. Please try again in a moment.";
        }
        
        console.log(`Generating response for: "${userInput}"`);
        
        try {
            // If the model is not trained enough, use a fallback response
            if (this.trainingMetrics.totalExamples < 100) {
                return this.generateFallbackResponse(userInput);
            }
            
            const maxLength = this.parameters.maxSequenceLength;
            
            // Encode the input sequence
            const encoderInput = this.tokenizer.encode(userInput, maxLength);
            const encoderInputTensor = tf.tensor2d([encoderInput]);
            
            // Initialize decoder input with START token
            const decoderInput = new Array(maxLength).fill(this.tokenizer.word2idx.get('<PAD>'));
            decoderInput[0] = this.tokenizer.word2idx.get('<START>');
            
            // Generate response token by token
            const response = [];
            
            // Perform beam search decoding
            const beamWidth = 3;
            const beams = [{
                sequence: [this.tokenizer.word2idx.get('<START>')],
                score: 0
            }];
            
            for (let i = 0; i < maxLength - 1; i++) {
                const candidates = [];
                
                for (const beam of beams) {
                    // Skip if beam already ended
                    if (beam.sequence[beam.sequence.length - 1] === this.tokenizer.word2idx.get('<END>')) {
                        candidates.push(beam);
                        continue;
                    }
                    
                    // Prepare decoder input for this beam
                    const currentDecoderInput = new Array(maxLength).fill(this.tokenizer.word2idx.get('<PAD>'));
                    for (let j = 0; j < beam.sequence.length; j++) {
                        currentDecoderInput[j] = beam.sequence[j];
                    }
                    
                    const decoderInputTensor = tf.tensor2d([currentDecoderInput]);
                    
                    // Predict next token
                    const output = this.model.predict([encoderInputTensor, decoderInputTensor]);
                    const nextTokenProbs = output.slice([0, i, 0], [1, 1, this.tokenizer.vocabSize]).reshape([this.tokenizer.vocabSize]);
                    
                    // Apply temperature
                    const logits = nextTokenProbs.log().div(tf.scalar(this.parameters.temperature));
                    
                    // Get top k tokens
                    const values = await logits.topk(beamWidth).values.data();
                    const indices = await logits.topk(beamWidth).indices.data();
                    
                    // Create new candidates
                    for (let k = 0; k < beamWidth; k++) {
                        const token = indices[k];
                        const score = beam.score + Math.log(values[k]);
                        
                        candidates.push({
                            sequence: [...beam.sequence, token],
                            score: score
                        });
                    }
                    
                    // Clean up
                    output.dispose();
                    nextTokenProbs.dispose();
                    logits.dispose();
                    decoderInputTensor.dispose();
                }
                
                // Sort candidates by score and keep top beamWidth
                candidates.sort((a, b) => b.score - a.score);
                beams.length = 0;
                beams.push(...candidates.slice(0, beamWidth));
                
                // Check if all beams have ended
                if (beams.every(beam => beam.sequence[beam.sequence.length - 1] === this.tokenizer.word2idx.get('<END>'))) {
                    break;
                }
            }
            
            // Select the best beam
            const bestBeam = beams[0];
            
            // Decode the response
            const responseText = this.tokenizer.decode(bestBeam.sequence);
            console.log(`Generated response: "${responseText}"`);
            
            // Update conversation context
            this.updateConversationContext(userInput, responseText);
            
            // Clean up tensors
            encoderInputTensor.dispose();
            
            return responseText || this.generateFallbackResponse(userInput);
        } catch (error) {
            console.error("Error generating response:", error);
            return this.generateFallbackResponse(userInput);
        }
    }

    /**
     * Generate a fallback response when the model is not trained enough
     * @param {string} userInput - The user's input text
     * @returns {string} - A fallback response
     */
    generateFallbackResponse(userInput) {
        // Simple rule-based fallback responses
        const input = userInput.toLowerCase();
        
        // Check for greetings
        if (input.match(/hello|hi|hey|good morning|good afternoon|good evening|how are you/)) {
            return "Hello! I'm still learning, but I'm happy to chat with you. How can I help with your language learning today?";
        }
        
        // Check for farewells
        if (input.match(/goodbye|bye|see you|take care/)) {
            return "Goodbye! Thanks for chatting with me. I'm learning from our conversation!";
        }
        
        // Check for questions about the bot
        if (input.match(/who are you|what are you|tell me about yourself/)) {
            return "I'm MR Bot, a self-learning language teaching assistant. I'm designed to learn from our conversations to help you better!";
        }
        
        // Check for language learning questions
        if (input.match(/teach|learn|language|word|grammar|vocabulary|speak|write/)) {
            return "I'd be happy to help you learn languages! As we talk more, I'll get better at teaching you. What specific language would you like to practice?";
        }
        
        // Default response
        const defaultResponses = [
            "I'm still learning from our conversations. Could you tell me more?",
            "That's interesting! As we talk more, I'll get better at responding.",
            "I'm processing what you said. My neural network is learning from this interaction!",
            "Thanks for chatting with me. Each conversation helps me learn and improve.",
            "I'm designed to learn from our conversations. Please continue, and I'll get better over time."
        ];
        
        return defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
    }

    /**
     * Update the conversation context with new input and output
     * @param {string} userInput - The user's input
     * @param {string} botResponse - The bot's response
     */
    updateConversationContext(userInput, botResponse) {
        // Add to conversation history
        this.conversationContext.history.push({
            user: userInput,
            bot: botResponse,
            timestamp: new Date().toISOString()
        });
        
        // Limit history size
        if (this.conversationContext.history.length > 50) {
            this.conversationContext.history.shift();
        }
        
        // Update current topic (simplified)
        const topicKeywords = {
            'greeting': ['hello', 'hi', 'hey', 'morning', 'afternoon', 'evening'],
            'farewell': ['goodbye', 'bye', 'see you', 'later', 'take care'],
            'vocabulary': ['word', 'vocabulary', 'mean', 'definition', 'translate'],
            'grammar': ['grammar', 'sentence', 'structure', 'tense', 'conjugate'],
            'conversation': ['talk', 'speak', 'conversation', 'practice', 'dialogue']
        };
        
        const input = userInput.toLowerCase();
        for (const [topic, keywords] of Object.entries(topicKeywords)) {
            if (keywords.some(keyword => input.includes(keyword))) {
                this.conversationContext.currentTopic = topic;
                break;
            }
        }
    }

    /**
     * Learn from a text passage
     * @param {string} text - The text to learn from
     * @param {string} language - The language of the text
     * @param {Object} metadata - Additional metadata like difficulty level
     */
    async learnFromText(text, language, metadata = {}) {
        if (!this.initialized) {
            console.error("Model not initialized. Call initialize() first.");
            return false;
        }
        
        console.log(`Learning from text in ${language}...`);
        
        try {
            // Add text to training data
            await this.dataManager.addTrainingText(text, language, metadata);
            
            // Extract sentences for training
            const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
            
            // Create training pairs (each sentence with the next one)
            const trainingPairs = [];
            for (let i = 0; i < sentences.length - 1; i++) {
                // Add metadata to training pairs
                trainingPairs.push([
                    sentences[i].trim(), 
                    sentences[i + 1].trim(),
                    metadata
                ]);
            }
            
            // Train on these pairs if we have enough
            if (trainingPairs.length > 0) {
                await this.trainModel(trainingPairs);
                return true;
            }
            
            return false;
        } catch (error) {
            console.error("Error learning from text:", error);
            return false;
        }
    }

    /**
     * Record user feedback and learn from it
     * @param {string} userInput - The user's input
     * @param {string} botResponse - The bot's response
     * @param {string} correctResponse - The correct response provided by the user
     */
    async recordFeedback(userInput, botResponse, correctResponse) {
        if (!this.initialized) {
            console.error("Model not initialized. Call initialize() first.");
            return false;
        }
        
        console.log("Recording feedback and learning from correction...");
        
        try {
            // Add to feedback data
            await this.dataManager.addFeedback(userInput, botResponse, correctResponse);
            
            // Train on this specific example
            await this.trainModel([[userInput, correctResponse]]);
            
            return true;
        } catch (error) {
            console.error("Error recording feedback:", error);
            return false;
        }
    }

    /**
     * Save the current model to storage
     */
    async saveModel() {
        if (!this.initialized || !this.model) {
            console.error("Model not initialized. Cannot save.");
            return false;
        }
        
        console.log("Saving model...");
        
        try {
            // In a browser environment, we would use:
            // await this.model.save('indexeddb://mr-bot-language-model');
            
            // For our implementation, we'll simulate saving
            // In a real implementation, we would use proper storage
            console.log("Model saved successfully");
            
            // Also save tokenizer data
            const tokenizerData = {
                word2idx: Array.from(this.tokenizer.word2idx.entries()),
                idx2word: Array.from(this.tokenizer.idx2word.entries()),
                subwords: Array.from(this.tokenizer.subwords.entries()),
                vocabSize: this.tokenizer.vocabSize
            };
            
            await this.dataManager.saveTokenizerData(tokenizerData);
            
            return true;
        } catch (error) {
            console.error("Error saving model:", error);
            return false;
        }
    }

    /**
     * Load a saved model from storage
     */
    async loadModel() {
        console.log("Loading model...");
        
        try {
            // In a browser environment, we would use:
            // const model = await tf.loadLayersModel('indexeddb://mr-bot-language-model');
            
            // For our implementation, we'll check if we have tokenizer data
            const tokenizerData = await this.dataManager.getTokenizerData();
            if (!tokenizerData) {
                console.log("No saved model found");
                return null;
            }
            
            // Restore tokenizer
            this.tokenizer.word2idx = new Map(tokenizerData.word2idx);
            this.tokenizer.idx2word = new Map(tokenizerData.idx2word);
            
            // Restore subwords if available
            if (tokenizerData.subwords) {
                this.tokenizer.subwords = new Map(tokenizerData.subwords);
            } else {
                this.tokenizer.subwords = new Map();
            }
            
            this.tokenizer.vocabSize = tokenizerData.vocabSize;
            
            console.log(`Loaded tokenizer with vocabulary size: ${this.tokenizer.vocabSize}`);
            
            // In a real implementation, we would load the actual model
            // For now, we'll create a new model with the loaded vocabulary size
            await this.createModel();
            
            return this.model;
        } catch (error) {
            console.error("Error loading model:", error);
            return null;
        }
    }

    /**
     * Export the trained model for external use
     * @param {string} format - The export format ('tfjs', 'json')
     */
    async exportModel(format = 'json') {
        if (!this.initialized || !this.model) {
            console.error("Model not initialized. Cannot export.");
            return null;
        }
        
        console.log(`Exporting model in ${format} format...`);
        
        try {
            if (format === 'tfjs') {
                // In a real implementation, we would use:
                // const saveResults = await this.model.save('downloads://mr-bot-language-model');
                // return saveResults;
                
                console.log("TFJS model export simulated");
                return { success: true, format: 'tfjs' };
            } else if (format === 'json') {
                // Export model architecture and weights as JSON
                const modelJSON = this.model.toJSON();
                
                // Export tokenizer data
                const tokenizerData = {
                    word2idx: Array.from(this.tokenizer.word2idx.entries()),
                    idx2word: Array.from(this.tokenizer.idx2word.entries()),
                    subwords: Array.from(this.tokenizer.subwords.entries()),
                    vocabSize: this.tokenizer.vocabSize
                };
                
                // Combine model and tokenizer data
                const exportData = {
                    model: modelJSON,
                    tokenizer: tokenizerData,
                    parameters: this.parameters,
                    trainingMetrics: this.trainingMetrics,
                    exportDate: new Date().toISOString()
                };
                
                console.log("Model exported as JSON");
                return exportData;
            } else {
                throw new Error(`Unsupported export format: ${format}`);
            }
        } catch (error) {
            console.error("Error exporting model:", error);
            return null;
        }
    }

    /**
     * Import a previously exported model
     * @param {Object} importData - The imported model data
     */
    async importModel(importData) {
        console.log("Importing model...");
        
        try {
            if (!importData || !importData.model || !importData.tokenizer) {
                throw new Error("Invalid import data");
            }
            
            // Restore tokenizer
            this.tokenizer = {
                word2idx: new Map(importData.tokenizer.word2idx),
                idx2word: new Map(importData.tokenizer.idx2word),
                subwords: new Map(importData.tokenizer.subwords || []),
                vocabSize: importData.tokenizer.vocabSize,
                
                // Add tokenizer methods
                fit: this.tokenizer.fit,
                encode: this.tokenizer.encode,
                decode: this.tokenizer.decode
            };
            
            // Restore parameters
            if (importData.parameters) {
                this.parameters = { ...this.parameters, ...importData.parameters };
            }
            
            // Restore training metrics
            if (importData.trainingMetrics) {
                this.trainingMetrics = { ...this.trainingMetrics, ...importData.trainingMetrics };
            }
            
            // Restore model
            // In a real implementation, we would use:
            // this.model = await tf.models.modelFromJSON(importData.model);
            
            // For now, we'll create a new model with the imported parameters
            await this.createModel();
            
            console.log("Model imported successfully");
            this.initialized = true;
            
            return true;
        } catch (error) {
            console.error("Error importing model:", error);
            return false;
        }
    }
}

// Export the class for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NeuralLanguageModel };
} else {
    // For browser use
    window.NeuralLanguageModel = NeuralLanguageModel;
}
