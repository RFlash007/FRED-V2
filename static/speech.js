// Speech-to-Text functionality for F.R.E.D. v2
// Improved implementation using patterns from F.R.E.D. v1
// Uses centralized olliePrint from script.js for consistent theming

class SpeechSystem {
    constructor() {
        this.socket = null;
        this.audioContext = null;
        this.microphone = null;
        this.processor = null;
        this.isRecording = false;
        this.sttEnabled = true;
        this.isListening = false;
        this.isToggling = false;
        
        // UI elements - Updated to match new HTML structure
        this.statusText = document.getElementById('speech-status-text');
        this.statusDot = document.getElementById('speech-status-dot');
        this.toggleButton = document.getElementById('speech-toggle');
        this.toggleText = document.getElementById('speech-toggle-text');
        this.feedbackOverlay = document.getElementById('speech-feedback');
        this.feedbackText = document.getElementById('speech-feedback-text');
        // Note: wake-word-help element doesn't exist in new UI, so we'll handle it gracefully
        this.wakeWordHelp = document.getElementById('wake-word-help');
        
        this.init();
    }
    
    init() {
        // Initialize SocketIO connection
        this.socket = io();
        
        // Set up event listeners
        this.setupSocketEvents();
        this.setupUIEvents();
        
        // Start speech system
        this.initializeSpeech();
    }
    
    setupSocketEvents() {
        // Connection events
        this.socket.on('connect', () => {
            olliePrint('Connected to F.R.E.D. speech system');
            this.updateStatus('Connected', 'online');
        });
        
        this.socket.on('disconnect', () => {
            olliePrint('Disconnected from F.R.E.D.');
            this.updateStatus('Disconnected', 'offline');
            this.stopRecording();
        });
        
        // Status updates
        this.socket.on('status', (data) => {
            olliePrint('Status update:', data.message);
            this.updateStatus(data.message, 'online');
            
            // Only update sttEnabled from server if we're not currently toggling
            // This prevents the server status from overriding our local toggle state
            if (!this.isToggling && data.hasOwnProperty('stt_enabled')) {
                this.sttEnabled = data.stt_enabled !== false;
                this.updateToggleButton();
            }
        });
        
        // Error handling
        this.socket.on('error', (data) => {
            console.error('Speech error:', data.message);
            this.updateStatus('Error: ' + data.message, 'error');
        });
        
        // F.R.E.D. acknowledgments (new v1-inspired feature)
        this.socket.on('fred_acknowledgment', (data) => {
            olliePrint('F.R.E.D. acknowledgment:', data.text);
            this.showFeedback('F.R.E.D. is listening...', 'listening');
            this.isListening = true;
            
            // Show in chat as system message
            this.addChatMessage(data.text, 'assistant');
        });
        
        // Transcription results
        this.socket.on('transcription_result', (data) => {
            olliePrint('Transcription:', data.text);
            
            // Check if this is a wake word activation (silent mode)
            const wakeWords = ['fred', 'hey fred', 'okay fred', 'hi fred', 'excuse me fred', 'fred are you there'];
            const isWakeWord = wakeWords.some(wake => data.text.toLowerCase().includes(wake));
            
            if (isWakeWord && !this.isListening) {
                // Silent wake word activation - just show listening feedback
                olliePrint('Wake word detected - activating silent listening mode');
                this.showFeedback('Listening...', 'listening');
                this.isListening = true;
                // Don't add wake word to chat history
                return;
            }
            
            // Regular transcription - add to chat and process
            this.addChatMessage(data.text, 'user');
            this.hideFeedback();
            this.isListening = false;
        });
        
        // Voice responses
        this.socket.on('voice_response', (data) => {
            if (data.response) {
                // Handle streaming response
                this.updateLastAssistantMessage(data.response);
            }
        });
    }
    
    setupUIEvents() {
        // Toggle button
        this.toggleButton.addEventListener('click', () => {
            this.toggleSpeech();
        });
        
        // ESC key to stop listening
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isListening) {
                this.stopListening();
            }
        });
    }
    
    initializeSpeech() {
        // DISABLED: Web audio capture - backend now uses direct microphone access
        olliePrint('Using direct backend microphone access - web audio disabled');
        this.updateStatus('Backend handling microphone directly', 'online');
        this.startSTT();
        
        // COMMENTED OUT: Web audio processing is now handled by backend
        /*
        // Request microphone permission and set up audio context
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.error('MediaDevices not supported in this browser');
            this.updateStatus('Browser not supported', 'error');
            return;
        }
        
        navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000
            }
        })
            .then((stream) => {
                this.setupAudioProcessing(stream);
                this.startSTT();
            })
            .catch((error) => {
                console.error('Microphone access denied:', error);
                let errorMessage = 'Microphone access denied';
                
                if (error.name === 'NotAllowedError') {
                    errorMessage = 'Microphone permission denied';
                } else if (error.name === 'NotFoundError') {
                    errorMessage = 'No microphone found';
                } else if (error.name === 'NotSupportedError') {
                    errorMessage = 'Browser not supported';
                }
                
                this.updateStatus(errorMessage, 'error');
            });
        */
    }
    
    setupAudioProcessing(stream) {
        // DISABLED: Audio processing now handled by backend directly
        olliePrint('Audio processing disabled - backend handles microphone directly');
        return;
        
        // COMMENTED OUT: Web audio processing
        /*
        // Create audio context
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        this.microphone = this.audioContext.createMediaStreamSource(stream);
        
        // Try to use AudioWorkletNode (modern approach) if available
        if (this.audioContext.audioWorklet) {
            this.setupModernAudioProcessing();
        } else {
            this.setupFallbackAudioProcessing();
        }
        
        olliePrint('Audio processing set up successfully');
        */
    }
    
    async setupModernAudioProcessing() {
        // DISABLED: Modern audio processing - backend handles audio directly
        olliePrint('Modern audio processing disabled - backend handles microphone');
        return;
        
        // COMMENTED OUT: AudioWorklet processing
        /*
        try {
            // Create inline AudioWorklet processor
            const audioWorkletCode = `
                class AudioProcessor extends AudioWorkletProcessor {
                    process(inputs, outputs, parameters) {
                        const input = inputs[0];
                        if (input.length > 0) {
                            const inputData = input[0];
                            // Convert to 16-bit PCM
                            const pcmData = this.convertToPCM16(inputData);
                            // Send to main thread
                            this.port.postMessage({
                                type: 'audioData',
                                data: pcmData
                            });
                        }
                        return true;
                    }
                    
                    convertToPCM16(float32Array) {
                        const buffer = new ArrayBuffer(float32Array.length * 2);
                        const view = new DataView(buffer);
                        
                        for (let i = 0; i < float32Array.length; i++) {
                            const sample = Math.max(-1, Math.min(1, float32Array[i]));
                            view.setInt16(i * 2, sample * 0x7FFF, true);
                        }
                        
                        return buffer;
                    }
                }
                
                registerProcessor('audio-processor', AudioProcessor);
            `;
            
            // Create blob URL for the worklet
            const blob = new Blob([audioWorkletCode], { type: 'application/javascript' });
            const workletUrl = URL.createObjectURL(blob);
            
            await this.audioContext.audioWorklet.addModule(workletUrl);
            
            // Create AudioWorkletNode
            this.processor = new AudioWorkletNode(this.audioContext, 'audio-processor');
            
            // Handle messages from the worklet
            this.processor.port.onmessage = (event) => {
                if (event.data.type === 'audioData' && this.isRecording && this.sttEnabled) {
                    // Send to server
                    this.socket.emit('audio_chunk', {
                        audio: this.arrayBufferToBase64(event.data.data)
                    });
                }
            };
            
            // Connect audio processing
            this.microphone.connect(this.processor);
            this.processor.connect(this.audioContext.destination);
            
            olliePrint('Using modern AudioWorkletNode for audio processing');
            
        } catch (workletError) {
            console.warn('AudioWorkletNode setup failed, falling back to ScriptProcessorNode:', workletError);
            this.setupFallbackAudioProcessing();
        }
        */
    }
    
    setupFallbackAudioProcessing() {
        // DISABLED: Fallback audio processing - backend handles audio directly
        olliePrint('Fallback audio processing disabled - backend handles microphone');
        return;
        
        // COMMENTED OUT: ScriptProcessorNode processing
        /*
        // Fallback to ScriptProcessorNode for older browsers
        console.warn('Falling back to ScriptProcessorNode - consider updating browser for better performance');
        
        this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
        
        this.processor.onaudioprocess = (event) => {
            if (!this.isRecording || !this.sttEnabled) return;
            
            const inputData = event.inputBuffer.getChannelData(0);
            
            // Convert to 16-bit PCM
            const pcmData = this.convertToPCM16(inputData);
            
            // Send to server
            this.socket.emit('audio_chunk', {
                audio: this.arrayBufferToBase64(pcmData)
            });
        };
        
        // Connect audio processing
        this.microphone.connect(this.processor);
        this.processor.connect(this.audioContext.destination);
        */
    }
    
    convertToPCM16(float32Array) {
        const buffer = new ArrayBuffer(float32Array.length * 2);
        const view = new DataView(buffer);
        
        for (let i = 0; i < float32Array.length; i++) {
            const sample = Math.max(-1, Math.min(1, float32Array[i]));
            view.setInt16(i * 2, sample * 0x7FFF, true);
        }
        
        return buffer;
    }
    
    arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
    
    startSTT() {
        if (!this.sttEnabled) return;
        
        this.socket.emit('start_stt');
        this.isRecording = true;
        this.updateStatus('Waiting for wake word...', 'listening');
        olliePrint('Started speech-to-text');
    }
    
    stopSTT() {
        this.socket.emit('stop_stt');
        this.isRecording = false;
        this.isListening = false;
        this.updateStatus('Speech recognition stopped', 'offline');
        this.hideFeedback();
        olliePrint('Stopped speech-to-text');
    }
    
    toggleSpeech() {
        this.isToggling = true;
        
        this.sttEnabled = !this.sttEnabled;
        
        this.socket.emit('toggle_stt', { enabled: this.sttEnabled });
        
        if (this.sttEnabled) {
            this.startSTT();
        } else {
            this.stopSTT();
        }
        
        this.updateToggleButton();
        
        // Reset toggle flag after a short delay to prevent race conditions
        setTimeout(() => {
            this.isToggling = false;
        }, 500);
    }
    
    stopListening() {
        // Send a cancellation signal to stop current listening
        this.socket.emit('voice_message', { text: 'nevermind' });
        this.isListening = false;
        this.hideFeedback();
        this.updateStatus('Cancelled', 'online');
    }
    
    updateStatus(message, status) {
        // Add null checks for all DOM elements
        if (this.statusText) {
            this.statusText.textContent = message;
        }
        
        if (this.statusDot) {
            // Update class names to match new CSS structure
            this.statusDot.className = `voice-status-dot ${status}`;
        }
        
        // Update wake word help visibility (only if element exists)
        if (this.wakeWordHelp) {
            if (status === 'listening' && !this.isListening) {
                this.wakeWordHelp.style.display = 'block';
            } else {
                this.wakeWordHelp.style.display = 'none';
            }
        }
    }
    
    updateToggleButton() {
        if (!this.toggleButton || !this.toggleText) return;
        
        if (this.sttEnabled) {
            this.toggleButton.classList.add('enabled');
            this.toggleButton.classList.remove('disabled');
            this.toggleText.textContent = 'VOICE ON';
            this.toggleButton.title = 'Disable speech recognition';
        } else {
            this.toggleButton.classList.add('disabled');
            this.toggleButton.classList.remove('enabled');
            this.toggleText.textContent = 'VOICE OFF';
            this.toggleButton.title = 'Enable speech recognition';
        }
    }
    
    showFeedback(message, type) {
        if (!this.feedbackText || !this.feedbackOverlay) return;
        
        this.feedbackText.textContent = message;
        this.feedbackOverlay.className = `speech-overlay ${type}`;
        this.feedbackOverlay.classList.remove('hidden');
    }
    
    hideFeedback() {
        if (!this.feedbackOverlay) return;
        this.feedbackOverlay.classList.add('hidden');
    }
    
    addChatMessage(text, sender) {
        // Integration with main chat system using proper message structure
        if (window.appendMessage) {
            // Use the main chat system's message function for proper styling
            window.appendMessage(sender, text);
        } else {
            // Fallback to basic message creation if appendMessage is not available
            const chatHistory = document.getElementById('chat-history');
            if (!chatHistory) return;
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const messageP = document.createElement('p');
            messageP.textContent = text;
            messageDiv.appendChild(messageP);
            
            chatHistory.appendChild(messageDiv);
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
    }
    
    updateLastAssistantMessage(newContent) {
        // Update the last assistant message (for streaming responses)
        const chatHistory = document.getElementById('chat-history');
        if (!chatHistory) return;
        
        const messages = chatHistory.querySelectorAll('.message.assistant');
        if (messages.length > 0) {
            const lastMessage = messages[messages.length - 1];
            const p = lastMessage.querySelector('.message-content p');
            if (p) {
                // Append new content for streaming
                p.textContent += newContent;
                
                // Scroll to bottom
                chatHistory.scrollTo({
                    top: chatHistory.scrollHeight,
                    behavior: 'smooth'
                });
            }
        } else if (newContent.trim()) {
            // Create new assistant message if none exists
            if (window.appendMessage) {
                window.appendMessage('assistant', newContent);
            }
        }
    }
}

// Initialize speech system when page loads
document.addEventListener('DOMContentLoaded', () => {
    olliePrint('Initializing F.R.E.D. speech system...');
    window.speechSystem = new SpeechSystem();
}); 